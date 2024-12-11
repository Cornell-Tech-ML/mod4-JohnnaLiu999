"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from typing import Optional
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

# Comment these out if not yet implemented
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        """Initialize a Tensor.

        Args:
        ----
            v (TensorData): The actual data of the tensor.
            back (Optional[History]): The history of operations used to construct this tensor.
            name (Optional[str]): The name of the tensor for debugging purposes.
            backend (Optional[TensorBackend]): The backend to use for the tensor.

        """
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Set whether the tensor requires gradient computation.

        Args:
        ----
            x (bool): Set to True if the tensor requires gradient computation; False otherwise.

        """
        self.history = History()

    def requires_grad(self) -> bool:
        """Check if the tensor requires a gradient.

        Returns
        -------
        bool
            True if the tensor has a computation history indicating that it requires
            gradient computation; False otherwise.

        """
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Returns
        Converted to numpy array

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a float"""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        """Returns a string representation of the tensor.

        This is a string representation of the tensor in row-major order.

        Returns
        -------
            str
                A string representation of the tensor.

        """
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        """Get the element at index `key`.

        Args:
        ----
            key: Union[int, UserIndex]
                The index of the element to get.

        Returns:
        -------
            float
                The element at index `key`.

        """
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        """Set the element at index `key` to `val`.

        Args:
        ----
            key: Union[int, UserIndex]
                The index of the element to set.
            val: float
                The new value of the element.

        """
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data"""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.


        Args:
        ----
            other : backward tensor (must broadcast with self)

        Returns:
        -------
            Expanded version of `other` with the right derivatives

        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Produce a zero tensor of size `shape` with the same backend.

        Args:
        ----
            shape : Optional[UserShape]
                Shape of the output tensor. If `None`, the shape is the same as the
                input tensor.

        Returns:
        -------
            out : Tensor
                A tensor of zeros with the specified shape and same backend.

        """

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor data info as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detach from backprop"""
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Check if the variable is constant.

        A variable is considered constant if it has no computation history, meaning
        it was not produced by any function and has no gradient information to propagate.

        Returns
        -------
        bool
            True if the variable is constant, False otherwise.

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Get the parent variables of this variable.

        The parent variables are the inputs that were used to compute this variable during
        the forward pass. These are stored in the history of the computation.

        Returns
        -------
        Iterable[Variable]
            An iterable containing the parent variables of this variable.

        Raises
        ------
        AssertionError
            If this variable has no computation history.

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule to compute the contribution of this variable to the gradient.

        This method is part of the interface for Variables that support automatic
        differentiation. It computes the gradient of the output with respect to this variable
        by applying the chain rule. The gradient of the output with respect to this variable
        is decomposed into the gradients of the parent variables by using the chain rule.

        Parameters
        ----------
        d_output : Any
            The gradient of the output with respect to this variable.

        Returns
        -------
        Iterable[Tuple[Variable, Any]]
            An iterable containing tuples, where each tuple consists of a parent variable and
            the corresponding contribution to the gradient.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Run backpropagation from the current Tensor to compute gradients.

        Args:
        ----
        grad_output : Optional[Tensor]
            The gradient of the final output (e.g., loss) with respect to the current
            Tensor. If `None`, default to a tensor with a single element set to 1.0.

        Returns:
        -------
        None
            This function doesn't return any value, but it updates the derivative values
            of each leaf node in the computation graph by calling `accumulate_derivative`.

        """
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        """Element-wise division of `self` by `b`.

        Args:
        ----
            b : TensorLike
                The tensor to divide `self` by.

        Returns:
        -------
            Tensor
                Element-wise division of `self` by `b`.

        """
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        """Element-wise division of `b` by `self`.

        Args:
        ----
            b : TensorLike
                The tensor to divide by `self`.

        Returns:
        -------
            Tensor
                Element-wise division of `b` by `self`.

        """
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    @property
    def shape(self) -> UserShape:
        """Returns
        shape of the tensor

        """
        return self._tensor.shape

    @property
    def dims(self) -> int:
        """Returns
        the number of dimensions in the tensor

        """
        return self._tensor.dims

    # Functions
    @property
    def size(self) -> int:
        """Returns
        the total number of elements in the tensor

        """
        return self._tensor.size

    def zero_grad_(self) -> None:
        """Sets the gradient of the tensor to None.

        This function is used to tell the tensor to release any references to the gradient
        and to set the gradient to None. This is useful when we want to reuse the same
        tensor with different gradients.
        """
        self.grad = None

    def __add__(self, other: TensorLike) -> Tensor:
        """Element-wise addition of `self` and `other`.

        Equivalent to `Add.apply(self, other)`.
        """
        return Add.apply(self, self._ensure_tensor(other))

    def __sub__(self, other: TensorLike) -> Tensor:
        """Element-wise subtraction of `self` and `other`.

        Equivalent to `Add.apply(self, Neg.apply(other))`.
        """
        return Add.apply(self, -self._ensure_tensor(other))

    def __mul__(self, other: TensorLike) -> Tensor:
        """Element-wise multiplication of `self` and `other`.

        Equivalent to `Mul.apply(self, other)`.

        Args:
        ----
        other: The tensor to multiply with this tensor element-wise.

        Returns:
        -------
        A new tensor containing the element-wise product of `self` and `other`.

        """
        return Mul.apply(self, self._ensure_tensor(other))

    def __div__(self, other: TensorLike) -> Tensor:
        """Divides `other` by `self` element-wise.

        Equivalent to `other / self`.

        Args:
        ----
            other: The tensor to divide by `self`.

        Returns:
        -------
            A new tensor with the result of the division.

        """
        other = self._ensure_tensor(other)
        return Mul.apply(other, Inv.apply(self))

    def __lt__(self, other: TensorLike) -> Tensor:
        """Returns a new tensor with elements where `self` is less than `other`.

        Equivalent to `self < other`.

        Args:
        ----
            other: The tensor to compare with.

        """
        # other = self._ensure_tensor(other)
        return LT.apply(self, self._ensure_tensor(other))

    def __eq__(self, other: TensorLike) -> Tensor:  # type: ignore[override]
        """Returns a new tensor with elements where `self` is equal to `other`.

        Equivalent to `self == other`.

        Args:
        ----
            other: The tensor to compare with.

        Returns:
        -------
            A new tensor with the result of the comparison.

        """
        return EQ.apply(self, self._ensure_tensor(other))

    def __gt__(self, other: TensorLike) -> Tensor:
        """Returns a new tensor with elements where `self` is greater than `other`.

        Equivalent to `other < self`.

        Args:
        ----
            other: The tensor to compare with.

        Returns:
        -------
            A new tensor with the result of the comparison.

        """
        return LT.apply(self._ensure_tensor(other), self)

    def __neg__(self) -> Tensor:
        """Returns a new tensor with the negated values of `self`.

        Equivalent to `-self`.
        """
        return Neg.apply(self)

    def __radd__(self, other: TensorLike) -> Tensor:
        """Returns a new tensor with the values of `other` added to `self`.

        Equivalent to `other + self`.

        Args:
        ----
            other: The tensor to add to.

        Returns:
        -------
            A new tensor with the result of the addition.

        """
        return self + other

    def __rmul__(self, other: TensorLike) -> Tensor:
        """Returns a new tensor with the values of `other` multiplied by `self`.

        Equivalent to `other * self`.

        Args:
        ----
            other: The tensor to multiply with.

        Returns:
        -------
            A new tensor with the result of the multiplication.

        """
        return self * other

    def all(self, dim: Optional[int] = None) -> Tensor:
        """Performs an all operation on the tensor.

        If `dim` is `None`, the operation is performed over the flattened tensor.
        Otherwise, the operation is performed over the given dimension.

        Returns
        -------
            A tensor with boolean values indicating whether all elements in the
            tensor are true.

        """
        if dim is None:
            return All.apply(self.view(self.size), self._ensure_tensor(0))
        else:
            return All.apply(self, self._ensure_tensor(dim))

    def is_close(self, other: Tensor) -> Tensor:
        """Checks if two tensors are elementwise close.

        Args:
        ----
            other: The other tensor to check.

        Returns:
        -------
            A tensor with boolean values indicating whether each element is close.

        """
        return IsClose.apply(self, other)

    def sigmoid(self) -> Tensor:
        """Computes the sigmoid activation function.

        The sigmoid activation function maps any real-valued number to a value between 0 and 1.

        Returns
        -------
        Tensor
            The output of the sigmoid activation function.

        """
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Computes the rectified linear unit (ReLU) activation function.

        The ReLU activation function takes the element-wise maximum between the input and zero.

        Returns
        -------
        Tensor
            The output of the ReLU activation function.

        """
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Computes the natural logarithm of the input tensor element-wise.

        The natural logarithm is the logarithm to the base-e, where e is approximately 2.71828.
        This function is often used in the logistic and softmax functions.

        Returns
        -------
        Tensor: The output is a tensor with the same shape as the input, where each
        element is the natural logarithm of the corresponding element in the input tensor.

        """
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Computes the exponential of the input tensor element-wise.

        The exponential of a number is the base-e number (approximately 2.71828)
        raised to that power. This function is often used in the logistic
        and softmax functions.

        Returns
        -------
        Tensor: The output is a tensor with the same shape as the input, where each
        element is the exponential of the corresponding element in the input tensor.

        """
        return Exp.apply(self)

    # Should take an optional dim argument

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Computes the sum of elements in the tensor along a specified dimension.

        If a dimension is specified, sums along that dimension. If no dimension is
        specified, sums all the elements in the tensor.

        Args:
        ----
            dim (Optional[int]): The dimension to reduce. If None, reduces over all elements.

        Returns:
        -------
            Tensor: A tensor containing the sum of elements.

        """
        if dim is None:
            return Sum.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        else:
            return Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Computes the mean of elements in the tensor along a specified dimension.

        If a dimension is specified, computes the mean along that dimension. If no dimension is
        specified, computes the mean over all the elements in the tensor.

        Args:
        ----
            dim (Optional[int]): The dimension to reduce. If None, computes the mean over all elements.

        Returns:
        -------
            Tensor: A tensor containing the mean of elements.

        """
        if dim is None:
            return self.sum() / self.size
        else:
            return self.sum(dim) / self.shape[dim]

    def permute(self, *order: int) -> Tensor:
        """Permutes the dimensions of the tensor according to the given order.

        Args:
        ----
            *order: int
                The permutation order.

        Returns:
        -------
            Tensor
                A new tensor with the permuted dimensions.

        """
        return Permute.apply(self, tensor(list(order)))

    def view(self, *dim: int) -> Tensor:
        """Returns a new tensor with the same data as this tensor, but with the
        specified shape.

        The shape is given as a variable number of arguments. The last argument
        is used as the size of the last dimension, and the second to last
        argument is used as the size of the second to last dimension, and so on.

        If the shape is not specified, this function returns a new tensor with
        the same shape as this tensor.

        Args:
        ----
            *dim: int
                The size of each dimension.

        Returns:
        -------
            Tensor
                A new tensor with the same data as this tensor, but with the
                specified shape.

        """
        return View.apply(self, tensor(list(dim)))
