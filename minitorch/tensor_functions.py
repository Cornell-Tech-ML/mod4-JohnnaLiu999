"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        """Call the backward function and wrap the result in a tuple

        Args:
        ----
            ctx: The context containing saved values from the forward pass.
            grad_out: Gradient of the output.

        Returns:
        -------
            Tuple of gradients of the inputs.

        """
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        """Call the forward function and return the result.

        Args:
        ----
            ctx: The context containing saved values from the forward pass.
            *inps: Input tensors.

        Returns:
        -------
            The result of the forward function.

        """
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the forward pass of negation.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The result of `t1.neg_map()`.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the backward pass of the function.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tensor: The gradient of the input computed during backpropagation.

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the forward pass of Inv.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            t1 (Tensor): Input tensor.

        Returns:
        -------
            Tensor: The result of inverting `t1`.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the backward pass of inv.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tensor: The gradient of the input computed during backpropagation.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Computes the forward pass of addition.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            t1 (Tensor): The first operand.
            t2 (Tensor): The second operand.

        Returns:
        -------
            Tensor: The result of `t1 + t2`.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the backward pass of addition.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradient of the first and second input.

        """
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:  # type: ignore # noqaTensor) -> Tensor:
        """Return 1 if all are true"""
        return a.f.mul_reduce(a, int(dim.item()))


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Computes the forward pass of multiplication.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            a (Tensor): First input.
            b (Tensor): Second input.

        Returns:
        -------
            Tensor: The result of multiplying `a` and `b`.

        """
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the backward pass of multiplication.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradient of the first and second input.

        """
        (x1, x2) = ctx.saved_values
        return grad_output.f.mul_zip(x2, grad_output), grad_output.f.mul_zip(
            x1, grad_output
        )


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Computes the forward pass of sigmoid.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            a (Tensor): Input tensor.

        Returns:
        -------
            Tensor: The result of sigmoid `a`.

        """
        out = a.f.sigmoid_map(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the backward pass of the sigmoid function.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tuple[Tensor]: Gradient of the input.

        """
        sigma: Tensor = ctx.saved_values[0]
        return sigma * (-sigma + 1.0) * grad_output


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Computes the forward pass of ReLU.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            a (Tensor): Input tensor.

        Returns:
        -------
            Tensor: The result of ReLU `a`.

        """
        ctx.save_for_backward(a)
        return a.f.relu_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the backward pass of ReLU.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tuple[Tensor]: Gradient of the input.

        """
        (a,) = ctx.saved_values
        return grad_output.f.relu_back_zip(a, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Computes the forward pass of log.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            a (Tensor): Input tensor.

        Returns:
        -------
            Tensor: The result of log `a`.

        """
        ctx.save_for_backward(a)
        out = a.f.log_map(a)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        """Computes the backward pass of the log function.

        Args:
        ----
            ctx (Context): The context from the forward pass containing saved values.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tuple[Tensor]: Gradient of the input.

        """
        (a,) = ctx.saved_values
        return (grad_output.f.log_back_zip(a, grad_output),)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Computes the forward pass of exp.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            a (Tensor): Input tensor.

        Returns:
        -------
            Tensor: The result of exp `a`.

        """
        out = a.f.exp_map(a)
        ctx.save_for_backward(out)

        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the backward pass of exp.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tuple[Tensor]: Gradient of the input.

        """
        (a,) = ctx.saved_values
        return grad_output.f.mul_zip(a, grad_output)


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Computes the forward pass of sum.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            a (Tensor): Input tensor.
            dim (Tensor): The dimension to reduce.

        Returns:
        -------
            Tensor: The result of sum `a`.

        """
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Computes the backward pass of sum.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradient of the input and the derivative of the loss with respect to the dim.

        """
        a_shape, dim = ctx.saved_values
        return grad_output, 0.0


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Computes the forward pass of less than.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            a (Tensor): First input tensor.
            b (Tensor): Second input tensor.

        Returns:
        -------
            Tensor: The result of less than `a` and `b`.

        """
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the backward pass of the "less than" operation.

        This method computes the gradients of the inputs with respect to the
        output of a "less than" operation, using the stored values in `ctx`.

        Args:
        ----
            ctx (Context): The context containing saved values for backpropagation.
            grad_output (Tensor): The gradient of the output with respect to the next layer.

        Returns:
        -------
            Tuple[Tensor, Tensor]: A tuple containing the gradients with respect to
            the inputs `a` and `b`.

        """
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Computes the forward pass of equality.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            a (Tensor): First input tensor.
            b (Tensor): Second input tensor.

        Returns:
        -------
            Tensor: The result of equality `a` and `b`.

        """
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the backward pass of equality.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tuple[Tensor]: Gradient of the input.

        """
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Computes the forward pass of is close.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            a (Tensor): First input tensor.
            b (Tensor): Second input tensor.

        Returns:
        -------
            Tensor: The result of is close `a` and `b`.

        """
        return a.f.is_close_zip(a, b)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        """Computes the forward pass of permutation.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            a (Tensor): Input tensor.
            order (Tensor): The permutation order.

        Returns:
        -------
            Tensor: The result of permuting `a` according to `order`.

        """
        ctx.save_for_backward(order)
        return a._new(a._tensor.permute(*[int(order[i]) for i in range(order.size)]))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Computes the backward pass of Permute.

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradient of the input and the derivative of the loss with respect to the order.

        """
        order: Tensor = ctx.saved_values[0]
        order2: List[int] = [
            a[0]
            for a in sorted(
                enumerate([order[i] for i in range(order.size)]), key=lambda x: x[1]
            )
        ]
        return grad_output._new(grad_output._tensor.permute(*order2)), 0.0


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Computes the forward pass of the view operation.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            a (Tensor): Input tensor.
            shape (Tensor): Shape to view the tensor as.

        Returns:
        -------
            Tensor: The result of viewing `a` as `shape`.

        Notes:
        -----
            The tensor must be contiguous to view.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            grad_output.zeros((1,)),
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        """Return the shape of the tensor data.

        Args:
        ----
            ls (Any): The tensor data to determine the shape of.

        Returns:
        -------
            List[int]: The shape of the tensor data.

        """
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        """Recursively flattens a nested list or tuple into a list of floats.

        Args:
        ----
            ls (Any): The nested list or tuple to be flattened.

        Returns:
        -------
            List[float]: A flat list containing all elements from the nested structure.

        """
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute the derivative of a function `f` with respect to the `arg`-th input
    at index `ind` using the central difference method.

    Args:
    ----
        f (Callable): The function to compute the derivative of.
        *vals (Tensor): The input tensors to the function `f`.
        arg (int): The input number to compute the derivative with respect to.
        epsilon (float): The step size for computing the central difference.
        ind (UserIndex): The index of the input tensor to compute the derivative at.

    Returns:
    -------
        float: The derivative of `f` with respect to the `arg`-th input at index `ind`.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()

    random.seed(10)
    out = f(*vals)

    for i, x in enumerate(vals):
        print(f"Tensor {i} requires_grad: {x.requires_grad}")
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        print("HERE IS x.grad: for fn", f, " :", x.grad)
        assert (
            x.grad is not None
        ), f"Gradient is None for function {f}, input {vals}, argument {i}, tensor: {x}"
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
