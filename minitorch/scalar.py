from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from dataclasses import field
from .autodiff import Context, Variable, backpropagate, central_difference

from .scalar_functions import (
    Inv,
    Mul,
    Add,
    Neg,
    Log,
    Exp,
    ReLU,
    Sigmoid,
    LT,  # Using Lt for both __lt__ and __gt__
    ScalarFunction,
    EQ,
)


ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes
    ----------
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


_var_count = 0


@dataclass
class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    data: float
    history: Optional[ScalarHistory] = field(default_factory=ScalarHistory)
    derivative: Optional[float] = None
    name: str = field(default="")
    unique_id: int = field(default=0)

    def __post_init__(self):
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))

    def __repr__(self) -> str:
        return f"Scalar({self.data})"

    def __mul__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, b)

    def __add__(self, b: ScalarLike) -> Scalar:
        """Add `self` to `b`."""
        return Add.apply(self, b)

    def __sub__(self, b: ScalarLike) -> Scalar:
        """Subtract `b` from `self`."""
        return Add.apply(self, Neg.apply(b))

    def __neg__(self) -> Scalar:
        """Negate `self`."""
        return Neg.apply(self)

    def __lt__(self, b: ScalarLike) -> Scalar:
        """Return `True` if `self` is less than `b`."""
        return LT.apply(self, b)

    def __gt__(self, b: ScalarLike) -> Scalar:
        """Return `True` if `self` is greater than `b`, using `lt`."""
        return LT.apply(b, self)

    def __eq__(self, b: ScalarLike) -> Scalar:
        """Return `True` if `self` is equal to `b`, using `eq`."""
        return EQ.apply(self, b)

    def log(self) -> Scalar:
        """Return the natural log of `self`."""
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Return the exponential of `self`."""
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Apply sigmoid function to `self`."""
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """Apply Relu function to `self`."""
        return ReLU.apply(self)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(b, Inv.apply(self))

    def __bool__(self) -> bool:
        return bool(self.data)

    def __radd__(self, b: ScalarLike) -> Scalar:
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        return self * b

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.__setattr__("derivative", 0.0)
        self.__setattr__("derivative", self.derivative + x)

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
        """Applies the chain rule to compute gradients with respect to the inputs.

        This method computes the gradients of the current `Scalar` with respect to its inputs
        by applying the chain rule. It propagates the gradient `d_output` of the final output
        (e.g., a loss function) backward through the operation that created this `Scalar`,
        using the `backward` method of the function stored in the `history`.

        Args:
        ----
            d_output (Any): The gradient of the final output (e.g., loss) with respect to this
                `Scalar`. This is the upstream gradient that gets propagated back through the
                current operation.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: A list of tuples where each tuple consists of an
                input `Variable` and its corresponding gradient with respect to this `Scalar`.
                The gradient is computed by applying the chain rule: the local gradient of the
                operation (with respect to the input) multiplied by `d_output`.

        Raises:
        ------
            AssertionError: If the `Scalar` does not have a history, last function, or context
                saved from the forward pass.

        """
        # Get the history of how this Scalar was created
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        return list(zip(h.inputs, x))

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object.

        Args:
        ----
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks that autodiff works on a Python function.

    Args:
    ----
        f (Callable): A function that takes `n` scalar values as input and returns a single scalar.
        *scalars (Scalar): The input scalar values on which the derivative check is performed.

    Raises:
    ------
        AssertionError: If the computed derivative does not match the expected derivative within the tolerance.

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
