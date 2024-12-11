from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies the forward function of a ScalarFunction subclass and returns a new `Scalar`
        instance with the result. This method manages the conversion of inputs into `Scalar`
        objects, calls the `_forward` function, and constructs the history for backpropagation.

        Args:
        ----
            *vals (ScalarLike): Variable-length argument list containing values that can
                either be instances of `minitorch.scalar.Scalar` or any type that can
                be converted into a `Scalar`.

        Returns:
        -------
            Scalar: A new `Scalar` instance containing the result of the forward function
            and a history object for backpropagation.

        Raises:
        ------
            AssertionError: If the return value of the `_forward` method is not a float.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass for addition.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            a (float): The first input.
            b (float): The second input.

        Returns:
        -------
            float: The result of adding `a` and `b`.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the gradient of addition with respect to both inputs.

        The derivative of addition is 1 for both `a` and `b`, so the gradient is
        simply the upstream gradient (`d_output`) passed to both inputs.

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, float]: The gradients with respect to `a` and `b` (both equal to `d_output`).

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass for the natural logarithm.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            a (float): The input value for which to compute the natural logarithm.

        Returns:
        -------
            float: The result of `log(a)`.

        Raises:
        ------
            ValueError: If `a` is less than or equal to 0.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform the backward pass for the logarithm function.

        Args:
        ----
            ctx (Context): The context from the forward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: The gradient with respect to `a`.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function f(x, y) = x * y"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of multiplication.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            a (float): First input.
            b (float): Second input.

        Returns:
        -------
            float: The result of multiplying `a` and `b`.

        """
        ctx.save_for_backward(a, b)
        c = a * b
        return c

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Perform the backward pass for multiplication.

        Args:
        ----
            ctx (Context): The context from the forward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, float]: Gradients with respect to `a` and `b`.

        """
        a, b = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inverse function f(x) = 1 / x"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of inversion.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            a (float): Input value.

        Returns:
        -------
            float: The result of `1 / a`.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform the backward pass for the inverse function.

        Args:
        ----
            ctx (Context): The context from the forward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float]: Gradient with respect to `a`.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)
        # return d_output * (-1 / (a * a))


class Neg(ScalarFunction):
    """Negation function f(x) = -x"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of negation.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            a (float): Input value.

        Returns:
        -------
            float: The result of `-a`.

        """
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform the backward pass for negation.

        Args:
        ----
            ctx (Context): The context from the forward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: Gradient with respect to `a`.

        """
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function f(x) = 1 / (1 + exp(-x))"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the sigmoid function.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            a (float): Input value.

        Returns:
        -------
            float: The result of the sigmoid function.

        """
        sig_a = operators.sigmoid(a)
        ctx.save_for_backward(sig_a)
        return sig_a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform the backward pass for the sigmoid function.

        Args:
        ----
            ctx (Context): The context from the forward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: Gradient with respect to `a`.

        """
        sigma: float = ctx.saved_values[0]
        return sigma * (1.0 - sigma) * d_output


class ReLU(ScalarFunction):
    """ReLU function f(x) = max(0, x)"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the ReLU function.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            a (float): Input value.

        Returns:
        -------
            float: The result of ReLU function.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform the backward pass for the ReLU function.

        Args:
        ----
            ctx (Context): The context from the forward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float]: Gradient with respect to `a`.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function f(x) = exp(x)"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the exponential function.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            a (float): Input value.

        Returns:
        -------
            float: The result of exp(a).

        """
        out = operators.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform the backward pass for the exponential function.

        Args:
        ----
            ctx (Context): The context from the forward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float]: Gradient with respect to `a`.

        """
        out: float = ctx.saved_values[0]
        return d_output * out


class LT(ScalarFunction):
    """Less than function f(x, y) = x < y"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of the less-than comparison.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            a (float): First input.
            b (float): Second input.

        Returns:
        -------
            float: 1.0 if `a < b`, otherwise 0.0.

        """
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Returns zero gradients for less-than comparisons as they are non-differentiable.

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, float]: Zero gradients for both inputs.

        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function f(x, y) = x == y"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of the equality comparison.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            a (float): First input.
            b (float): Second input.

        Returns:
        -------
            float: 1.0 if `a == b`, otherwise 0.0.

        """
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Returns zero gradients for equality comparisons as they are non-differentiable.

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, float]: Zero gradients for both inputs.

        """
        return 0.0, 0.0
