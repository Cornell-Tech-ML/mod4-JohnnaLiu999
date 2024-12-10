"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

# from responses import start


def mul(x: float, y: float) -> float:
    """Multiply `x` by `y`"""
    return x * y


def id(x: float) -> float:
    """Return input `x` unchanged"""
    return x


def add(x: float, y: float) -> float:
    """Add `x` and `y`"""
    return x + y


def neg(x: float) -> float:
    """Negate `x`"""
    return -x


def lt(x: float, y: float) -> float:
    """Return True if `x` is less than `y`, otherwise returns False"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Return True if `x` is equal to  `y`, otherwise returns False"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the maximum value between `x` and `y`"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Check if floats `x` and `y` are within 1e-2"""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Calculate the sigmoid activation function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU (Rectified Linear Unit) activation function.

    Args:
    ----
        x (Number): The input value.

    Returns:
    -------
        Number: The result of the ReLU function, which is max(0, x).

    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Calculate the natural logarithm of `x`. Raises ValueError if `x` <= 0."""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Calculate the exponential function of `x` (returns e^x)."""
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Compute the derivative of the natural logarithm function, scaled by a second argument.

    Args:
    ----
        x (Number): The input value, must be positive.
        d (Number): The scaling factor for the derivative.

    Returns:
    -------
        float: The scaled derivative of the log function, calculated as d / x.

    Raises:
    ------
        ValueError: If x is less than or equal to 0, as the derivative is undefined for non-positive numbers.

    Example:
    -------
        >>> log_back(2, 3)
        1.5

    """
    return d / (x + EPS)


def inv(x: float) -> float:
    """Calculate the reciprocal x (1/x). Raises ValueError if `x` is 0."""
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Compute the derivative of the reciprocal function, scaled by a second argument.

    Args:
    ----
        x (float): The input value, must not be 0.
        d (float): The scaling factor for the derivative.

    Returns:
    -------
        float: The scaled derivative of the reciprocal function, calculated as -d / x^2.

    Raises:
    ------
        ValueError: If x is equal to 0, as the derivative is undefined.

    Example:
    -------
        >>> inv_back(2, 3)
        -0.75

    """
    return -(1.0 / x**2) * d


def relu_back(x: float, d: float) -> float:
    """Compute the derivative of the ReLU function, scaled by a second argument.

    Args:
    ----
        x (float): The input value.
        d (float): The scaling factor for the derivative.

    Returns:
    -------
        Number: The scaled derivative of the ReLU function. Returns d if x > 0, otherwise 0.

    Example:
    -------
        >>> relu_back(3, 2)
        2
        >>> relu_back(-1, 2)
        0

    """
    return d if x > 0 else 0.0


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order tensor map function

    Args:
    ----
        fn: Funciton from one vlaue to one value

    Returns:
    -------
        A funciton that takes a list, applied 'fn' to each elmenet, and returns a new list.

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipwith (or map2).

    Args:
    ----
        fn: Function that takes two floats and returns a float.

    Returns:
    -------
        A function that takes two equally sized lists (lst1, lst2) and produces a new list applyigng fn(x, y) on each pair of elements)

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Reduces an iterable to a single value using a given function.

    Args:
    ----
        fn: A function that takes two floats and returns a float.
            This function will be applied cumulatively to the elements of the iterable.
        lst: An iterable of floats to be reduced.
        start: The initial value for the reduction.

    Returns:
    -------
        A float that is the result of reducing the iterable using the function `fn`.

    Raises:
    ------
        TypeError: If the iterable `lst` is empty.

    Example:
    -------
        >>> reduce(lambda x, y: x + y, [1.0, 2.0, 3.0, 4.0])
        10.0

    """

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


def negList(lst: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list."""
    # return map(lambda x: -x, lst)
    return map(neg)(lst)


def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists."""
    # return zipWith(lambda x, y: x + y, lst1, lst2)
    return zipWith(add)(lst1, lst2)


def sum(lst: Iterable[float]) -> float:
    """Sum all elements in a list."""
    # if not lst:
    #     return 0.0  # Return 0 for an empty list
    # return reduce(lambda x, y: x + y, lst)
    return reduce(add, 0.0)(lst)


def prod(lst: Iterable[float]) -> float:
    """Calculate the product of all elements in a list."""
    return reduce(mul, 1.0)(lst)
