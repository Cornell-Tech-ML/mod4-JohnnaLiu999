import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generate a list of N random 2D points.

    Args:
    ----
        N (int): The number of 2D points to generate.

    Returns:
    -------
        list of tuple: A list containing N tuples, where each tuple consists of two random floats
        representing the x and y coordinates of a point.

    Example:
    -------
        >>> make_pts(3)
        [(0.314, 0.895), (0.501, 0.023), (0.754, 0.627)]

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generate a graph with N random 2D points and binary labels using a simple rule (y=1 when x < .5 and 0 otherwise).

    Args:
    ----
        N (int): The number of 2D points to generate.

    Returns:
    -------
        Graph: A Graph object containing N points (X) and their corresponding binary labels (y).

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generate a graph with N random 2D points and binary labels using a diagonal rule (y=1 when x_1, x_2 < .5, 0 otherwise).

    Args:
    ----
        N (int): The number of 2D points to generate.

    Returns:
    -------
        Graph: A Graph object containing N points (X) and their corresponding binary labels (y).

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generate a graph with N random 2D points and binary labels using a split rule (y=1 when x_1 < .2 or > .8, 0 otherwise).

    Args:
    ----
        N (int): The number of 2D points to generate.

    Returns:
    -------
        Graph: A Graph object containing N points (X) and their corresponding binary labels (y).

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generate a graph with N random 2D points and binary labels using an XOR rule (1 when x_1 < .5 and x_2 < .5 or x_1 > .5 and x_2 < .5, 0 otherwise).

    Args:
    ----
        N (int): The number of 2D points to generate.

    Returns:
    -------
        Graph: A Graph object containing N points (X) and their corresponding binary labels (y).

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generate a graph with N random 2D points and binary labels using a circular rule (y = 1 when within a circle with radius sqrt(.1), centered at (.5 ,.5), 0 otherwise ).

    Args:
    ----
        N (int): The number of 2D points to generate.

    Returns:
    -------
        Graph: A Graph object containing N points (X) and their corresponding binary labels (y).

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generate a graph with N random 2D points and binary labels using a spiral rule (y=1 when within one spiral, y=0 within another).

    Args:
    ----
        N (int): The number of 2D points to generate.

    Returns:
    -------
        Graph: A Graph object containing N points (X) and their corresponding binary labels (y).

    """

    def x(t: float) -> float:
        """Compute the x-coordinate for a point on the spiral.

        Args:
        ----
            t (float): The input parameter, typically representing time or angle.

        Returns:
        -------
            float: The x-coordinate of the point on the spiral.

        """
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        """Compute the y-coordinate for a point on the spiral.

        Args:
        ----
            t (float): The input parameter, typically representing time or angle.

        Returns:
        -------
            float: The y-coordinate of the point on the spiral.

        """
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
