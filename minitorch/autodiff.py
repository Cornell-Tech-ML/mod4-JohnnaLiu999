from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol, List


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # Convert vals to a list to modify the argument at index `arg`
    vals_forward = list(vals)
    vals_backward = list(vals)

    # Modify the values for forward and backward differences
    vals_forward[arg] += epsilon
    vals_backward[arg] -= epsilon

    # Compute the central difference
    return (f(*vals_forward) - f(*vals_backward)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    """Protocol for Variable-like objects that support automatic differentiation."""

    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative for this variable during the backward pass.

        This method is called to update the derivative associated with this variable
        after the backward computation is performed.

        Parameters
        ----------
        x : Any
            The value to accumulate as the derivative for this variable. The type of
            this value depends on the specific implementation of the Variable.

        Returns
        -------
        None

        """
        ...

    @property
    def unique_id(self) -> int:
        """Returns a unique identifier for this variable.

        This method provides a way to distinguish between different variable instances
        within a computation graph.

        Returns
        -------
        int
            A unique integer identifier for this variable.

        """
        ...

    def is_leaf(self) -> bool:
        """Returns a unique identifier for this variable.

        This method provides a way to distinguish between different variable instances
        within a computation graph.

        Returns
        -------
        int
            A unique integer identifier for this variable.

        """
        ...

    def is_constant(self) -> bool:
        """Checks if this variable is constant (i.e., does not require a derivative).

        A constant variable is not involved in any computation requiring backpropagation,
        so no gradient needs to be computed for it.

        Returns
        -------
        bool
            True if this variable is constant, otherwise False.

        """
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables of this variable in the computation graph.

        The parent variables are those that were involved in the computation that
        produced this variable. This property allows traversal of the computation graph.

        Returns
        -------
        Iterable[Variable]
            An iterable containing the parent variables of this variable.

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute the contribution of this variable to the gradient.

        This method computes how the gradient of the output (with respect to this variable)
        can be decomposed into the gradients of the parent variables using the chain rule.

        Parameters
        ----------
        d_output : Any
            The gradient of the output with respect to this variable.

        Returns
        -------
        Iterable[Tuple[Variable, Any]]
            An iterable of tuples where each tuple contains a parent variable and the
            corresponding contribution to the gradient.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    result: List[Variable] = []

    def visit(var: Variable) -> None:
        """Recursively visits variables in the computation graph and adds them
        to the result list if they are not constant.

        Parameters
        ----------
        var : Variable
            The variable to visit, typically a node in the computation graph.

        Returns
        -------
        None

        """
        if var.unique_id in visited or var.is_constant():
            return
        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    visit(m)
        visited.add(var.unique_id)
        result.insert(0, var)
        # if var.unique_id not in visited:
        #     visited.add(var.unique_id)
        #     # Visit all parents (dependencies) first
        #     for parent in var.parents:
        #         visit(parent)
        #     # After visiting parents, add this variable to the result if it's not constant
        #     if not var.is_constant():
        #         result.append(var)

    visit(variable)

    # Since we want the variables from the right-most (output) to left-most (input),
    # reverse the result to reflect this order.
    return result


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to compute derivatives
    for the leaf nodes.

    Args:
    ----
    variable : Variable
        The variable from which to start backpropagation, typically the output of
        the computation graph.
    deriv : Any
        The initial derivative with respect to the variable, typically the gradient
        of the final output (e.g., 1 if starting backpropagation from a scalar loss).

    Returns:
    -------
    None
        This function doesn't return any value, but it updates the derivative values
        of each leaf node in the computation graph by calling `accumulate_derivative`.

    """
    # Step 1: Topologically sort the computation graph
    topo_order = topological_sort(variable)

    # Step 2: Dictionary to store derivatives for each variable
    derivatives = {}

    # The derivative of the final variable (output) is the provided derivative
    derivatives[variable.unique_id] = deriv

    # Step 3: Propagate derivatives backward through the graph
    for var in topo_order:
        d_var = derivatives[var.unique_id]

        if var.is_leaf():
            var.accumulate_derivative(d_var)

        else:  # skip leaves
            # Apply the chain rule to propagate the gradient to the parents
            for parent, local_grad in var.chain_rule(d_var):
                if parent.is_constant():
                    continue
                derivatives.setdefault(parent.unique_id, 0.0)
                derivatives[parent.unique_id] = (
                    derivatives[parent.unique_id] + local_grad
                )


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the tensors that were saved during the forward pass for use in the backward pass.

        These saved tensors are used for gradient calculations during backpropagation.

        Returns
        -------
        Tuple[Any, ...]
            A tuple containing the saved tensor values, which can be of any type depending
            on the specific operation.

        """
        return self.saved_values
