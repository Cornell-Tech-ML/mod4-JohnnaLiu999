from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
    MAX_DIMS,
)

# ops
if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides, Index


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        """Call a map function"""
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Map placeholder"""
        ...

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Zip placeholder"""
        ...

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Apply a reduction operation to the tensor.

        Args:
        ----
            fn (Callable[[float, float], float]): The binary function to reduce with.
            start (float): The initial value for the reduction.

        Returns:
        -------
            Callable[[Tensor, int], Tensor]: A function that reduces the tensor along a specified dimension.

        """
        ...

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiply"""
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
        ----
            ops : tensor operations object see `tensor_ops.py`


        Returns:
        -------
            A collection of tensor functions

        """
        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
        ----
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
        -------
            new tensor data

        """
        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            """Maps a function over the tensor.

            Args:
            ----
                a (Tensor): The tensor to map over.
                out (Optional[Tensor], optional): The output tensor. Defaults to None.

            Returns:
            -------
                Tensor: The output tensor.

            """
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
        -------
            :class:`TensorData` : new tensor data

        """
        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """Higher-order tensor reduce function.

        This function returns a new function (`fn_reduce`) that performs a reduction
        operation over a given dimension of a tensor using the provided binary function `fn`.

        Example:
        -------
            Apply a reduction operation over a tensor using a custom function `fn`:

            ```python
            fn_reduce = reduce(fn)
            out = fn_reduce(a, dim)
            ```

        A simple version of reduction over a 2D tensor:

            ```python
            for j in range(tensor.shape[1]):
                out[1, j] = start
                for i in range(tensor.shape[0]):
                    out[1, j] = fn(out[1, j], a[i, j])
            ```

        Args:
        ----
            fn (Callable[[float, float], float]): A function that takes two floats and returns a float.
            start (float, optional): The starting value for the reduction. Defaults to 0.0.

        Returns:
        -------
            Callable[[Tensor, int], Tensor]: A function that takes a tensor and an integer (dimension)
            and returns a reduced tensor over that dimension.

        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            """Reduce a tensor with the given function.

            Args:
            ----
                a (Tensor): The tensor to reduce
                dim (int): The dimension to reduce over

            Returns:
            -------
                Tensor: The reduced tensor

            """
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        """Matrix multiplication"""
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


# Implementations.


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
    ----
        fn: function from float-to-float to apply

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,  # storage is the 1d array of the items
        in_shape: Shape,  # shape is the 1 dimensional item
        in_strides: Strides,  # conduct the funciton to all of hte items out the out_storage
    ) -> None:
        """Low-level implementation of tensor map between
        tensors with *possibly different strides*.

        Args:
        ----
            out: The output tensor.
            out_shape: The shape of the output tensor.
            out_strides: The strides of the output tensor.
            in_storage: The input tensor storage.
            in_shape: The shape of the input tensor.
            in_strides: The strides of the input tensor.

        Returns:
        -------
            None

        """
        out_idx: Index = np.zeros(MAX_DIMS, dtype=np.int32)
        in_idx: Index = np.zeros(MAX_DIMS, dtype=np.int32)
        # in_idx = np.empty(len(in_shape), dtype=np.int32)
        for out_ordinal_idx in range(len(out)):
            to_index(out_ordinal_idx, out_shape, out_idx)
            broadcast_index(out_idx, out_shape, in_shape, in_idx)
            o = index_to_position(out_idx, out_strides)
            out[o] = fn(in_storage[index_to_position(in_idx, in_strides)])

    return _map


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
    ----
        fn: function mapping two floats to float to apply

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_idx: Index = np.zeros(MAX_DIMS, np.int32)
        a_idx = np.zeros(MAX_DIMS, np.int32)
        b_idx = np.zeros(MAX_DIMS, np.int32)

        for out_ordinal_idx in range(len(out)):
            to_index(out_ordinal_idx, out_shape, out_idx)
            o = index_to_position(out_idx, out_strides)
            broadcast_index(out_idx, out_shape, a_shape, a_idx)
            broadcast_index(out_idx, out_shape, b_shape, b_idx)

            out[o] = fn(
                a_storage[index_to_position(a_idx, a_strides)],
                b_storage[index_to_position(b_idx, b_strides)],
            )

    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
    ----
        fn: reduction function mapping two floats to float

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        """Perform a reduction operation on the input tensor.

        Args:
        ----
            out (Storage): The output storage where the reduced result will be stored.
            out_shape (Shape): The shape of the output tensor.
            out_strides (Strides): The strides for the output tensor.
            a_storage (Storage): The input storage for the tensor to be reduced.
            a_shape (Shape): The shape of the input tensor.
            a_strides (Strides): The strides for the input tensor.
            reduce_dim (int): The dimension along which the reduction is performed.

        Returns:
        -------
            None: This function modifies the `out` storage in place.

        """
        out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
        reduce_size = a_shape[reduce_dim]
        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            for s in range(reduce_size):
                out_index[reduce_dim] = s
                j = index_to_position(out_index, a_strides)
                out[o] = fn(out[o], a_storage[j])

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
