# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from minitorch.tensor import Tensor
from minitorch.tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from minitorch.tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT compile a function for CUDA device execution.

    Args:
    ----
        fn (Fn): The Python function to compile.
        **kwargs: Additional arguments for the Numba JIT compiler.

    Returns:
    -------
        Fn: A device-compiled version of the input function.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """JIT compile a CUDA kernel function.

    Args:
    ----
        fn (Callable): The Python function to compile.
        **kwargs: Additional arguments for the Numba JIT compiler.

    Returns:
    -------
        FakeCUDAKernel: A compiled CUDA kernel.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Create a CUDA zip operation for element-wise transformations on two tensors.

        This method compiles a binary function `fn` for execution on CUDA. The compiled
        function is applied element-wise to two tensors using a CUDA kernel.

        Args:
        ----
            fn (Callable[[float, float], float]): A binary function that maps two floats
                                                to a float.

        Returns:
        -------
            Callable[[Tensor, Tensor], Tensor]: A callable that applies the transformation
                                                to two tensors.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)

        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Create a CUDA reduce operation for tensor reduction along a dimension.

        This method compiles a binary reduction function `fn` for execution on CUDA.
        The compiled function reduces a tensor along a specified dimension using the
        CUDA kernel.

        Args:
        ----
            fn (Callable[[float, float], float]): A binary function for reduction.
            start (float): The initial value for the reduction operation.

        Returns:
        -------
            Callable[[Tensor, int], Tensor]: A callable that performs the reduction on
                                            the tensor.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Perform matrix multiplication on two tensors using CUDA.

        This method supports 2D matrix multiplication or batched 3D matrix multiplication.
        Shared memory and CUDA blocks are used for efficient computation.

        Args:
        ----
            a (Tensor): The first input tensor.
            b (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: The resulting tensor from the matrix multiplication.

        Notes:
        -----
        - Both input tensors are reshaped to 3D for compatibility with batched operations.
        - The function ensures the matrix dimensions match the required shape for multiplication.

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            out[index_to_position(out_index, out_strides)] = fn(
                in_storage[index_to_position(in_index, in_strides)]
            )

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            a_data = a_storage[index_to_position(a_index, a_strides)]
            b_data = b_storage[index_to_position(b_index, b_strides)]
            out[index_to_position(out_index, out_strides)] = fn(a_data, b_data)

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """Practice CUDA kernel for summing elements within blocks.

    This kernel performs a block-wise sum of an input array `a` and stores
    the results in the output array `out`. Shared memory is used for efficiency.

    Args:
    ----
        out (Storage): Storage for the output tensor.
        a (Storage): Storage for the input tensor.
        size (int): Length of the input tensor `a`.

    Notes:
    -----
    - The input tensor is divided into blocks of size `BLOCK_DIM`.
    - Each block computes the sum of its elements using shared memory.
    - The sum for each block is written to the corresponding cell in `out`.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # TODO: Implement for Task 3.3.
    # raise NotImplementedError("Need to implement for Task 3.3")
    if i < size:
        value = float(a[i])
        cache[pos] = value
        cuda.syncthreads()
    else:
        cache[pos] = 0.0
    if i < size:
        for j in [1, 2, 4, 8, 16]:
            if pos % (j * 2) == 0:
                cache[pos] += cache[pos + j]
                cuda.syncthreads()
        if pos == 0:
            out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Compute the sum of elements in a tensor using a CUDA kernel.

    This function invokes the `_sum_practice` kernel to compute a block-wise
    sum of the input tensor.

    Args:
    ----
        a (Tensor): The input tensor to be summed.

    Returns:
    -------
        TensorData: A tensor containing the block-wise sums.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")
        cache[pos] = reduce_value
        if out_pos < out_size:
            to_index(out_pos, out_shape, out_index)
            dim = a_shape[reduce_dim]
            out_index[reduce_dim] = out_index[reduce_dim] * BLOCK_DIM + pos

            if out_index[reduce_dim] < dim:
                cache[pos] = a_storage[index_to_position(out_index, a_strides)]
                cuda.syncthreads()
                idx = 0
                while 2**idx < BLOCK_DIM:
                    if pos % ((2**idx) * 2) == 0:
                        cache[pos] = fn(cache[pos], cache[pos + (2**idx)])
                        cuda.syncthreads()
                    idx += 1
            if pos == 0:
                out[index_to_position(out_index, out_strides)] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """__mm_practice is a CUDA matrix multiplication kernel.

    This kernel assumes that the input and output matrices are in row-major
    order. The kernel divides the matrix into blocks of size BLOCK_DIM x
    BLOCK_DIM and performs the matrix multiplication block by block.

    The kernel uses shared memory to store the tiles of the input matrices,
    and uses registers to store the partial results of the multiplication.

    The kernel also uses synchronization barriers to ensure that all threads
    in the block have finished loading the data before performing the
    multiplication.

    Parameters
    ----------
    out : Storage
        Storage for the output matrix.
    a : Storage
        Storage for the input matrix A.
    b : Storage
        Storage for the input matrix B.
    size : int
        Size of the matrix (number of rows and columns).

    Returns
    -------
    None

    """
    BLOCK_DIM = 32  # Define the block size for shared memory tiles

    # TODO: Implement for Task 3.3.
    # raise NotImplementedError("Need to implement for Task 3.3")
    # Allocate shared memory for tiles of matrix A and B
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Get thread indices within the block
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y

    # If the thread indices exceed the matrix size, return early (out of bounds)
    if i >= size or j >= size:
        return

    # Moving data to shared memory
    a_shared[i, j] = a[size * i + j]
    b_shared[i, j] = b[size * i + j]
    cuda.syncthreads()

    # Initialize an accumulator for the result of this thread
    accum = 0.0

    # Perform the dot product for the current row (i) and column (j)
    for k in range(size):
        accum += a_shared[i, k] * b_shared[k, j]

    # Write the result to the output matrix in global memory
    out[size * i + j] = accum


# Compile the CUDA kernel with Numba
jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """CUDA kernel for matrix multiplication using shared memory.

    This kernel performs block-based matrix multiplication using tiles stored
    in shared memory for efficient computation.

    Args:
    ----
        out (Storage): Storage for the output matrix.
        a (Storage): Storage for the first input matrix.
        b (Storage): Storage for the second input matrix.
        size (int): Number of rows and columns in the square matrices.

    Notes:
    -----
    - Input and output matrices are assumed to be in row-major order.
    - The kernel divides matrices into tiles of size `BLOCK_DIM x BLOCK_DIM`.
    - Synchronization barriers ensure correctness during computation.

    """
    (size, _) = a.shape  # Extract the size of the square matrix
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)  # Set threads per block
    blockspergrid = 1  # Single block for simplicity (for now)

    # Create an output tensor and transfer it to CUDA
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    # Launch the compiled CUDA kernel with grid and block dimensions
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    # Determine batch strides for A and B to handle broadcasting
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    # Define the block size for shared memory tiles
    BLOCK_DIM = 32
    # Allocate shared memory for tiles of matrix A and B
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Compute the global thread position (i, j) in the output matrix
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # Compute the local thread position (pi, pj) within the block
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.
    # raise NotImplementedError("Need to implement for Task 3.4")
    # Initialize an accumulator for the dot product result
    accum = 0.0

    # Iterate over tiles of the shared dimension
    for idx in range(0, a_shape[2], BLOCK_DIM):
        # Compute the global column index for A and B
        k = idx + pj
        # Load tiles of A into shared memory (if within bounds)
        if i < a_shape[1] and k < a_shape[2]:
            # We get the absolute value in a_storage by multiplying the batch dimension and indices with the strides
            a_shared[pi, pj] = a_storage[
                a_batch_stride * batch + a_strides[1] * i + a_strides[2] * k
            ]
        # Load tiles of B into shared memory (if within bounds)
        k = idx + pi
        # j and k must be within the shape. b has shape [batch, k, j]
        if j < b_shape[2] and k < b_shape[1]:
            # Getting absolute value in b_storage by multiplying the batch dimension and indices with the strides
            b_shared[pi, pj] = b_storage[
                b_batch_stride * batch + b_strides[2] * j + b_strides[1] * k
            ]
        # Synchronize threads to ensure all data is loaded
        cuda.syncthreads()
        # Perform the matrix multiplication for this tile
        for k in range(BLOCK_DIM):
            if (idx + k) < a_shape[2]:
                accum += a_shared[pi, k] * b_shared[k, pj]
    # Write the final result to the output matrix (if within bounds)
    if i < out_shape[1] and j < out_shape[2]:
        # We find the absolute position in out storage by multiplying the strides with the batch dimension and the indices and set it to accum
        out[out_strides[0] * batch + out_strides[1] * i + out_strides[2] * j] = accum


# Compile the CUDA kernel with Numba
tensor_matrix_multiply = jit(_tensor_matrix_multiply)
