from typing import Tuple, TypeVar, Any
import numpy as np
from numba import cuda
from numba import njit as _njit

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Wrapper for CUDA JIT compilation.

    Args:
    ----
        fn: Function to compile
        **kwargs: Additional arguments to pass to numba.cuda.jit

    Returns:
    -------
        FakeCUDAKernel: Compiled CUDA kernel

    """
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


@cuda.jit
def cuda_kernel_conv1d(
    out: Any,
    out_shape: np.ndarray,
    out_strides: np.ndarray,
    input: Any,
    input_shape: np.ndarray,
    input_strides: np.ndarray,
    weight: Any,
    weight_shape: np.ndarray,
    weight_strides: np.ndarray,
    reverse: bool,
) -> None:
    """Perform 1D convolution using CUDA kernels.

    Given:
        Input:  (batch, in_channels, width)
        Weight: (out_channels, in_channels, kernel_width)

    Produces:
        Output: (batch, out_channels, width)

    Args:
    ----
        out (Any): Output tensor storage on device.
        out_shape (np.ndarray): Shape [batch, out_channels, out_width].
        out_strides (np.ndarray): Strides of the output tensor.
        input (Any): Input tensor storage on device.
        input_shape (np.ndarray): Shape [batch, in_channels, width].
        input_strides (np.ndarray): Strides of the input tensor.
        weight (Any): Weight (kernel) tensor storage on device.
        weight_shape (np.ndarray): Shape [out_channels, in_channels, kernel_width].
        weight_strides (np.ndarray): Strides of the weight tensor.
        reverse (bool): If True, anchor kernel from the right, else from the left.

    Returns:
    -------
        None

    """
    batch_ = out_shape[0]
    out_channels = out_shape[1]
    out_width = out_shape[2]

    in_channels = input_shape[1]
    width = input_shape[2]
    kw = weight_shape[2]

    idx = cuda.grid(1)
    total = batch_ * out_channels * out_width
    if idx >= total:
        return

    b = idx // (out_channels * out_width)
    oc = (idx // out_width) % out_channels
    ow = idx % out_width

    accum = 0.0
    for ic in range(in_channels):
        for k in range(kw):
            iw = ow + k if not reverse else ow - k
            if 0 <= iw < width:
                in_pos = (
                    b * input_strides[0] + ic * input_strides[1] + iw * input_strides[2]
                )
                weight_pos = (
                    oc * weight_strides[0]
                    + ic * weight_strides[1]
                    + k * weight_strides[2]
                )
                accum += input[in_pos] * weight[weight_pos]

    out_pos = b * out_strides[0] + oc * out_strides[1] + ow * out_strides[2]
    out[out_pos] = accum


@cuda.jit
def cuda_kernel_conv2d(
    out: Any,
    out_shape: np.ndarray,
    out_strides: np.ndarray,
    input: Any,
    input_shape: np.ndarray,
    input_strides: np.ndarray,
    weight: Any,
    weight_shape: np.ndarray,
    weight_strides: np.ndarray,
    reverse: bool,
) -> None:
    """Perform 2D convolution using CUDA kernels.

    Given:
        Input:  (batch, in_channels, height, width)
        Weight: (out_channels, in_channels, kh, kw)

    Produces:
        Output: (batch, out_channels, height, width)

    Args:
    ----
        out (Any): Output tensor storage on device.
        out_shape (np.ndarray): Shape [batch, out_channels, out_height, out_width].
        out_strides (np.ndarray): Strides of the output tensor.
        input (Any): Input tensor storage on device.
        input_shape (np.ndarray): Shape [batch, in_channels, height, width].
        input_strides (np.ndarray): Strides of the input tensor.
        weight (Any): Weight (kernel) tensor storage on device.
        weight_shape (np.ndarray): Shape [out_channels, in_channels, kh, kw].
        weight_strides (np.ndarray): Strides of the weight tensor.
        reverse (bool): If True, anchor kernel bottom-right, else top-left.

    Returns:
    -------
        None

    """
    batch_ = out_shape[0]
    out_channels = out_shape[1]
    out_height = out_shape[2]
    out_width = out_shape[3]

    in_channels = input_shape[1]
    height = input_shape[2]
    width = input_shape[3]

    kh = weight_shape[2]
    kw = weight_shape[3]

    total = batch_ * out_channels * out_height * out_width
    idx = cuda.grid(1)
    if idx >= total:
        return

    b = idx // (out_channels * out_height * out_width)
    oc = (idx // (out_height * out_width)) % out_channels
    remainder = idx % (out_height * out_width)
    i = remainder // out_width
    j = remainder % out_width

    accum = 0.0
    for ic in range(in_channels):
        for ki in range(kh):
            for kj in range(kw):
                ih = i + ki if not reverse else i - ki
                iw = j + kj if not reverse else j - kj

                if (0 <= ih < height) and (0 <= iw < width):
                    weight_pos = (
                        oc * weight_strides[0]
                        + ic * weight_strides[1]
                        + ki * weight_strides[2]
                        + kj * weight_strides[3]
                    )
                    in_pos = (
                        b * input_strides[0]
                        + ic * input_strides[1]
                        + ih * input_strides[2]
                        + iw * input_strides[3]
                    )
                    accum += weight[weight_pos] * input[in_pos]

    out_pos = (
        b * out_strides[0]
        + oc * out_strides[1]
        + i * out_strides[2]
        + j * out_strides[3]
    )
    out[out_pos] = accum


def launch_conv1d(out: Tensor, input: Tensor, weight: Tensor, reverse: bool) -> Tensor:
    """Launch the 1D convolution CUDA kernel.

    Args:
    ----
        out (Tensor): Pre-allocated output tensor (batch, out_channels, width).
        input (Tensor): Input tensor (batch, in_channels, width).
        weight (Tensor): Weight tensor (out_channels, in_channels, kernel_width).
        reverse (bool): If True, anchor kernel from the right, else from the left.

    Returns:
    -------
        Tensor: Output tensor after convolution on CPU.

    """
    out_gpu = out.gpu()
    input_gpu = input.gpu()
    weight_gpu = weight.gpu()

    out_size = out.size
    threads_per_block = 256
    blocks = (out_size + threads_per_block - 1) // threads_per_block

    cuda_kernel_conv1d[blocks, threads_per_block](  # type: ignore
        out_gpu._tensor._storage,
        np.array(out.shape, dtype=np.int32),
        np.array(out._tensor._strides, dtype=np.int32),
        input_gpu._tensor._storage,
        np.array(input.shape, dtype=np.int32),
        np.array(input._tensor._strides, dtype=np.int32),
        weight_gpu._tensor._storage,
        np.array(weight.shape, dtype=np.int32),
        np.array(weight._tensor._strides, dtype=np.int32),
        reverse,
    )

    cuda.synchronize()
    return out_gpu.cpu()


def launch_conv2d(out: Tensor, input: Tensor, weight: Tensor, reverse: bool) -> Tensor:
    """Launch the 2D convolution CUDA kernel.

    Args:
    ----
        out (Tensor): Pre-allocated output tensor (batch, out_channels, height, width).
        input (Tensor): Input tensor (batch, in_channels, height, width).
        weight (Tensor): Weight tensor (out_channels, in_channels, kh, kw).
        reverse (bool): If True, anchor kernel bottom-right, else top-left.

    Returns:
    -------
        Tensor: Output tensor after convolution on CPU.

    """
    out_gpu = out.gpu()
    input_gpu = input.gpu()
    weight_gpu = weight.gpu()

    out_size = out.size
    threads_per_block = 256
    blocks = (out_size + threads_per_block - 1) // threads_per_block

    cuda_kernel_conv2d[blocks, threads_per_block](  # type: ignore
        out_gpu._tensor._storage,
        np.array(out.shape, dtype=np.int32),
        np.array(out._tensor._strides, dtype=np.int32),
        input_gpu._tensor._storage,
        np.array(input.shape, dtype=np.int32),
        np.array(input._tensor._strides, dtype=np.int32),
        weight_gpu._tensor._storage,
        np.array(weight.shape, dtype=np.int32),
        np.array(weight._tensor._strides, dtype=np.int32),
        reverse,
    )
    cuda.synchronize()
    return out_gpu.cpu()


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute the forward pass of 1D convolution.

        Args:
        ----
            ctx (Context): Autodiff context for saving intermediate values.
            input (Tensor): Input tensor (batch, in_channels, width).
            weight (Tensor): Weight tensor (out_channels, in_channels, kernel_width).

        Returns:
        -------
            Tensor: Output tensor (batch, out_channels, width).

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, w))  # removed ndim
        launch_conv1d(output, input, weight, False)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute gradients for 1D convolution.

        Args:
        ----
            ctx (Context): Context with saved tensors from forward pass.
            grad_output (Tensor): Gradient w.r.t. output (batch, out_channels, width)

        Returns:
        -------
            (Tensor, Tensor):
                grad_input: (batch, in_channels, width)
                grad_weight: (out_channels, in_channels, kernel_width)

        """
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels2, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        launch_conv1d(grad_weight, new_input, new_grad_output, False)
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        launch_conv1d(grad_input, grad_output, new_weight, True)
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute the forward pass of 2D convolution.

        Args:
        ----
            ctx (Context): Context object for saving intermediate values.
            input (Tensor): (batch, in_channels, height, width)
            weight (Tensor): (out_channels, in_channels, kh, kw)

        Returns:
        -------
            Tensor: (batch, out_channels, height, width)

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))  # removed ndim
        launch_conv2d(output, input, weight, False)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute gradients for 2D convolution.

        Args:
        ----
            ctx (Context): Context from forward pass.
            grad_output (Tensor): Gradient w.r.t. output (batch, out_channels, height, width)

        Returns:
        -------
            (Tensor, Tensor):
                grad_input: (batch, in_channels, height, width)
                grad_weight: (out_channels, in_channels, kh, kw)

        """
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels2, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        launch_conv2d(grad_weight, new_input, new_grad_output, False)
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        launch_conv2d(grad_input, grad_output, new_weight, True)
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
