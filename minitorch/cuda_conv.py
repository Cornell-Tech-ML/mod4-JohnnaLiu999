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
        **kwargs: Additional arguments for Numba JIT

    Returns:
    -------
        Compiled function

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

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right

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
    out_size: int,
    input: Any,
    input_shape: np.ndarray,
    input_strides: np.ndarray,
    weight: Any,
    weight_shape: np.ndarray,
    weight_strides: np.ndarray,
    reverse: bool,
) -> None:
    """2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right

    """
    batch_ = out_shape[0]
    out_channels = out_shape[1]
    out_height = out_shape[2]
    out_width = out_shape[3]

    in_channels = input_shape[1]
    height = input_shape[2]
    width = input_shape[3]
    out_channels_, in_channels_, kh, kw = weight_shape

    idx = cuda.grid(1)
    total = batch_ * out_channels * out_height * out_width
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


def launch_conv1d(
    out: Tensor,
    input: Tensor,
    weight: Tensor,
    reverse: bool,
) -> Tensor:
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
    out_arr = cuda.to_device(out._tensor._storage)
    input_arr = cuda.to_device(input._tensor._storage)
    weight_arr = cuda.to_device(weight._tensor._storage)

    out_size = out.size
    threads_per_block = 256
    blocks = (out_size + threads_per_block - 1) // threads_per_block

    cuda_kernel_conv1d[blocks, threads_per_block](  # type: ignore
        out_arr,
        np.array(out.shape, dtype=np.int32),
        np.array(out._tensor._strides, dtype=np.int32),
        input_arr,
        np.array(input.shape, dtype=np.int32),
        np.array(input._tensor._strides, dtype=np.int32),
        weight_arr,
        np.array(weight.shape, dtype=np.int32),
        np.array(weight._tensor._strides, dtype=np.int32),
        reverse,
    )
    cuda.synchronize()

    out_arr.copy_to_host(out._tensor._storage)
    return out


def launch_conv2d(
    out: Tensor,
    input: Tensor,
    weight: Tensor,
    reverse: bool,
) -> Tensor:
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
    out_arr = cuda.to_device(out._tensor._storage)
    input_arr = cuda.to_device(input._tensor._storage)
    weight_arr = cuda.to_device(weight._tensor._storage)

    out_size = out.size
    threads_per_block = 256
    blocks = (out_size + threads_per_block - 1) // threads_per_block

    cuda_kernel_conv2d[blocks, threads_per_block](  # type: ignore
        out_arr,
        np.array(out.shape, dtype=np.int32),
        np.array(out._tensor._strides, dtype=np.int32),
        out_size,
        input_arr,
        np.array(input.shape, dtype=np.int32),
        np.array(input._tensor._strides, dtype=np.int32),
        weight_arr,
        np.array(weight.shape, dtype=np.int32),
        np.array(weight._tensor._strides, dtype=np.int32),
        reverse,
    )
    cuda.synchronize()

    out_arr.copy_to_host(out._tensor._storage)
    return out


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x w
            weight : out_channel x in_channel x kw

        Returns:
        -------
            batch x out_channel x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Manually create a zero tensor without zeros(...) or ndim.
        size = batch * out_channels * w
        output = Tensor.make(
            [0.0] * size, (batch, out_channels, w), backend=input.backend
        )

        launch_conv1d(output, input, weight, False)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute gradients for 1D convolution.

        Args:
        ----
            ctx: Context object containing saved tensors from forward pass
            grad_output: Gradient of loss with respect to conv output
                Shape: batch x out_channels x width

        Returns:
        -------
            tuple of:
                grad_input: Gradient with respect to input
                    Shape: batch x in_channels x width
                grad_weight: Gradient with respect to weight
                    Shape: out_channels x in_channels x kernel_width

        """
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape

        # Create grad_weight as zeros
        gw_size = in_channels2 * out_channels * kw
        grad_weight = Tensor.make(
            [0.0] * gw_size,
            (in_channels2, out_channels, kw),
            backend=grad_output.backend,
        )

        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        launch_conv1d(grad_weight, new_input, new_grad_output, False)
        grad_weight = grad_weight.permute(1, 0, 2)

        # Create grad_input as zeros
        gi_size = batch * in_channels * w
        grad_input = Tensor.make(
            [0.0] * gi_size, (batch, in_channels, w), backend=input.backend
        )

        new_weight = weight.permute(1, 0, 2)
        launch_conv1d(grad_input, grad_output, new_weight, True)
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
        -------
            (:class:`Tensor`) : batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2

        # Create zero tensor for output
        size = batch * out_channels * h * w
        output = Tensor.make(
            [0.0] * size, (batch, out_channels, h, w), backend=input.backend
        )

        launch_conv2d(output, input, weight, False)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute gradients for 2D convolution.

        Args:
        ----
            ctx: Context object containing saved tensors from forward pass
            grad_output: Gradient of loss with respect to conv output
                Shape: batch x out_channels x height x width

        Returns:
        -------
            tuple of:
                grad_input: Gradient w.r.t. input
                    Shape: batch x in_channels x height x width
                grad_weight: Gradient w.r.t. weight
                    Shape: out_channels x in_channels x kernel_height x kernel_width

        """
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape

        # Create grad_weight as zeros
        gw_size = in_channels2 * out_channels * kh * kw
        grad_weight = Tensor.make(
            [0.0] * gw_size,
            (in_channels2, out_channels, kh, kw),
            backend=grad_output.backend,
        )

        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        launch_conv2d(grad_weight, new_input, new_grad_output, False)
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        # Create grad_input as zeros
        gi_size = batch * in_channels * h * w
        grad_input = Tensor.make(
            [0.0] * gi_size, (batch, in_channels, h, w), backend=input.backend
        )

        new_weight = weight.permute(1, 0, 2, 3)
        launch_conv2d(grad_input, grad_output, new_weight, True)
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
