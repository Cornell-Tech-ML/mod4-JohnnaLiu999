from typing import Tuple, TypeVar, Any
import numpy as np
from numba import cuda
from numba import njit

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Shape,
    Strides,
    Storage,
    to_index,
    index_to_position,
    broadcast_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")

# CUDA kernel for 1D Convolution
@cuda.jit
def cuda_kernel_conv1d(
    out, out_shape, out_strides,
    input, input_shape, input_strides,
    weight, weight_shape, weight_strides,
    reverse
):
    # Unpack shapes
    batch_ = out_shape[0]
    out_channels = out_shape[1]
    out_width = out_shape[2]

    batch = input_shape[0]
    in_channels = input_shape[1]
    width = input_shape[2]

    out_channels_ = weight_shape[0]
    in_channels_ = weight_shape[1]
    kw = weight_shape[2]

    # Verify shape matches (not strictly necessary on the GPU, but good for logic)
    # assert batch == batch_ and in_channels == in_channels_ and out_channels == out_channels_

    # Compute the global thread ID
    idx = cuda.grid(1)
    total = batch_ * out_channels * out_width
    if idx >= total:
        return

    # Convert the linear index 'idx' into (b, oc, ow)
    b = idx // (out_channels * out_width)
    oc = (idx // out_width) % out_channels
    ow = idx % out_width

    accum = 0.0
    for ic in range(in_channels):
        for k in range(kw):
            iw = ow + k if not reverse else ow - k
            if 0 <= iw < width:
                in_pos = b * input_strides[0] + ic * input_strides[1] + iw * input_strides[2]
                weight_pos = oc * weight_strides[0] + ic * weight_strides[1] + k * weight_strides[2]
                accum += input[in_pos] * weight[weight_pos]

    out_pos = b * out_strides[0] + oc * out_strides[1] + ow * out_strides[2]
    out[out_pos] = accum


# CUDA kernel for 2D Convolution
@cuda.jit
def cuda_kernel_conv2d(
    out, out_shape, out_strides,
    input, input_shape, input_strides,
    weight, weight_shape, weight_strides,
    reverse
):
    # Unpack shapes
    batch_ = out_shape[0]
    out_channels = out_shape[1]
    out_height = out_shape[2]
    out_width = out_shape[3]

    batch = input_shape[0]
    in_channels = input_shape[1]
    height = input_shape[2]
    width = input_shape[3]

    out_channels_ = weight_shape[0]
    in_channels_ = weight_shape[1]
    kh = weight_shape[2]
    kw = weight_shape[3]

    # total output size
    total = batch_ * out_channels * out_height * out_width
    idx = cuda.grid(1)
    if idx >= total:
        return

    # Convert linear index 'idx' into (b, oc, i, j)
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
                    weight_pos = oc * weight_strides[0] + ic * weight_strides[1] + ki * weight_strides[2] + kj * weight_strides[3]
                    in_pos = b * input_strides[0] + ic * input_strides[1] + ih * input_strides[2] + iw * input_strides[3]
                    accum += weight[weight_pos] * input[in_pos]

    out_pos = b * out_strides[0] + oc * out_strides[1] + i * out_strides[2] + j * out_strides[3]
    out[out_pos] = accum


# Launch functions from Python
def launch_conv1d(
    out: Tensor, input: Tensor, weight: Tensor, reverse: bool
):
    # Move data to GPU if not already
    out_gpu = out.gpu()
    input_gpu = input.gpu()
    weight_gpu = weight.gpu()

    # Set up grid and block dimensions
    out_size = out.size
    threads_per_block = 256
    blocks = (out_size + threads_per_block - 1) // threads_per_block

    cuda_kernel_conv1d[blocks, threads_per_block](
        out_gpu._tensor._storage,
        np.array(out.shape, dtype=np.int32),
        np.array(out._tensor._strides, dtype=np.int32),
        input_gpu._tensor._storage,
        np.array(input.shape, dtype=np.int32),
        np.array(input._tensor._strides, dtype=np.int32),
        weight_gpu._tensor._storage,
        np.array(weight.shape, dtype=np.int32),
        np.array(weight._tensor._strides, dtype=np.int32),
        reverse
    )

    cuda.synchronize()

    return out_gpu.cpu()  # Bring data back to CPU if needed


def launch_conv2d(
    out: Tensor, input: Tensor, weight: Tensor, reverse: bool
):
    # Move data to GPU
    out_gpu = out.gpu()
    input_gpu = input.gpu()
    weight_gpu = weight.gpu()

    out_size = out.size
    threads_per_block = 256
    blocks = (out_size + threads_per_block - 1) // threads_per_block

    cuda_kernel_conv2d[blocks, threads_per_block](
        out_gpu._tensor._storage,
        np.array(out.shape, dtype=np.int32),
        np.array(out._tensor._strides, dtype=np.int32),
        input_gpu._tensor._storage,
        np.array(input.shape, dtype=np.int32),
        np.array(input._tensor._strides, dtype=np.int32),
        weight_gpu._tensor._storage,
        np.array(weight.shape, dtype=np.int32),
        np.array(weight._tensor._strides, dtype=np.int32),
        reverse
    )

    cuda.synchronize()

    return out_gpu.cpu()


# Example functions like those in CPU code, but calling CUDA kernels instead
class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, w))
        launch_conv1d(output, input, weight, False)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape

        # Grad w.r.t weight
        grad_weight = grad_output.zeros((in_channels2, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        launch_conv1d(grad_weight, new_input, new_grad_output, False)
        grad_weight = grad_weight.permute(1, 0, 2)

        # Grad w.r.t input
        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        launch_conv1d(grad_input, grad_output, new_weight, True)
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        launch_conv2d(output, input, weight, False)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
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
