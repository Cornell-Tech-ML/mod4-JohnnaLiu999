from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw
    
    # Ensure input is contiguous before viewing
    input = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    # After permute, call contiguous() again before viewing
    input = input.permute(0, 1, 2, 4, 3, 5).contiguous()
    input = input.view(batch, channel, new_height, new_width, kh * kw)

    return input, new_height, new_width

    """
    out = input.view(batch, channel, new_height, kh, new_width, kw)

    # Now permute to (batch, channel, new_height, new_width, kh, kw)
    out = out.permute(0, 1, 2, 4, 3, 5)

    # Finally, view into (batch, channel, new_height, new_width, kh*kw)
    out = out.view(batch, channel, new_height, new_width, kh * kw)

    return out, new_height, new_width

    
    # Reshape the input into (batch, channel, new_height, kh, new_width, kw)
    out = input.reshape((batch, channel, new_height, kh, new_width, kw))
    # Rearrange to (batch, channel, new_height, new_width, kh*kw)
    out = out.permute(0, 1, 2, 4, 3, 5)
    out = out.reshape((batch, channel, new_height, new_width, kh * kw))

    return out, new_height, new_width

    #raise NotImplementedError("Need to implement for Task 4.3")
    """

# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D

    Args:
    ----
        a: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        :class:`Tensor`: Pooled tensor

    """
    tiled_input, new_height, new_width = tile(input, kernel)
    # Perform mean and ensure contiguity before view
    out = tiled_input.mean(dim=4).contiguous()
    out = out.view(out.shape[0], out.shape[1], new_height, new_width)
    return out

    """
    batch, channel, height, width = input.shape
    tiled_input, new_height, new_width = tile(input, kernel)
    out = tiled_input.mean(dim=-1)

    return out.view(batch, channel, new_height, new_width)
    """


# TODO: Implement for Task 4.4.

class Max(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Max function"""
        ctx.save_for_backward(t1, dim)
        return t1.f.max_reduce(t1, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Max function"""
        t1, dim = ctx.saved_values
        return grad_output * argmax(t1, dim), 0.0


def max(t1: Tensor, dim: int) -> Tensor:
    """Max function"""
    return Max.apply(t1, t1._ensure_tensor(dim))

def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor

    Args:
    ----
        input: Tensor
        dim: Dimension to compute argmax

    Returns:
    -------
        Tensor of indices

    """
    max_val = max(input, dim)
    return input == max_val

def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor

    Args:
    ----
        input: Tensor
        dim: Dimension to compute softmax along

    Returns:
    -------
        Tensor of softmax values

    """
    exp = input.exp()
    return exp / exp.sum(dim=dim)

def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor

    Args:
    ----
        input: Tensor
        dim: Dimension to compute logsoftmax along

    Returns:
    -------
        Tensor of log softmax values

    """
    return softmax(input, dim).log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling to an image tensor

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    input, new_height, new_width = tile(input, kernel)
    return (
        max(input, dim=4)
        .contiguous()
        .view(input.shape[0], input.shape[1], new_height, new_width)
    )


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise.

    Args:
    ----
        input: Tensor to apply dropout to
        rate: Probability of dropping a position (0.0 to 1.0)
        ignore: If True, disable dropout and return input unchanged

    Returns:
    -------
        Tensor with dropout applied

    """
    if ignore:
        return input

    # Use tensor backend's random function instead of direct random method
    rand_val = rand(input.shape, backend=input.backend)
    mask = rand_val > rate

    scale = 1.0 / (1.0 - rate) if rate != 1.0 else 1.0
    return mask * input * scale





"""
max_reduce = FastOps.reduce(operators.max, -1e9)

class Max(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for max should be max reduction."""
        ctx.save_for_backward(a, dim)
        return max_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for max should be argmax."""
        original_tensor, dim = ctx.saved_values
        arg_max = argmax(original_tensor, int(dim.item()))
        return grad_output * arg_max, 0.0

def max(a: Tensor, dim: Optional[int] = None) -> Tensor:
    """Computes the Max of all elements in the tensor or along a dimension."""
    if dim is None:
        return Max.apply(a.contiguous().view(a.size), tensor(0.0))
    else:
        return Max.apply(a, tensor(dim))

def softmax(a: Tensor, dim: Optional[int] = None) -> Tensor:
    """Compute the softmax as a tensor.
    math:
        softmax(a) = exp(a) / sum(exp(a))
        = exp(a - max(a)) / sum(exp(a - max(a)))

    Args:
    ----
        a: input tensor
        dim: dimension to apply softmax

    Returns:
    -------
        :class:`Tensor` : softmax tensor

    """
    # If dim is None, treat tensor as 1D
    if dim is None:
        a = a.contiguous().view(a.size)
        dim = 0

    # Subtract max for numerical stability
    a_max = max(a, dim)
    exp_a = (a - a_max).exp()

    # Compute sum of exponentials and divide
    sum_exp_a = exp_a.sum(dim)
    return exp_a / sum_exp_a

def logsoftmax(a: Tensor, dim: Optional[int] = None) -> Tensor:
    """Compute the log of the softmax as a tensor.
    math:
        log(softmax(a)) = log(exp(a - max(a)) / sum(exp(a - max(a))))
        = log(exp(a - max(a))) - log(sum(exp(a - max(a))))
        = a - max(a) - log(sum(exp(a - max(a))))

    Args:
    ----
        a: input tensor
        dim: dimension to apply logsoftmax

    Returns:
    -------
        :class:`Tensor` : logsoftmax tensor

    """
    if dim is None:
        a = a.contiguous().view(a.size)
        dim = 0

    a_max = max(a, dim)
    sum_exp_a = (a - a_max).exp().sum(dim)
    return (a - a_max) - sum_exp_a.log()

def maxpool2d(a: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D

    Args:
    ----
        a: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        :class:`Tensor`: Pooled tensor

    """
    batch, channel, height, width = a.shape
    tiled_input, new_height, new_width = tile(a, kernel)
    out = max(tiled_input, dim=-1)

    return out.view(batch, channel, new_height, new_width)

def dropout(a: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise.

    Args:
    ----
        a: input tensor
        p: probability [0, 1) of dropping out each position
        ignore: turn off dropout

    Returns:
    -------
        :class:`Tensor` : tensor with dropout applied

    """
    if ignore or p == 0:
        return a

    mask = rand(a.shape) > p
    return a * mask

"""
def max(t1: Tensor, dim: Optional[int] = None) -> Tensor:
    """Computes the Max of all elements in the tensor or along a dimension."""
    if dim is None:
        return Max.apply(t1.contiguous().view(t1.size), tensor(0.0))
    else:
        return Max.apply(t1, tensor(dim))


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor along `dim`.

    Returns a mask with 1s at the max positions and 0s elsewhere.
    """
    max_val = max(input, dim)
    # Ensure output is float (if needed)
    return (input == max_val).float()


def softmax(a: Tensor, dim: Optional[int] = None) -> Tensor:
    """Compute the softmax as a tensor.
    math:
        softmax(a) = exp(a) / sum(exp(a))
        = exp(a - max(a)) / sum(exp(a - max(a)))

    Args:
    ----
        a: input tensor
        dim: dimension to apply softmax

    Returns:
    -------
        :class:`Tensor` : softmax tensor

    """
    # If dim is None, treat tensor as 1D
    if dim is None:
        a = a.contiguous().view(a.size)
        dim = 0

    # Subtract max for numerical stability
    a_max = max(a, dim)
    exp_a = (a - a_max).exp()

    # Compute sum of exponentials and divide
    sum_exp_a = exp_a.sum(dim)
    return exp_a / sum_exp_a


def logsoftmax(a: Tensor, dim: Optional[int] = None) -> Tensor:
    """Compute the log of the softmax as a tensor.
    math:
        log(softmax(a)) = log(exp(a - max(a)) / sum(exp(a - max(a))))
        = log(exp(a - max(a))) - log(sum(exp(a - max(a))))
        = a - max(a) - log(sum(exp(a - max(a))))

    Args:
    ----
        a: input tensor
        dim: dimension to apply logsoftmax

    Returns:
    -------
        :class:`Tensor` : logsoftmax tensor

    """
    if dim is None:
        a = a.contiguous().view(a.size)
        dim = 0

    a_max = max(a, dim)
    sum_exp_a = (a - a_max).exp().sum(dim)
    return (a - a_max) - sum_exp_a.log()

def maxpool2d(a: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D

    Args:
    ----
        a: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        :class:`Tensor`: Pooled tensor

    """
    batch, channel, height, width = a.shape
    tiled_input, new_height, new_width = tile(a, kernel)
    out = max(tiled_input, dim=-1)

    return out.view(batch, channel, new_height, new_width)


def dropout(a: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise.

    Args:
    ----
        a: input tensor
        p: probability [0, 1) of dropping out each position
        ignore: turn off dropout

    Returns:
    -------
        :class:`Tensor` : tensor with dropout applied

    """
    if ignore or p == 0:
        return a

    mask = rand(a.shape) > p
    return a * mask
"""