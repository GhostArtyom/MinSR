import torch
from torch import nn
from torch.nn import functional as F

from .initializers import lecun_normal
from ..global_defs import get_params_dtype


class NoGradLayer(nn.Module):
    """A layer whose state is registered as buffers."""


class Lambda(NoGradLayer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class Sequential(nn.Sequential):
    def __init__(self, layers, holomorphic=False):
        super().__init__(*layers)
        self.holomorphic = holomorphic

    @property
    def layers(self):
        return tuple(self)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Sequential(self.layers[index], self.holomorphic)
        return super().__getitem__(index)


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, use_bias=True, kernel_init=lecun_normal,
                 bias_init=nn.init.zeros_, *, key=None):
        super().__init__(in_features, out_features, bias=use_bias, dtype=get_params_dtype())
        kernel_init(self.weight, key)
        if self.bias is not None:
            bias_init(self.bias)

    def forward(self, x):
        return super().forward(x.to(self.weight.dtype))


class Conv(nn.Module):
    def __init__(self, num_spatial_dims, in_channels, out_channels, kernel_size,
                 stride=1, padding="CIRCULAR", dilation=1, groups=1, use_bias=True,
                 kernel_init=lecun_normal, bias_init=nn.init.zeros_, *, key=None):
        super().__init__()
        dims = num_spatial_dims
        if dims not in (1, 2, 3) or in_channels % groups or out_channels % groups:
            raise ValueError("Invalid convolution dimensions or groups.")
        self.kernel_size = (kernel_size,) * dims if isinstance(kernel_size, int) else tuple(kernel_size)
        self.dilation = (dilation,) * dims if isinstance(dilation, int) else tuple(dilation)
        self.stride = stride
        self.groups = groups
        self.conv = (F.conv1d, F.conv2d, F.conv3d)[dims - 1]
        dtype = get_params_dtype()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_channels, dtype=dtype)) if use_bias else None
        kernel_init(self.weight, key)
        if self.bias is not None:
            bias_init(self.bias)
        self.padding = padding
        # The original padding is asymmetric for an even kernel.
        self._circular_pads = tuple(
            (((size - 1) * rate // 2), size * rate // 2)
            for size, rate in zip(self.kernel_size, self.dilation)
        )

    def forward(self, x):
        x = x.to(self.weight.dtype)
        padding = self.padding
        if padding == "CIRCULAR":
            dims = len(self.kernel_size)
            if all(max(pair) <= size for pair, size in zip(self._circular_pads, x.shape[-dims:])):
                pad = tuple(value for pair in reversed(self._circular_pads) for value in pair)
                x = F.pad(x, pad, mode="circular")
            else:
                for axis, (left, right) in enumerate(self._circular_pads, start=x.ndim - dims):
                    size = x.shape[axis]
                    indices = torch.arange(-left, size + right, device=x.device).remainder(size)
                    x = x.index_select(axis, indices)
            padding = 0
        elif isinstance(padding, str):
            padding = padding.lower()
            if padding == "same":
                strides = (self.stride,) * len(self.kernel_size) if isinstance(self.stride, int) else self.stride
                pads = []
                for size, k, d, s in zip(x.shape[-len(self.kernel_size):], self.kernel_size, self.dilation, strides):
                    total = max(0, ((size - 1) // s) * s + (k - 1) * d + 1 - size)
                    pads.append((total // 2, total - total // 2))
                x = F.pad(x, tuple(value for pair in reversed(pads) for value in pair))
                padding = 0
        elif not isinstance(padding, int) and isinstance(padding[0], (tuple, list)):
            x = F.pad(x, tuple(value for pair in reversed(padding) for value in pair))
            padding = 0
        return self.conv(x, self.weight, self.bias, self.stride, padding, self.dilation, self.groups)


class Conv1d(Conv):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)


class Conv2d(Conv):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)


class Conv3d(Conv):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)
