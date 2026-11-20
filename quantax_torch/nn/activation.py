import math

import torch

from .modules import NoGradLayer
from ..global_defs import get_default_dtype, get_params_dtype, get_sites


class Scale(NoGradLayer):
    def __init__(self, scale):
        super().__init__()
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float64))

    def forward(self, x):
        return x * self.scale


class ScaleFn(Scale):
    def __init__(self, fn, features, scaling=1.0):
        generator = torch.Generator().manual_seed(0)
        x = torch.randn((1000, int(features)), dtype=get_params_dtype(), generator=generator)
        target = 0.1 * math.sqrt(get_sites().nsites)
        errors = []
        for scale in range(100):
            out = fn(x * (scale / 100)).abs().log().sum(dim=1)
            error = (out.std(correction=0) - target).square()
            errors.append(torch.nan_to_num(error, nan=torch.inf))
        scale = scaling * int(torch.stack(errors).argmin()) / 100
        super().__init__(scale)
        self.fn = fn

    def forward(self, x):
        return self.fn(super().forward(x))


class Theta0Layer(NoGradLayer):
    def __init__(self):
        super().__init__()
        self.register_buffer("theta0", torch.zeros((), dtype=get_default_dtype()))


class SinhShift(Theta0Layer):
    def forward(self, x):
        x = x.to(self.theta0.dtype)
        return ((x + self.theta0).exp() - (-x + self.theta0).exp()) / 2 + self.theta0.exp()


class Exp(Theta0Layer):
    def forward(self, x):
        return (x.to(self.theta0.dtype) + self.theta0).exp()


class ExpSum(Theta0Layer):
    def __init__(self, feature_dims=1):
        super().__init__()
        self.feature_dims = feature_dims

    def forward(self, x):
        dims = tuple(range(-self.feature_dims, 0))
        return (x.to(self.theta0.dtype).sum(dim=dims) + self.theta0).exp()


class Prod(ExpSum):
    def forward(self, x):
        x = x.to(self.theta0.dtype).flatten(start_dim=x.ndim - self.feature_dims)
        return (x * (self.theta0 / x.shape[-1]).exp()).prod(dim=-1)


def crelu(x):
    return torch.complex(x.real.relu(), x.imag.relu())


def cardioid(x):
    return 0.5 * (1 + x.angle().cos()) * x


def pair_cpl(x, dim=0):
    real, imag = x.chunk(2, dim=dim)
    return torch.complex(real, imag)
