import math

import torch
from torch import nn
from torch.nn import functional as F

from .activation import ExpSum, Prod, Scale, ScaleFn
from .initializers import he_normal
from .modules import Conv, Lambda, Sequential
from .nqs_layers import ReshapeConv
from ..global_defs import get_lattice, is_default_cpl


class ResBlock(nn.Module):
    def __init__(self, lattice, channels, kernel_size, first):
        super().__init__()
        in_channels = lattice.shape[-1] if first else channels
        self.conv1 = Conv(lattice.dim, in_channels, channels, kernel_size, kernel_init=he_normal)
        self.conv2 = Conv(lattice.dim, channels, channels, kernel_size, kernel_init=he_normal)
        self.first = first
        self.channel_dim = -(lattice.dim + 1)

    def _normalize(self, x):
        var, mean = x.var_mean(dim=self.channel_dim, correction=0, keepdim=True)
        return F.gelu((x - mean) / (var + 1e-6).sqrt(), approximate="tanh")

    def forward(self, x):
        y = x / math.sqrt(2) if self.first else self._normalize(x)
        return x + self.conv2(self._normalize(self.conv1(y)))


def ResProd(depth, channels, kernel_size, final_actfn):
    if depth < 2 or depth % 2:
        raise ValueError("ResProd requires positive even depth.")
    lattice = get_lattice()
    blocks = [ResBlock(lattice, channels, kernel_size, i == 0) for i in range(depth // 2)]
    features = channels * math.prod(lattice.shape[:-1])
    scale = ScaleFn(final_actfn, features, scaling=1 / math.sqrt(len(blocks)))
    return Sequential([ReshapeConv(), *blocks, scale, Prod(lattice.dim + 1)])


def _convolution_stack(depth, channels, kernel_size, intermediate):
    if depth < 1:
        raise ValueError("depth must be positive.")
    lattice = get_lattice()
    layers = []
    for i in range(depth):
        in_channels = lattice.shape[-1] if i == 0 else channels
        layers.append(Conv(lattice.dim, in_channels, channels, kernel_size, use_bias=False))
        if i < depth - 1:
            layers.append(Lambda(intermediate))
    return layers


def SinhCosh(depth, channels, kernel_size):
    lattice = get_lattice()
    features = channels * math.prod(lattice.shape[:-1])
    scale = ScaleFn(torch.cosh, features).scale
    convs = _convolution_stack(depth, channels, kernel_size, torch.sinh)
    return Sequential(
        [ReshapeConv(), Scale(scale), *convs, Lambda(torch.cosh), Prod(lattice.dim + 1)],
        holomorphic=is_default_cpl(),
    )


def SchmittNet(depth, channels, kernel_size):
    lattice = get_lattice()
    features = channels * math.prod(lattice.shape[:-1])

    def final(x):
        return x**2 / 2 - x**4 / 14 + x**6 / 45

    def intermediate(x):
        return x - x**3 / 3 + x**5 * 2 / 15

    scale = ScaleFn(lambda x: final(x).exp(), features).scale
    convs = _convolution_stack(depth, channels, kernel_size, intermediate)
    return Sequential(
        [ReshapeConv(), Scale(scale), *convs, Lambda(final), ExpSum(lattice.dim + 1)],
        holomorphic=is_default_cpl(),
    )
