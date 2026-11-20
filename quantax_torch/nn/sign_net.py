import math

import torch
from torch import nn

from .modules import Conv, Lambda, Sequential
from .nqs_layers import ReshapeConv
from ..global_defs import get_generator, get_lattice
from ..sites import TriangularB
from ..utils.spins import neel, stripe


class SgnNet(Sequential):
    def __init__(self, kernel=None, output="sign", neg=False):
        if output not in ("sign", "phase", "cos"):
            raise ValueError(f"Unknown sign network output {output!r}.")
        lattice = get_lattice()
        dims = tuple(range(-lattice.dim - 1, 0))
        conv = Conv(lattice.dim, lattice.shape[-1], 1, lattice.shape[:-1], use_bias=False)
        with torch.no_grad():
            if kernel is not None:
                conv.weight.copy_(torch.as_tensor(kernel).reshape_as(conv.weight))
            elif output == "sign":
                nn.init.zeros_(conv.weight)
            else:
                nn.init.normal_(conv.weight, std=math.pi / 4, generator=get_generator())

        def activation(x):
            if output == "phase":
                value = (1j * x).exp().sum(dim=dims)
                value = value / value.abs()
            else:
                value = x.cos().mean(dim=dims)
                if output == "sign":
                    value = value.sign()
            return -value if neg else value

        super().__init__([ReshapeConv(), conv, Lambda(activation)])


def MarshallSign(output="sign"):
    return SgnNet(math.pi / 4 * neel(), output, (get_lattice().nsites // 4) % 2 == 1)


def StripeSign(output="sign", alternate_dim=1):
    return SgnNet(math.pi / 4 * stripe(alternate_dim), output, (get_lattice().nsites // 4) % 2 == 1)


def Neel120(output="phase"):
    lattice = get_lattice()
    lx, ly = lattice.shape[:2]
    x = 2 * torch.arange(lx)
    y = torch.zeros(ly) if isinstance(lattice, TriangularB) else torch.arange(ly)
    kernel = math.pi / 3 * (x[:, None] + y).remainder(3) - math.pi / 6
    return SgnNet(kernel, output)
