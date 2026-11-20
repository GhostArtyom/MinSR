import math

from torch import nn

from .activation import Exp, Scale, SinhShift, pair_cpl
from .initializers import he_normal
from .modules import Conv, Lambda, Sequential
from .nqs_layers import ConvSymmetrize, ReshapeConv
from ..global_defs import get_lattice, is_default_cpl, is_params_cpl
from ..symmetry import Identity


class ResBlock(nn.Module):
    def __init__(self, lattice, channels, kernel_size, nblock, total_blocks):
        super().__init__()
        in_channels = lattice.shape[-1] if nblock == 0 else channels
        self.conv1 = Conv(lattice.dim, in_channels, channels, kernel_size, kernel_init=he_normal)
        self.conv2 = Conv(lattice.dim, channels, channels, kernel_size, kernel_init=he_normal,
                          use_bias=nblock != total_blocks - 1)
        self.nblock = nblock

    def forward(self, x):
        scale = math.sqrt(2 if self.nblock == 0 else self.nblock + 1)
        y = x / scale
        if self.nblock > 0:
            y = y.relu()
        return x + self.conv2(self.conv1(y).relu())


def ResSum(depth, channels, kernel_size, use_sinh=False, trans_symm=None):
    if is_params_cpl() or depth < 2 or depth % 2:
        raise ValueError("ResSum requires real parameters and positive even depth.")
    lattice = get_lattice()
    blocks = [ResBlock(lattice, channels, kernel_size, i, depth // 2) for i in range(depth // 2)]
    layers = [ReshapeConv(), *blocks, Scale(1 / math.sqrt(len(blocks)))]
    if is_default_cpl():
        if channels % 2:
            raise ValueError("Complex ResSum requires an even number of channels.")
        layers.append(Lambda(lambda x: pair_cpl(x, dim=-(lattice.dim + 1))))
    layers.append(SinhShift() if use_sinh else Exp())
    if trans_symm is not Identity():
        layers.append(ConvSymmetrize(trans_symm))
    net = Sequential(layers)
    net.config = dict(depth=depth, channels=channels, kernel_size=kernel_size, use_sinh=use_sinh)
    return net
