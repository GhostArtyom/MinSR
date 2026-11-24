import math

from torch import nn

from .activation import Exp, Scale, SinhShift, pair_cpl
from .initializers import he_normal
from .modules import Conv, Lambda, Sequential
from .nqs_layers import ConvSymmetrize, ReshapeConv
from ..global_defs import get_lattice, is_default_cpl, is_params_cpl
from ..symmetry import Identity


class ResBlock(nn.Module):
    def __init__(self, lattice, channels, kernel_size, nblock, total_blocks, *, device=None, generator=None):
        super().__init__()
        in_channels = lattice.shape[-1] if nblock == 0 else channels
        self.conv1 = Conv(lattice.dim, in_channels, channels, kernel_size, kernel_init=he_normal,
                          device=device, key=generator)
        self.conv2 = Conv(lattice.dim, channels, channels, kernel_size, kernel_init=he_normal,
                          use_bias=nblock != total_blocks - 1, device=device, key=generator)
        self.nblock = nblock

    @property
    def _input_scale(self):
        return math.sqrt(2 if self.nblock == 0 else self.nblock + 1)

    def forward(self, x, *, convolve=None):
        y = x / self._input_scale
        if self.nblock > 0:
            y = y.relu()
        if convolve is None:
            return x + self.conv2(self.conv1(y).relu())
        return convolve(self.conv2, convolve(self.conv1, y, relu=True), residual=x)


def ResSum(depth, channels, kernel_size, use_sinh=False, trans_symm=None, *, device=None, generator=None):
    if is_params_cpl() or depth < 2 or depth % 2:
        raise ValueError("ResSum requires real parameters and positive even depth.")
    lattice = get_lattice()
    blocks = [
        ResBlock(lattice, channels, kernel_size, i, depth // 2, device=device, generator=generator)
        for i in range(depth // 2)
    ]
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
    if device is not None:
        net.to(device)
    return net
