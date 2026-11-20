import torch

from .modules import NoGradLayer
from ..global_defs import get_lattice, get_params_dtype
from ..symmetry import TransND


class ReshapeConv(NoGradLayer):
    def __init__(self):
        super().__init__()
        self.shape = get_lattice().shape
        self.dtype = get_params_dtype()

    def forward(self, x):
        return x.reshape(*x.shape[:-1], *self.shape).movedim(-1, -len(self.shape)).to(self.dtype)


class ConvSymmetrize(NoGradLayer):
    def __init__(self, trans_symm=None):
        super().__init__()
        if trans_symm is None:
            trans_symm = TransND()
        self.feature_dims = get_lattice().dim + 1
        self.register_buffer("eigval", trans_symm.eigval)

    def forward(self, x):
        shape = x.shape[:-self.feature_dims]
        x = x.reshape(*shape, -1, self.eigval.numel())
        return (x * self.eigval).mean(dim=(-2, -1))
