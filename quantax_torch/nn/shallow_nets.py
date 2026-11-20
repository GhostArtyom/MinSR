import math

import torch

from .activation import Prod, ScaleFn
from .modules import Conv, Linear, Sequential
from .nqs_layers import ReshapeConv
from ..global_defs import get_lattice, get_sites, is_params_cpl


def SingleDense(features, actfn, use_bias=True, holomorphic=False):
    linear = Linear(get_sites().nsites, features, use_bias)
    return Sequential([linear, ScaleFn(actfn, features), Prod()], holomorphic)


def RBM_Dense(features, use_bias=True):
    return SingleDense(features, torch.cosh, use_bias, holomorphic=is_params_cpl())


def SingleConv(channels, actfn, use_bias=True, holomorphic=False):
    lattice = get_lattice()
    conv = Conv(lattice.dim, lattice.shape[-1], channels, lattice.shape[:-1], use_bias=use_bias)
    features = channels * math.prod(lattice.shape[:-1])
    return Sequential([ReshapeConv(), conv, ScaleFn(actfn, features), Prod(lattice.dim + 1)], holomorphic)


def RBM_Conv(channels, use_bias=True):
    return SingleConv(channels, torch.cosh, use_bias, holomorphic=is_params_cpl())
