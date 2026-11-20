import numpy as np
import torch

from .sites import Lattice, Sites


_dtype = torch.float64
_params_dtype = torch.float32
_seed = torch.initial_seed()
_generators = {}
_torch_dtypes = {
    np.dtype(name): getattr(torch, name)
    for name in ("float16", "float32", "float64", "complex64", "complex128")
}


def _floating_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        if dtype.is_floating_point or dtype.is_complex:
            return dtype
    else:
        return _torch_dtypes[np.dtype(dtype)]
    raise ValueError("Expected a floating-point or complex dtype.")


def set_default_dtype(dtype):
    global _dtype
    _dtype = _floating_dtype(dtype)


def set_params_dtype(dtype):
    global _params_dtype
    _params_dtype = _floating_dtype(dtype)


def get_default_dtype():
    return _dtype


def get_params_dtype():
    return _params_dtype


def is_default_cpl():
    return _dtype.is_complex


def is_params_cpl():
    return _params_dtype.is_complex


def set_random_seed(seed):
    global _seed
    _seed = int(seed)
    _generators.clear()


def get_generator(device="cpu"):
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device not in _generators:
        _generators[device] = torch.Generator(device=device).manual_seed(_seed)
    return _generators[device]


def get_sites():
    if Sites._SITES is None:
        raise RuntimeError("Define the sites before constructing a state or operator.")
    return Sites._SITES


def get_lattice():
    lattice = get_sites()
    if not isinstance(lattice, Lattice):
        raise RuntimeError("The active sites are not a lattice.")
    return lattice
