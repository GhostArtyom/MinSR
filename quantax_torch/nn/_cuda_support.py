"""CUDA architecture and runtime checks for optional kernels."""

import os
from importlib.util import find_spec
from shutil import which

import torch


def cuda_capability(device):
    if device.type != "cuda" or torch.version.cuda is None:
        return None
    return torch.cuda.get_device_capability(device)


def triton_runtime_issue():
    if find_spec("triton") is None:
        return "Triton is unavailable"
    compiler = which(os.environ["CC"]) if "CC" in os.environ else which("gcc") or which("clang")
    return None if compiler else "C compiler is unavailable"


def supports_tuned_kernels(device):
    return cuda_capability(device) == (12, 0) and triton_runtime_issue() is None


def validated_ada_res_sum(network, lattice_shape, nsamples, max_parallel, *, channels=(16,)):
    return (tuple(lattice_shape) == (10, 10, 1) and nsamples == 1000 and max_parallel == 32768
            and any(network == dict(depth=16, channels=width, kernel_size=3, use_sinh=True) for width in channels))
