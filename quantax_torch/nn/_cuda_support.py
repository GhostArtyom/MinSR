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


def periodic_cuda_issue(device):
    capability = cuda_capability(device)
    if capability is None:
        return "requires NVIDIA CUDA"
    if capability < (8, 0):
        return "Triton CUDA requires compute capability 8.0 or later"
    return triton_runtime_issue()
