"""Platform gate for the checkout's tuned Triton kernels."""

import os
from importlib.util import find_spec
from shutil import which

import torch


def supports_tuned_kernels(device):
    if device.type != "cuda" or torch.version.cuda is None:
        return False
    return (torch.cuda.get_device_capability(device) == (12, 0) and find_spec("triton") is not None
            and bool(which(os.environ["CC"]) if "CC" in os.environ else which("gcc") or which("clang")))
