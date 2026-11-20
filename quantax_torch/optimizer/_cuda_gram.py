"""Optional FP64 triangular Gram update using PyTorch's CUDA BLAS handle."""

import ctypes
import sys
from functools import cache

import torch


@cache
def _blas(version):
    if version is None or not sys.platform.startswith("linux"):
        return None
    try:
        library = ctypes.CDLL(f"libcublas.so.{version.split('.', 1)[0]}")
        syrk = library.cublasDsyrk_v2
        pointer_mode = library.cublasGetPointerMode_v2
    except (OSError, AttributeError):
        return None
    ptr = ctypes.c_void_p
    integer = ctypes.c_int
    syrk.argtypes = (ptr, integer, integer, integer, integer, ptr, ptr, integer, ptr, ptr, integer)
    syrk.restype = integer
    pointer_mode.argtypes = (ptr, ctypes.POINTER(integer))
    pointer_mode.restype = integer
    return syrk, pointer_mode


def add_gram_lower_(out, block):
    """Return False before writing if cuBLAS or this layout is unavailable."""
    if (out.dtype != torch.float64 or block.dtype != torch.float64
            or not out.is_cuda or out.device != block.device or block.ndim != 2
            or out.shape != (len(block), len(block)) or block.is_neg() or out.is_neg()
            or not (out.is_contiguous() and block.is_contiguous())
            or (torch.is_grad_enabled() and (out.requires_grad or block.requires_grad))
            or torch.are_deterministic_algorithms_enabled()):
        return False
    functions = _blas(torch.version.cuda)
    if functions is None:
        return False
    syrk, pointer_mode = functions
    with torch.cuda.device(out.device):
        if torch.cuda.is_current_stream_capturing():
            return False
        # PyTorch selects the handle and binds the active stream. Do not
        # change any modes on this handle shared with its other operations.
        handle = torch.cuda.current_blas_handle()
        mode = ctypes.c_int()
        if pointer_mode(handle, ctypes.byref(mode)) or mode.value != 0:
            return False
        n, p = block.shape
        one = ctypes.c_double(1.0)
        coeff = ctypes.byref(one)
        # Row-major (N,P) is column-major (P,N); the column-major upper
        # triangle is the row-major lower triangle. Use op(A) = A^T.
        status = syrk(handle, 1, 1, n, p, coeff, block.data_ptr(), p, coeff, out.data_ptr(), n)
    if status:
        raise RuntimeError(f"cuBLAS FP64 Gram update failed (status {status}).")
    return True
