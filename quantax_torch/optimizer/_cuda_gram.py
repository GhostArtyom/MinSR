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


def add_gram_tiled_(out, block, tile):
    for start in range(0, len(out), tile):
        end = min(start + tile, len(out))
        out[start:, start:end].addmm_(block[start:], block[start:end].mT)


def mirror_gram_(out, tile, lower_only):
    for start in range(0, len(out), tile):
        end = min(start + tile, len(out))
        if lower_only:
            diagonal = out[start:end, start:end]
            diagonal.copy_(diagonal.tril() + diagonal.tril(-1).mT)
        out[:start, start:end].copy_(out[start:end, :start].mT)


def tune_gram(out, block, tile):
    """Time each backend with the existing output buffer, before its first accumulation."""
    timings = {}
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for backend in ("full GEMM", "tiled GEMM", "cuBLAS SYRK"):
        out.zero_()
        start.record()
        if backend == "full GEMM":
            out.addmm_(block, block.mT)
        elif backend == "tiled GEMM":
            add_gram_tiled_(out, block, tile)
            mirror_gram_(out, tile, False)
        else:
            if not add_gram_lower_(out, block):
                continue
            mirror_gram_(out, tile, True)
        stop.record()
        stop.synchronize()
        timings[backend] = start.elapsed_time(stop)
    out.zero_()
    best = min(timings, key=timings.get)
    if timings[best] >= 0.95 * timings["full GEMM"]:
        return "full GEMM"
    return best
