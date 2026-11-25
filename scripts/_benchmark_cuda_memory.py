"""Timestamp PyTorch allocator statistics and CUDA driver headroom."""

from time import time_ns

import torch


def start_cuda_step(device):
    torch.cuda.reset_peak_memory_stats(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "driver_total_bytes": total,
        "driver_free_before_bytes": free,
        "allocated_before_bytes": torch.cuda.memory_allocated(device),
        "reserved_before_bytes": torch.cuda.memory_reserved(device),
        "started_unix_ns": time_ns(),
    }


def finish_cuda_step(device, memory):
    finished = time_ns()
    free, _ = torch.cuda.mem_get_info(device)
    memory.update(
        finished_unix_ns=finished,
        driver_free_after_bytes=free,
        allocated_peak_bytes=torch.cuda.max_memory_allocated(device),
        reserved_peak_bytes=torch.cuda.max_memory_reserved(device),
    )
    return memory
