"""Plain-text run logging shared by the examples."""

import logging
import os
import platform
import subprocess
from datetime import datetime
from itertools import count
from pathlib import Path


def create_run(output_dir, system_name, logger_name, log_file="training.log"):
    """Create a run directory and attach its exclusive log file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    system_dir = Path(output_dir) / system_name
    run_dir = system_dir / timestamp
    for index in count(1):
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            if not run_dir.exists():
                raise  # A parent path may be a file rather than a directory.
            run_dir = system_dir / f"{timestamp}_{index:02d}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(run_dir / log_file, mode="x", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    return run_dir, logger, handler


def close_run(logger, handler):
    """Close only the file handler attached for this run."""
    logger.removeHandler(handler)
    handler.close()


def _cpu_info():
    model = ""
    try:
        with Path("/proc/cpuinfo").open(encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                key, _, value = line.partition(":")
                if key.strip() == "model name" and value.strip():
                    model = value.strip()
                    break
    except OSError:
        pass
    model = model or platform.processor() or platform.machine() or "unavailable"
    cores = os.cpu_count() or "unknown"
    return f"{model} ({cores} logical cores)"


def _ram_total():
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return f"{pages * page_size / 2**30:.2f} GiB total"
    except (AttributeError, OSError, ValueError):
        pass
    return "unavailable"


def _gpu_inventory(devices):
    # Query installed NVIDIA hardware even when JAX is configured to use the CPU.
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=2,
        )
        gpus = []
        for line in result.stdout.splitlines():
            name, sep, memory = line.rpartition(",")
            if sep and name.strip() and memory.strip():
                gpus.append(f"{name.strip()} ({memory.strip()} VRAM)")
        if gpus:
            return "; ".join(gpus)
    except (OSError, subprocess.SubprocessError):
        pass
    gpus = [device.device_kind for device in devices if device.platform in {"gpu", "cuda", "rocm"}]
    return "; ".join(gpus) if gpus else "not detected by nvidia-smi or JAX"


def log_devices(logger, devices):
    """Log the active JAX devices and the hardware visible to the process."""
    logger.info(f"Devices: {devices}")
    logger.info(f"CPU: {_cpu_info()}")
    logger.info(f"GPU: {_gpu_inventory(devices)}")
    logger.info(f"RAM: {_ram_total()}")


def log_progress(logger, energy, variance, step, total_steps, step_seconds, elapsed_seconds):
    """Log energy, variance, completed steps, and durations in seconds."""
    logger.info(
        f"E={float(energy):.8f}, Var={float(variance):.4f}, "
        f"{step}/{total_steps}, {step_seconds:.2f}, {elapsed_seconds:.2f}"
    )
