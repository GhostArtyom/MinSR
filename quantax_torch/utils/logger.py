"""Plain-text run logging shared by the examples."""

import logging
import os
import platform
from datetime import datetime
from itertools import count
from pathlib import Path
from time import perf_counter


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
    import torch

    gpus = []
    for device in devices:
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            gpus.append(f"{props.name} ({props.total_memory / 2**30:.2f} GiB VRAM)")
    return "; ".join(gpus) if gpus else "not used"


def log_devices(logger, devices):
    """Log the active PyTorch devices and the hardware visible to the process."""
    logger.info(f"Devices: {devices}")
    logger.info(f"CPU: {_cpu_info()}")
    logger.info(f"GPU: {_gpu_inventory(devices)}")
    logger.info(f"RAM: {_ram_total()}")


def thermalization_progress(logger, start_time):
    """Log elapsed seconds since the last report and since the stage started."""
    previous_time = start_time

    def report(step, total):
        nonlocal previous_time
        end = perf_counter()
        seconds = end - previous_time if step else 0.0
        logger.info(f"Thermalization {step:d}/{total:d}, {seconds:.2f}, {end - start_time:.2f}")
        previous_time = end

    return report


def log_progress(logger, energy, variance, step, total_steps, step_seconds, elapsed_seconds):
    """Log energy, variance, completed steps, and durations in seconds."""
    logger.info(
        f"E={float(energy):.8f}, Var={float(variance):.4f}, "
        f"{step}/{total_steps}, {step_seconds:.2f}, {elapsed_seconds:.2f}"
    )
