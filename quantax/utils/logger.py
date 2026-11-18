"""Plain-text run logging shared by the examples."""

import logging
from datetime import datetime
from pathlib import Path


def create_run(output_dir, system_name, logger_name, log_file="training.log"):
    """Create a run directory and attach its exclusive log file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / system_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(run_dir / log_file, mode="x", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return run_dir, logger, handler


def close_run(logger, handler):
    """Close only the file handler attached for this run."""
    logger.removeHandler(handler)
    handler.close()


def log_progress(logger, energy, variance, step, total_steps, step_seconds, elapsed_seconds):
    """Log energy, variance, completed steps, and durations in seconds."""
    logger.info(
        f"E={float(energy):.6f}, Var={float(variance):.6g}, "
        f"{step}/{total_steps}, {step_seconds:.2f}, {elapsed_seconds:.2f}"
    )
