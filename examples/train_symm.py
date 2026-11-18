"""Set PARAM_FILE to unsymmetrized weights from train.py before running."""

import logging
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import quantax as qtx
from quantax.symmetry import TotalSz, SpinInverse, C4v

ROOT = Path(__file__).resolve().parents[1]
L = 10
J1, J2 = 1.0, 0.5
DEPTH = 16  # Network settings must match the input weights.
CHANNELS = 16
KERNEL_SIZE = 3
USE_SINH = True
PARAM_FILE = ROOT / "params_20000.eqx"  # Set to the checkpoint in runs/<system>/<timestamp>/.
NSAMPLES = 10000
STEPS = 10000
STEP_SIZE = 1e-3
MAX_PARALLEL = 1024  # Limits forward batches, not all MinSR gradient/matrix memory.
OUTPUT_DIR = ROOT / "runs"


def setup_plot_font():
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
    except ValueError:
        for path in font_manager.findSystemFonts():
            try:
                if font_manager.get_font(path).family_name == "Arial":
                    font_manager.fontManager.addfont(path)
            except (OSError, RuntimeError):
                continue
        font_manager.findfont("Arial", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none"})


def main():
    start_time = perf_counter()
    system_name = f"j1j2_square_{L}x{L}_J1-{J1:g}_J2-{J2:g}"
    local_tz = ZoneInfo("Asia/Hong_Kong")
    timestamp = datetime.now(local_tz).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(OUTPUT_DIR) / system_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    logger = logging.getLogger("minsr.train_symm")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(run_dir / "training.log", mode="x", encoding="utf-8")
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    try:
        logger.info(f"Run directory: {run_dir}; timezone: {local_tz.key}")
        logger.info(f"System: {system_name}; periodic boundaries; total Sz=0")
        logger.info(f"Steps={STEPS}; samples={NSAMPLES}; max_parallel={MAX_PARALLEL}; step_size={STEP_SIZE}; reweight=1.0")
        logger.info(f"Network: ResSum(depth={DEPTH}, channels={CHANNELS}, kernel_size={KERNEL_SIZE}, use_sinh={USE_SINH})")
        logger.info(f"Symmetry: TotalSz(0) + SpinInverse() + C4v()")
        logger.info(f"Input weights: {Path(PARAM_FILE).resolve()}")
        if not Path(PARAM_FILE).is_file():
            raise FileNotFoundError(f"Set PARAM_FILE to an existing train.py checkpoint: {PARAM_FILE}")
        if STEPS < 1 or NSAMPLES < 1 or MAX_PARALLEL < 16:
            raise ValueError("Require STEPS >= 1, NSAMPLES >= 1 and MAX_PARALLEL >= 16 for this symmetry.")
        logger.info(f"Devices: {jax.devices()}")
        setup_plot_font()
        qtx.sites.Square(L)
        H = qtx.operator.Heisenberg(J=[J1, J2], n_neighbor=[1, 2], msr=True)
        net = qtx.nn.ResSum(DEPTH, CHANNELS, KERNEL_SIZE, use_sinh=USE_SINH)
        state = qtx.state.Variational(
            net, param_file=PARAM_FILE, symm=TotalSz() + SpinInverse() + C4v(), max_parallel=MAX_PARALLEL,
        )
        # Reweighted sampling for more efficient training (only for N <= 100).
        sampler = qtx.sampler.NeighborExchange(state, NSAMPLES, reweight=1.0, n_neighbor=[1, 2])
        optimizer = qtx.optimizer.MinSR(state, H)
        jax.block_until_ready(sampler.current_wf)
        energy = qtx.utils.DataTracer()
        VarE = qtx.utils.DataTracer()
        baseline = -199.0859 if (L, J1, J2) == (10, 1.0, 0.5) else None
        title = f"J1-J2 Heisenberg, square {L}x{L}, J1={J1:g}, J2={J2:g} (symm)"

        for n in range(STEPS):
            step_start = perf_counter()
            samples = sampler.sweep()
            step = optimizer.get_step(samples)
            state.update(step * STEP_SIZE)
            energy.append(optimizer.energy)
            VarE.append(optimizer.VarE)
            if (n + 1) % 10 == 0 or n + 1 == STEPS:
                for tracer, prefix, ylabel, reference in (
                    (energy, "energy", "Energy", baseline),
                    (VarE, "VarE", "Energy variance", None),
                ):
                    tracer.plot(batch=min(10, n + 1), start=-1000, baseline=reference)
                    plt.gca().lines[-1].set_marker("o")
                    plt.title(title)
                    plt.xlabel("Update index")
                    plt.ylabel(ylabel)
                    plt.tight_layout()
                    plt.savefig(run_dir / f"{prefix}_{system_name}_symm.pdf")
                    plt.close()
            if (n + 1) % 100 == 0 or n + 1 == STEPS:
                state.save(run_dir / f"params_{system_name}_symm_{n + 1}.eqx")
            jax.block_until_ready(state.ansatz)
            step_end = perf_counter()
            logger.info(
                f"E={float(optimizer.energy):.6f}, Var={float(optimizer.VarE):.6g}, "
                f"{n + 1}/{STEPS}, {step_end - step_start:.2f}, {step_end - start_time:.2f}"
            )
    except (Exception, KeyboardInterrupt):
        logger.exception(f"Training stopped after {perf_counter() - start_time:.2f} s")
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()
        plt.close("all")
    return run_dir


if __name__ == "__main__":
    main()
