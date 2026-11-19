import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import quantax as qtx
from quantax.symmetry import TotalSz
from quantax.utils import (
    create_run, close_run, log_devices, log_progress, setup_plot_font, save_training_plots,
)

L = 10
J1, J2 = 1.0, 0.5
DEPTH, CHANNELS, KERNEL_SIZE, USE_SINH = 16, 16, 3, True
NSAMPLES = 1000  # 10000
STEPS = 20000
STEP_SIZE = 1e-3
MAX_PARALLEL = 32768
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "runs"


def main():
    start_time = perf_counter()
    system_name = f"j1j2_square_{L}x{L}_J1-{J1:g}_J2-{J2:g}"
    run_dir, logger, handler = create_run(OUTPUT_DIR, system_name, "minsr.train")

    try:
        logger.info(f"Run directory: {run_dir}")
        logger.info(f"System: {system_name}; periodic boundaries; total Sz=0")
        logger.info(f"Steps={STEPS}; samples={NSAMPLES}; max_parallel={MAX_PARALLEL}; step_size={STEP_SIZE}; reweight=1.0")
        logger.info(f"Network: ResSum(depth={DEPTH}, channels={CHANNELS}, kernel_size={KERNEL_SIZE}, use_sinh={USE_SINH})")
        logger.info(f"Symmetry: TotalSz(0)")
        log_devices(logger, jax.devices())
        setup_plot_font()

        lattice = qtx.sites.Square(L)
        H = qtx.operator.Heisenberg(J=[J1, J2], n_neighbor=[1, 2], msr=True)
        net = qtx.nn.ResSum(DEPTH, CHANNELS, KERNEL_SIZE, use_sinh=USE_SINH)
        state = qtx.state.Variational(net, symm=TotalSz(), max_parallel=MAX_PARALLEL)

        # Reweighted sampling for more efficient training (only for N <= 100).
        sampler = qtx.sampler.NeighborExchange(state, NSAMPLES, reweight=1.0, n_neighbor=[1, 2])
        optimizer = qtx.optimizer.MinSR(state, H)
        jax.block_until_ready(sampler.current_wf)

        energy = qtx.utils.DataTracer()
        VarE = qtx.utils.DataTracer()
        baseline = -199.0859 if (L, J1, J2) == (10, 1.0, 0.5) else None
        title = f"J1-J2 Heisenberg, square {L}x{L}, J1={J1:g}, J2={J2:g}"

        for n in range(STEPS):
            step_start = perf_counter()
            samples = sampler.sweep()
            step = optimizer.get_step(samples)
            state.update(step * STEP_SIZE)

            energy.append(optimizer.energy)
            VarE.append(optimizer.VarE)
            if (n + 1) % 10 == 0 or n + 1 == STEPS:
                save_training_plots(energy, VarE, run_dir, title, n + 1, baseline=baseline)
            if (n + 1) % 100 == 0 or n + 1 == STEPS:
                state.save(run_dir / f"params_{n + 1}.eqx")

            jax.block_until_ready(state.ansatz)
            step_end = perf_counter()
            log_progress(
                logger, optimizer.energy, optimizer.VarE, n + 1, STEPS,
                step_end - step_start, step_end - start_time,
            )
    except (Exception, KeyboardInterrupt):
        logger.exception(f"Training stopped after {perf_counter() - start_time:.2f} s")
        raise
    finally:
        close_run(logger, handler)
        plt.close("all")
    return run_dir


if __name__ == "__main__":
    main()
