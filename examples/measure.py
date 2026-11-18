"""
Choose the network from
1: 16 layers, 16 channels every layer, 3x3 kernels 34944 parameters in total
2: 16 layers, 32 channels every layer, 3x3 kernels, 139008 parameters in total
3: 30 layers, 64 channels every layer, 3x3 kernels, 1071488 parameters in total

The variational energies of these networks correspond to the results of ResNet2(MinSR) 
in Fig. 2b
"""

import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import jax
import quantax as qtx
from quantax.symmetry import TotalSz, SpinInverse, C4v

ROOT = Path(__file__).resolve().parents[1]
network_idx = 1
# Each preset specifies depth, channels, and the matching supplied weights.
NETWORKS = {
    1: (16, 16, "square10x10_1616.eqx"),
    2: (16, 32, "square10x10_1632.eqx"),
    3: (30, 64, "square10x10_3064.eqx"),
}
KERNEL_SIZE = 3
USE_SINH = True  # Network settings must match the input weights.
L = 10
J1, J2 = 1.0, 0.5
PARAM_FILE = None  # None uses the supplied weights selected by network_idx.
NSAMPLES = 10000
STEPS = 100
MAX_PARALLEL = 1024
THERMAL_STEPS = 100 * L * L
SWEEP_STEPS = 10 * L * L
OUTPUT_DIR = ROOT / "runs"


def main():
    start_time = perf_counter()
    system_name = f"j1j2_square_{L}x{L}_J1-{J1:g}_J2-{J2:g}"
    local_tz = ZoneInfo("Asia/Hong_Kong")
    timestamp = datetime.now(local_tz).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(OUTPUT_DIR) / system_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    logger = logging.getLogger("minsr.measure")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(run_dir / "measurement.log", mode="x", encoding="utf-8")
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    try:
        logger.info(f"Run directory: {run_dir}; timezone: {local_tz.key}")
        logger.info(f"System: {system_name}; periodic boundaries; total Sz=0")
        logger.info(f"Steps={STEPS}; samples={NSAMPLES}; max_parallel={MAX_PARALLEL}; reweight=2.0")
        logger.info(f"Sampling: thermal_steps={THERMAL_STEPS}; sweep_steps={SWEEP_STEPS}")
        logger.info(f"Symmetry: TotalSz(0) + SpinInverse() + C4v()")
        if network_idx not in NETWORKS:
            raise ValueError("Wrong network index.")
        if STEPS < 1 or NSAMPLES < 1 or MAX_PARALLEL < 16:
            raise ValueError("Require STEPS >= 1, NSAMPLES >= 1 and MAX_PARALLEL >= 16 for this symmetry.")
        depth, channels, filename = NETWORKS[network_idx]
        param_file = Path(PARAM_FILE) if PARAM_FILE is not None else ROOT / "params" / filename
        logger.info(f"Network: index={network_idx}; ResSum(depth={depth}, channels={channels}, kernel_size={KERNEL_SIZE}, use_sinh={USE_SINH})")
        logger.info(f"Input weights: {param_file.resolve()}")
        if not param_file.is_file():
            raise FileNotFoundError(f"Measurement weights do not exist: {param_file}")
        logger.info(f"Devices: {jax.devices()}")
        qtx.sites.Square(L)
        H = qtx.operator.Heisenberg(J=[J1, J2], n_neighbor=[1, 2], msr=True)
        net = qtx.nn.ResSum(depth, channels, KERNEL_SIZE, use_sinh=USE_SINH)
        state = qtx.state.Variational(
            net, param_file=param_file, symm=TotalSz() + SpinInverse() + C4v(), max_parallel=MAX_PARALLEL,
        )
        sampler = qtx.sampler.NeighborExchange(
            state, NSAMPLES, thermal_steps=THERMAL_STEPS, sweep_steps=SWEEP_STEPS, n_neighbor=[1, 2],
        )
        jax.block_until_ready(sampler.current_wf)
        energy_data = qtx.utils.DataTracer()
        second_moment_data = qtx.utils.DataTracer()
        fields = ["step", "energy", "second_moment", "mean_energy", "mean_variance", "step_seconds", "elapsed_seconds"]
        with (run_dir / f"measurement_{system_name}.csv").open("x", newline="", encoding="utf-8") as data_file:
            writer = csv.DictWriter(data_file, fieldnames=fields)
            writer.writeheader()
            data_file.flush()
            for i in range(STEPS):
                step_start = perf_counter()
                samples = sampler.sweep()
                # return_var=True returns the second moment, not the centered variance.
                energy, second_moment = H.expectation(state, samples, return_var=True)
                energy_data.append(energy)
                second_moment_data.append(second_moment)
                mean_energy = float(energy_data.mean())
                mean_variance = float(second_moment_data.mean()) - mean_energy**2
                step_end = perf_counter()
                writer.writerow(dict(zip(fields, [
                    i + 1, energy, second_moment, mean_energy, mean_variance,
                    step_end - step_start, step_end - start_time,
                ])))
                data_file.flush()
                logger.info(
                    f"E={mean_energy:.6f}, Var={mean_variance:.6g}, "
                    f"{i + 1}/{STEPS}, {step_end - step_start:.2f}, {step_end - start_time:.2f}"
                )

        summary = {
            "system": system_name, "network_idx": network_idx,
            "network": {"depth": depth, "channels": channels, "kernel_size": KERNEL_SIZE, "use_sinh": USE_SINH},
            "param_file": str(param_file.resolve()), "symmetry": "TotalSz(0) + SpinInverse() + C4v()",
            "nsamples": NSAMPLES, "steps": STEPS, "max_parallel": MAX_PARALLEL,
            "thermal_steps": THERMAL_STEPS, "sweep_steps": SWEEP_STEPS,
            "energy": mean_energy, "energy_per_site": mean_energy / (L * L), "variance": mean_variance,
            "seconds": perf_counter() - start_time,
        }
        (run_dir / f"summary_{system_name}.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
    except (Exception, KeyboardInterrupt):
        logger.exception(f"Measurement stopped after {perf_counter() - start_time:.2f} s")
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()
    return run_dir


if __name__ == "__main__":
    main()
