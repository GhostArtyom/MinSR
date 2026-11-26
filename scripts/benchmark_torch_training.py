"""Time synchronized PyTorch training steps or compare CUDA batch combinations."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter


def main():
    from examples_torch import workflows
    from examples_torch.run_pipeline import Config, add_common_options, add_sampling_options
    import quantax_torch as qtx
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_options(parser)
    add_sampling_options(parser)
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--matrix", action="store_true", help="Compare batch combinations for all three CUDA stages.")
    parser.add_argument("--save-tuning", action="store_true", help="Save verified matrix winners to the CUDA cache.")
    parser.add_argument("--reweight", type=float, choices=(1.0, 2.0),
                        help="Sampling exponent for an individual training benchmark. Default: 1.0.")
    parser.add_argument("--param-file", type=Path)
    parser.add_argument("--steps", type=int, default=5, help="Measured training steps (after warmup).")
    parser.add_argument("--warmup", type=int, default=2, help="Training steps excluded from the medians.")
    args = vars(parser.parse_args())
    symmetric, matrix, save_tuning, reweight, param_file, steps, warmup, thermal, sweep = (
        args.pop(k) for k in ("symmetric", "matrix", "save_tuning", "reweight", "param_file", "steps", "warmup",
                              "thermal_steps", "sweep_steps")
    )
    if steps < 1 or warmup < 1:
        parser.error("--steps and --warmup must be positive")
    config = Config(**args)
    if matrix:
        if reweight is not None:
            parser.error("--reweight applies only to an individual training benchmark")
        if (torch.device(config.device).type != "cuda" or torch.version.cuda is None
                or not torch.cuda.is_available()):
            parser.error("--matrix requires an NVIDIA CUDA device")
        if symmetric or any(value is not None for value in (
            config.jacobian_batch, config.projected_forward_batch, config.local_energy_forward_batch,
            config.eigh_backend,
        )):
            parser.error("--matrix sets the stages, batch sizes and eigensolver itself")
        from ._benchmark_torch_matrix import run_matrix

        return run_matrix(config, param_file, thermal, sweep, warmup, steps, save_tuning)
    if save_tuning:
        parser.error("--save-tuning requires --matrix")
    from scripts._benchmark_cuda_memory import finish_cuda_step, start_cuda_step

    thermal, sweep = config.sampling(thermal, sweep)
    if torch.device(config.device).type == "cuda" and config.eigh_backend is not None:
        torch.backends.cuda.preferred_linalg_library(config.eigh_backend)
    hamiltonian, state = workflows._system(config, symmetric, param_file)
    sync = (lambda: torch.cuda.synchronize(state.device)) if state.device.type == "cuda" else (lambda: None)

    def timed(fn, *inputs):
        sync()
        start = perf_counter()
        out = fn(*inputs)
        sync()
        return perf_counter() - start, out

    reweight = 1.0 if reweight is None else reweight
    setup, sampler = timed(workflows._sampler, config, state, reweight, thermal, sweep, None)
    opt = qtx.optimizer.MinSR(state, hamiltonian, jacobian_batch=config.jacobian_batch)
    get_ebar = opt.get_Ebar
    eloc_seconds = 0.0

    def timed_ebar(samples):
        nonlocal eloc_seconds
        eloc_seconds, rhs = timed(get_ebar, samples)
        return rhs

    opt.get_Ebar = timed_ebar
    print(json.dumps(dict(config=asdict(config), symmetric=symmetric, checkpoint=param_file, reweight=reweight,
                          thermal=thermal, sweep=sweep, setup_seconds=setup, torch=torch.__version__,
                          device=torch.cuda.get_device_name(state.device) if state.device.type == "cuda" else "cpu"),
                     default=str), flush=True)
    rows = []
    memory_rows = []
    cuda = state.device.type == "cuda"
    for i in range(warmup + steps):
        memory = start_cuda_step(state.device) if cuda else None
        sampling, samples = timed(sampler.sweep)
        optimize, update = timed(opt.get_step, samples)
        if not update.isfinite().all():
            raise FloatingPointError(f"Nonfinite update at step {i + 1}")
        apply, _ = timed(state.update, update * config.step_size)
        row = dict(step=i + 1, warmup=i < warmup, sampling=sampling, local_energy=eloc_seconds,
                   minsr=optimize - eloc_seconds, update=apply, total=sampling + optimize + apply,
                   energy=opt.energy.item(), variance=opt.VarE.item())
        weights = samples.reweight_factor.to(torch.float64)
        row["importance_effective_samples"] = (weights.sum().square() / weights.square().sum()).item()
        row["parameter_update_norm"] = torch.linalg.vector_norm(update[1:]).item() * config.step_size
        if memory is not None:
            row["cuda_memory"] = finish_cuda_step(state.device, memory)
            row["jacobian_batch"] = opt._last_jacobian_batch
            local_batch = hamiltonian._local_energy_batch
            row["local_energy_forward_batch"] = None if local_batch is None else local_batch * state.nsymm
            row["derivative_cache"] = opt._last_cache_plan
            memory_rows.append(memory)
        if i >= warmup:
            rows.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    summary = {
        "medians": {k: median(row[k] for row in rows) for k in ("sampling", "local_energy", "minsr", "update", "total")},
        "peak_mib": max((row["allocated_peak_bytes"] / 2**20 for row in memory_rows), default=None),
        "peak_reserved_mib": max((row["reserved_peak_bytes"] / 2**20 for row in memory_rows), default=None),
    }
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
