"""Time synchronized PyTorch training steps for one configuration per process."""

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
    parser.add_argument("--param-file", type=Path)
    parser.add_argument("--steps", type=int, default=5, help="Measured training steps (after warmup).")
    parser.add_argument("--warmup", type=int, default=2, help="Training steps excluded from the medians.")
    args = vars(parser.parse_args())
    symmetric, param_file, steps, warmup, thermal, sweep = (
        args.pop(k) for k in ("symmetric", "param_file", "steps", "warmup", "thermal_steps", "sweep_steps")
    )
    if steps < 1 or warmup < 1:
        parser.error("--steps and --warmup must be positive")
    config = Config(**args)
    thermal, sweep = config.sampling(thermal, sweep)
    hamiltonian, state = workflows._system(config, symmetric, param_file)
    sync = (lambda: torch.cuda.synchronize(state.device)) if state.device.type == "cuda" else (lambda: None)

    def timed(fn, *inputs):
        sync()
        start = perf_counter()
        out = fn(*inputs)
        sync()
        return perf_counter() - start, out

    setup, sampler = timed(workflows._sampler, config, state, 1.0, thermal, sweep)
    opt = qtx.optimizer.MinSR(state, hamiltonian, jacobian_batch=config.jacobian_batch)
    get_ebar = opt.get_Ebar
    eloc_seconds = 0.0

    def timed_ebar(samples):
        nonlocal eloc_seconds
        eloc_seconds, rhs = timed(get_ebar, samples)
        return rhs

    opt.get_Ebar = timed_ebar
    print(json.dumps(dict(config=asdict(config), symmetric=symmetric, checkpoint=param_file,
                          thermal=thermal, sweep=sweep, setup_seconds=setup, torch=torch.__version__,
                          device=torch.cuda.get_device_name(state.device) if state.device.type == "cuda" else "cpu"),
                     default=str), flush=True)
    rows = []
    if state.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(state.device)
    for i in range(warmup + steps):
        sampling, samples = timed(sampler.sweep)
        optimize, update = timed(opt.get_step, samples)
        if not update.isfinite().all():
            raise FloatingPointError(f"Nonfinite update at step {i + 1}")
        apply, _ = timed(state.update, update * config.step_size)
        row = dict(step=i + 1, warmup=i < warmup, sampling=sampling, local_energy=eloc_seconds,
                   minsr=optimize - eloc_seconds + apply, total=sampling + optimize + apply,
                   energy=opt.energy.item(), variance=opt.VarE.item())
        if i >= warmup:
            rows.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    print(json.dumps(dict(medians={k: median(row[k] for row in rows)
                                   for k in ("sampling", "local_energy", "minsr", "total")},
                          peak_mib=torch.cuda.max_memory_allocated(state.device) / 2**20
                          if state.device.type == "cuda" else None)), flush=True)


if __name__ == "__main__":
    main()
