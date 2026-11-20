import csv
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import quantax_torch as qtx
from quantax_torch.utils import (
    DataTracer, close_run, create_run, log_devices, log_progress, save_training_plots, setup_plot_font,
)


def _system(config, symmetric, checkpoint=None):
    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Select --device cpu for a CPU run.")
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
    qtx.set_random_seed(config.seed)
    qtx.sites.Square(config.L)
    hamiltonian = qtx.operator.Heisenberg(J=[config.J1, config.J2], n_neighbor=[1, 2], msr=True)
    symm = qtx.symmetry.TotalSz()
    if symmetric:
        symm += qtx.symmetry.SpinInverse() + qtx.symmetry.C4v()
    net = qtx.nn.ResSum(**config.network)
    state = qtx.state.Variational(net, symm=symm, max_parallel=config.max_parallel, device=device)
    if checkpoint is not None:
        if checkpoint.suffix == ".eqx":
            qtx.state.load_res_sum_eqx(checkpoint, net)
        else:
            network = state.load(checkpoint).get("network")
            if network is not None and network != config.network:
                raise ValueError(f"Checkpoint network {network} does not match {config.network}.")
    return hamiltonian, state


def _checkpoint(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input weights do not exist: {path}")
    return path


def _sector(symmetric):
    return "TotalSz(0) + SpinInverse() + C4v()" if symmetric else "TotalSz(0)"


def _header(logger, config, state, steps, thermal, sweep, symmetric, reweight, checkpoint=None):
    logger.info("System: %s | periodic boundaries | total Sz=0", config.system)
    logger.info("Steps=%d | samples=%d | max_parallel=%d | reweight=%g",
                steps, config.nsamples, config.max_parallel, reweight)
    logger.info("Sampling: thermal_steps=%d | sweep_steps=%d", thermal, sweep)
    logger.info("Network: ResSum %s", config.network)
    logger.info("Symmetry: %s", _sector(symmetric))
    logger.info("PyTorch %s | seed=%d", torch.__version__, config.seed)
    if checkpoint is not None:
        logger.info("Input weights: %s", checkpoint)
    log_devices(logger, [state.device])


def _sampler(config, state, reweight, thermal, sweep):
    return qtx.sampler.NeighborExchange(
        state, config.nsamples, reweight=reweight, n_neighbor=[1, 2], thermal_steps=thermal, sweep_steps=sweep,
    )


def train(config, *, steps, param_file=None, symmetric=False, thermal_steps=None, sweep_steps=None):
    if steps < 1:
        raise ValueError("steps must be positive.")
    if symmetric and param_file is None:
        raise ValueError("Symmetry continuation requires --param-file.")
    checkpoint = None if param_file is None else _checkpoint(param_file)
    thermal, sweep = config.sampling(thermal_steps, sweep_steps)
    suffix = "_symm" if symmetric else ""
    start = perf_counter()
    run, logger, handler = create_run(config.output_dir, config.system, f"minsr.torch.train{suffix}",
                                      log_file=f"training{suffix}.log")
    try:
        hamiltonian, state = _system(config, symmetric, checkpoint)
        _header(logger, config, state, steps, thermal, sweep, symmetric, 1.0, checkpoint)
        logger.info("Step size: %g | Jacobian batch=%d", config.step_size, config.jacobian_batch)
        setup_plot_font()
        sampler = _sampler(config, state, 1.0, thermal, sweep)
        optimizer = qtx.optimizer.MinSR(state, hamiltonian, jacobian_batch=config.jacobian_batch)
        energies, variances = DataTracer(), DataTracer()
        title = f"J1-J2 Heisenberg, square {config.L}x{config.L}, J1={config.J1:g}, J2={config.J2:g}"
        baseline = -199.0859 if (config.L, config.J1, config.J2) == (10, 1.0, 0.5) else None
        if symmetric:
            title += " (symm)"
        for n in range(1, steps + 1):
            step_start = perf_counter()
            samples = sampler.sweep()
            update = optimizer.get_step(samples)
            if not bool(update.isfinite().all()):
                raise FloatingPointError(f"Non-finite update at step {n}.")
            state.update(update * config.step_size)
            energy, variance = optimizer.energy.item(), optimizer.VarE.item()
            energies.append(energy)
            variances.append(variance)
            if n % 10 == 0 or n == steps:
                save_training_plots(energies, variances, run, title, n, baseline=baseline, suffix=suffix)
            if n % 100 == 0 or n == steps:
                state.save(run / f"params{suffix}_{n}.pt", metadata={
                    "network": config.network, "L": config.L, "system": config.system,
                    "symmetry": _sector(symmetric), "step": n, "seed": config.seed,
                })
            end = perf_counter()
            log_progress(logger, energy, variance, n, steps, end - step_start, end - start)
    except (Exception, KeyboardInterrupt):
        logger.exception("Training stopped after %.2f s", perf_counter() - start)
        raise
    finally:
        close_run(logger, handler)
        plt.close("all")
    return run


def measure(config, *, steps, param_file, preset=None, thermal_steps=None, sweep_steps=None):
    if steps < 1:
        raise ValueError("steps must be positive.")
    checkpoint = _checkpoint(param_file)
    thermal, sweep = config.sampling(thermal_steps, sweep_steps, measurement=True)
    start = perf_counter()
    run, logger, handler = create_run(config.output_dir, config.system, "minsr.torch.measure", "measurement.log")
    try:
        hamiltonian, state = _system(config, True, checkpoint)
        _header(logger, config, state, steps, thermal, sweep, True, 2.0, checkpoint)
        sampler = _sampler(config, state, 2.0, thermal, sweep)
        fields = ["step", "energy", "second_moment", "mean_energy", "mean_variance", "step_seconds", "elapsed_seconds"]
        sum_energy = 0.0
        sum_second = 0.0
        with (run / "measurement.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            stream.flush()
            for i in range(1, steps + 1):
                step_start = perf_counter()
                energy, second = hamiltonian.expectation(state, sampler.sweep(), return_var=True)
                sum_energy += energy
                sum_second += second
                mean_energy = sum_energy / i
                mean_variance = sum_second / i - mean_energy**2
                end = perf_counter()
                writer.writerow([i, energy, second, mean_energy, mean_variance, end - step_start, end - start])
                stream.flush()
                log_progress(logger, mean_energy, mean_variance, i, steps, end - step_start, end - start)
        summary = {
            "system": config.system, "network_idx": preset, "network": config.network,
            "param_file": str(checkpoint), "symmetry": _sector(True),
            "nsamples": config.nsamples, "steps": steps, "max_parallel": config.max_parallel,
            "thermal_steps": thermal, "sweep_steps": sweep,
            "energy": mean_energy, "energy_per_site": mean_energy / config.L**2, "variance": mean_variance,
            "seconds": perf_counter() - start, "seed": config.seed, "device": str(state.device),
        }
        (run / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (Exception, KeyboardInterrupt):
        logger.exception("Measurement stopped after %.2f s", perf_counter() - start)
        raise
    finally:
        close_run(logger, handler)
    return run
