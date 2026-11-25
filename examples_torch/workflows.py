import csv
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import quantax_torch as qtx
from quantax_torch.nn._cuda_support import periodic_cuda_issue
from quantax_torch.sampler.metropolis import _counted_exchange_forward
from quantax_torch.utils import (
    DataTracer, close_run, create_run, log_devices, log_progress, save_training_plots, setup_plot_font,
)
from quantax_torch.utils.logger import thermalization_progress
from ._cuda_tuning import first_cuda_measurement, first_cuda_step


def _periodic_forward_issue(config, device, symmetric):
    if not (config.projected_periodic_conv if symmetric else config.ordinary_periodic_conv):
        return "disabled"
    issue = periodic_cuda_issue(device)
    if issue is not None:
        return issue
    if qtx.get_params_dtype() != torch.float32 or torch.is_autocast_enabled("cuda"):
        return "requires FP32 without autocast"
    # Leave ordinary shapes outside the tuned channel tiles to channels-last cuDNN.
    plain_shape = ((config.kernel_size == 2 and 8 <= config.channels <= 64)
                   or (config.kernel_size >= 3 and config.channels % 16 == 0))
    if not symmetric and not plain_shape:
        return "shape outside tuned range"
    return None


def _cuda_inference(config, device, symmetric):
    if _periodic_forward_issue(config, device, symmetric) is not None:
        return None
    from quantax_torch.nn._periodic_cuda import res_sum_forward

    return res_sum_forward


def _local_energy_graph_issue(config, inference):
    if not config.projected_local_energy_graph:
        return "disabled"
    if inference is None:
        return "no periodic forward"
    return None


def _hamiltonian(config, device, symmetric, inference):
    local_energy_graph = symmetric and _local_energy_graph_issue(config, inference) is None
    return qtx.operator.Heisenberg(
        J=[config.J1, config.J2], n_neighbor=[1, 2], msr=True,
        local_energy_graph=local_energy_graph,
        reuse_local_energy_pool=local_energy_graph,
        tune_local_energy=(not symmetric and device.type == "cuda")
        or (local_energy_graph and config.projected_forward_batch is None),
        local_energy_forward_batch=config.local_energy_forward_batch,
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
    inference = _cuda_inference(config, device, symmetric)
    hamiltonian = _hamiltonian(config, device, symmetric, inference)
    symm = qtx.symmetry.TotalSz()
    if symmetric:
        symm += qtx.symmetry.SpinInverse() + qtx.symmetry.C4v()
    if config.local_energy_forward_batch is not None and config.local_energy_forward_batch % symm.nsymm:
        raise ValueError(f"local_energy_forward_batch must be a multiple of the orbit size {symm.nsymm}.")
    if device.type == "cuda" and checkpoint is None:
        # Keep initialization draws from advancing the sampler's generator.
        generator = torch.Generator(device=device).manual_seed(config.seed)
        net = qtx.nn.ResSum(**config.network, device=device, generator=generator)
    else:
        net = qtx.nn.ResSum(**config.network)
    if device.type == "cuda" and not symmetric:
        net.to(memory_format=torch.channels_last)
    if symmetric and device.type == "cuda":
        inference_batch = config.projected_forward_batch or max(1024, symm.nsymm)
    else:
        inference_batch = 2048 if inference is not None else None
    state = qtx.state.Variational(
        net, symm=symm, max_parallel=config.max_parallel, device=device, inference_batch=inference_batch,
        inference_fn=inference,
    )
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
    logger.info(f"System: {config.system} | periodic boundaries | total Sz=0")
    logger.info(
        f"Steps={steps:d} | samples={config.nsamples:d} | max_parallel={config.max_parallel:d} | reweight={reweight:g}"
    )
    logger.info(f"Sampling: thermal_steps={thermal:d} | sweep_steps={sweep:d}")
    if state.device.type == "cuda":
        forward = "periodic IEEE FP32"
        if state.inference_fn is None:
            reason = _periodic_forward_issue(config, state.device, symmetric)
            forward = f"cuDNN ({reason})"
        batch = f" | cap={state.forward_batch} inputs" if state.inference_batch is not None else ""
        logger.info(f"CUDA forward: {forward}{batch}")
    logger.info(f"Network: ResSum {config.network}")
    logger.info(f"Symmetry: {_sector(symmetric)}")
    logger.info(f"PyTorch {torch.__version__} | seed={config.seed:d}")
    if checkpoint is not None:
        logger.info(f"Input weights: {checkpoint}")
    elif state.device.type == "cuda":
        logger.info(f"Initial weights: generated on {state.device}")
    else:
        logger.info("Initial weights: generated on CPU")
    log_devices(logger, [state.device])


def _sampler(config, state, reweight, thermal, sweep, thermal_progress):
    compact = state.device.type == "cuda" and (state.nsymm > 1 or _counted_exchange_forward(state) is not None)
    return qtx.sampler.NeighborExchange(
        state, config.nsamples, reweight=reweight, n_neighbor=[1, 2], thermal_steps=thermal, sweep_steps=sweep,
        cuda_graph=state.nsymm == 1 or config.projected_exchange_graph,
        skip_unchanged=compact,
        thermal_progress=thermal_progress,
    )


def _log_cuda_paths(logger, config, state, sampler, hamiltonian, optimizer=None):
    if state.device.type != "cuda":
        return
    if sampler._compact_key is not None:
        capacity = getattr(sampler._cuda_graph, "capacity", None)
        detail = f" | {capacity} chains/group" if capacity is not None else ""
        logger.info(f"CUDA sampling: compacted graph{detail}")
    elif sampler._cuda_graph is not None:
        logger.info("CUDA sampling: graph")
    elif state.nsymm > 1:
        if not config.projected_exchange_graph:
            reason = "graph disabled"
        elif state.inference_fn is None:
            reason = "no periodic forward"
        else:
            reason = "no graph"
        logger.info(f"CUDA sampling: eager compacted ({reason})")
    if state.nsymm > 1:
        if getattr(hamiltonian, "_local_energy_graph_used", False):
            graph = "cached graph" if hamiltonian._local_energy_workspaces else "graph"
            batch = hamiltonian._local_energy_batch or state.forward_batch // state.nsymm
            logger.info(f"CUDA local energy: {graph} | {batch * state.nsymm} orbit members/batch")
        else:
            reason = _local_energy_graph_issue(config, state.inference_fn)
            detail = f" ({reason})" if reason is not None else ""
            batch = config.local_energy_forward_batch
            size = f" | {batch} orbit members/batch" if batch is not None else ""
            logger.info(f"CUDA local energy: eager{detail}{size}")
    elif (batch := getattr(hamiltonian, "_local_energy_batch", None)) is not None:
        logger.info(f"CUDA local energy: eager | {batch} inputs/batch")
    if optimizer is not None:
        optimizer.log_backends(logger)


def train(config, *, steps, param_file=None, symmetric=False, thermal_steps=None, sweep_steps=None):
    if steps < 1:
        raise ValueError("steps must be positive.")
    if symmetric and param_file is None:
        raise ValueError("Symmetry-projected training requires --param-file.")
    checkpoint = None if param_file is None else _checkpoint(param_file)
    thermal, sweep = config.sampling(thermal_steps, sweep_steps)
    suffix = "_symm" if symmetric else ""
    start = perf_counter()
    run, logger, handler = create_run(config.output_dir, config.system, f"minsr.torch.train{suffix}",
                                      log_file=f"training{suffix}.log")
    previous_backend = None
    try:
        hamiltonian, state = _system(config, symmetric, checkpoint)
        if state.device.type == "cuda" and config.eigh_backend is not None:
            previous_backend = torch.backends.cuda.preferred_linalg_library()
            torch.backends.cuda.preferred_linalg_library(config.eigh_backend)
        _header(logger, config, state, steps, thermal, sweep, symmetric, 1.0, checkpoint)
        if state.device.type == "cuda":
            logger.info(f"CUDA eigensolver: {torch.backends.cuda.preferred_linalg_library().name}")
        logger.info(f"Step size: {config.step_size:g} | Jacobian batch={config.jacobian_batch or 'auto'}")
        optimizer = qtx.optimizer.MinSR(
            state, hamiltonian, jacobian_batch=config.jacobian_batch, logger=logger,
        )
        setup_plot_font()
        sampler = _sampler(config, state, 1.0, thermal, sweep, thermalization_progress(logger, start))
        energies, variances = DataTracer(), DataTracer()
        title = f"J1-J2 Heisenberg, square {config.L}x{config.L}, J1={config.J1:g}, J2={config.J2:g}"
        baseline = -199.0859 if (config.L, config.J1, config.J2) == (10, 1.0, 0.5) else None
        if symmetric:
            title += " (symm)"
        tune = state.device.type == "cuda" and torch.version.cuda is not None
        for n in range(1, steps + 1):
            step_start = perf_counter()
            samples = sampler.sweep()
            if tune and n == 1:
                update = first_cuda_step(config, optimizer, samples, symmetric, logger)
            else:
                update = optimizer.get_step(samples)
            if not bool(update.isfinite().all()):
                raise FloatingPointError(f"Non-finite update at step {n}.")
            state.update(update * config.step_size)
            if n == 1:
                _log_cuda_paths(logger, config, state, sampler, hamiltonian, optimizer)
            energy, variance = optimizer.energy.item(), optimizer.VarE.item()
            energies.append(energy)
            variances.append(variance)
            plot_every = 10 if n <= 100 else 100
            if n % plot_every == 0 or n == steps:
                save_training_plots(energies, variances, run, title, n, baseline=baseline, suffix=suffix)
            if n % 100 == 0 or n == steps:
                state.save(run / f"params{suffix}_{n}.pt", metadata={
                    "network": config.network, "L": config.L, "system": config.system,
                    "symmetry": _sector(symmetric), "step": n, "seed": config.seed,
                })
            end = perf_counter()
            log_progress(logger, energy, variance, n, steps, end - step_start, end - start)
    except (Exception, KeyboardInterrupt):
        logger.exception(f"Training stopped after {perf_counter() - start:.2f} s")
        raise
    finally:
        if previous_backend is not None:
            torch.backends.cuda.preferred_linalg_library(previous_backend)
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
        sampler = _sampler(config, state, 2.0, thermal, sweep, thermalization_progress(logger, start))
        tune = getattr(hamiltonian, "_tune_local_energy", False)
        fields = ["step", "energy", "second_moment", "mean_energy", "mean_variance", "step_seconds", "elapsed_seconds"]
        sum_energy = 0.0
        sum_second = 0.0
        with (run / "measurement.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            stream.flush()
            for i in range(1, steps + 1):
                step_start = perf_counter()
                samples = sampler.sweep()
                if tune and i == 1:
                    energy, second = first_cuda_measurement(config, state, hamiltonian, samples, logger)
                else:
                    energy, second = hamiltonian.expectation(state, samples, return_var=True)
                if i == 1:
                    _log_cuda_paths(logger, config, state, sampler, hamiltonian)
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
            "inference_batch": state.forward_batch,
            "local_energy_batch": (getattr(hamiltonian, "_local_energy_batch", None) or 0) * state.nsymm
                                  or state.forward_batch,
            "projected_periodic_conv": state.inference_fn is not None,
            "projected_exchange_graph": sampler._compact_key is not None,
            "projected_local_energy_graph": getattr(hamiltonian, "_local_energy_graph_used", False),
            "thermal_steps": thermal, "sweep_steps": sweep,
            "energy": mean_energy, "energy_per_site": mean_energy / config.L**2, "variance": mean_variance,
            "seconds": perf_counter() - start, "seed": config.seed, "device": str(state.device),
        }
        (run / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (Exception, KeyboardInterrupt):
        logger.exception(f"Measurement stopped after {perf_counter() - start:.2f} s")
        raise
    finally:
        close_run(logger, handler)
    return run
