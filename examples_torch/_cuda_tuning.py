"""Persist CUDA performance choices across training and measurement runs."""

import hashlib
import json
import os
from contextlib import suppress
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch


def _cache_file(config, state, device, stage):
    gpu = torch.cuda.get_device_properties(device)
    key = {
        "version": 3,
        "gpu": [gpu.name, str(getattr(gpu, "uuid", "")), gpu.total_memory, gpu.major, gpu.minor],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "eigh": None if stage == "measure" else torch.backends.cuda.preferred_linalg_library().name,
        "system": [config.L, config.J1, config.J2],
        "network": config.network,
        "dtype": str(state.dtype),
        "parameter_dtype": str(next(state.parameters()).dtype),
        "samples": config.nsamples,
        "stage": stage,
        "max_parallel": config.max_parallel,
        "inference": [state.inference_fn is not None, state.forward_batch],
        "forward_batch": config.projected_forward_batch,
        "local_energy_graph": config.projected_local_energy_graph,
        "exchange_graph": config.projected_exchange_graph,
    }
    if config.local_energy_forward_batch is not None:
        key["fixed_local_energy_batch"] = config.local_energy_forward_batch
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    directory = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return directory / "minsr" / "torch-cuda-tuning" / f"{digest}.json", key


def _load_record(path):
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def _read_record(path, key):
    record = _load_record(path)
    return record if record.get("key") == key else {}


def _write_record(path, record, logger):
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, encoding="utf-8", mode="w", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
        logger.info(f"CUDA tuning saved: {path}")
    except (OSError, ValueError) as error:
        logger.warning(f"Could not save CUDA tuning to {path}: {error}")
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def _batch_profile(key):
    if not isinstance(key, dict):
        return None
    gpu, system, network = (key.get(name) for name in ("gpu", "system", "network"))
    if not (isinstance(gpu, list) and len(gpu) == 5
            and isinstance(system, list) and len(system) == 3
            and isinstance(network, dict)):
        return None
    return key | {
        "gpu": gpu[:1] + gpu[2:],
        "system": system[0],
        "network": {name: value for name, value in network.items() if name not in ("depth", "channels")},
        "samples": None,
    }


def _nearby_batch(path, key):
    profile = _batch_profile(key)
    count, depth, channels = key["samples"], key["network"]["depth"], key["network"]["channels"]
    nearest = None
    for candidate in sorted(path.parent.glob("*.json")):
        if candidate == path:
            continue
        record = _load_record(candidate)
        previous = record.get("key")
        if _batch_profile(previous) != profile:
            continue
        batch = record.get("jacobian_batch")
        previous_count = previous.get("samples")
        previous_depth, previous_channels = (previous["network"].get(name) for name in ("depth", "channels"))
        if not all(type(value) is int and value > 0
                   for value in (batch, previous_count, previous_depth, previous_channels)):
            continue
        sizes = ((count, previous_count), (depth, previous_depth), (channels, previous_channels))
        if batch > previous_count or any(max(pair) > 2 * min(pair) for pair in sizes):
            continue
        distances = [max(pair) / min(pair) for pair in sizes]
        score = (max(distances), sum(distances), previous["gpu"][1] != key["gpu"][1])
        # Larger models can need more activation storage and wider derivative blocks.
        growth = max(1, depth / previous_depth) * max(1, (channels / previous_channels)**2)
        hint = min(count, max(1, int(batch / growth)))
        if nearest is None or score < nearest[0]:
            nearest = score, hint
    return None if nearest is None else nearest[1]


def _initial_batch(optimizer, samples, path, key, logger):
    count = len(samples.spins)
    estimate = int(optimizer._derivative_batch(samples.spins))
    hint = _nearby_batch(path, key)
    # A pessimistic estimate can subtract an entire Jacobian even for the
    # unchunked path, which produces its derivative blocks one at a time.
    forward_batch = optimizer.state.forward_batch
    forward_samples = count if forward_batch is None else forward_batch // optimizer.state.nsymm
    probe = min(count, max(1, forward_samples), 32)
    batch = max(probe, estimate if hint is None else hint)
    logger.info(f"CUDA Jacobian batch initial: {batch} | estimate={estimate} | nearby={hint}")
    return batch


def _evaluate_local_energy(state, hamiltonian, samples, evaluate, batch, logger):
    if type(batch) is not int or not hamiltonian._set_local_energy_batch(state, samples.spins.device, batch):
        return evaluate(samples)

    maximum = state._maximum.clone()
    try:
        return evaluate(samples)
    except torch.cuda.OutOfMemoryError:
        logger.warning("Cached CUDA local-energy batch ran out of memory. Retrying with the default batch.")
    if hamiltonian._local_energy_workspaces is not None:
        hamiltonian._local_energy_workspaces.clear()
    base = state.forward_batch // state.nsymm
    hamiltonian._local_energy_batch = min(base, 2048) if state.nsymm == 1 else base
    state._maximum.copy_(maximum)
    torch.cuda.empty_cache()
    return evaluate(samples)


@torch.no_grad()
def first_cuda_step(config, optimizer, samples, symmetric, logger):
    """Select and verify the batch while computing the first training update."""
    state, hamiltonian = optimizer.state, optimizer.hamiltonian
    device = samples.spins.device
    path, key = _cache_file(config, state, device, "train_symm" if symmetric else "train")
    record = _read_record(path, key)
    saved_batch = record.get("jacobian_batch")
    if type(saved_batch) is not int or not 1 <= saved_batch <= len(samples.spins):
        saved_batch = None

    gram = record.get("gram_backends", {})
    if isinstance(gram, dict):
        for width, backend in gram.items():
            if (len(width) <= len(str(state.nparams)) and width.isascii() and width.isdecimal()
                    and backend in ("full GEMM", "tiled GEMM", "cuBLAS SYRK")):
                optimizer._gram_backend_choices[device, torch.Size((len(samples.spins), int(width)))] = backend

    Ebar = _evaluate_local_energy(state, hamiltonian, samples, optimizer.get_Ebar,
                                 record.get("local_energy_batch"), logger)

    manual_batch = optimizer.jacobian_batch
    if manual_batch is not None:
        batch = manual_batch
    elif saved_batch is None:
        batch = _initial_batch(optimizer, samples, path, key, logger)
    else:
        batch = saved_batch
        logger.info(f"CUDA Jacobian batch loaded: {batch} | {path}")
    optimizer.jacobian_batch = batch
    cache_limit = optimizer._cache_limit
    update = optimizer._step_with_retry(samples, Ebar)

    # Keep the tuned Jacobian batch on disk after an OOM fallback.
    verified = (manual_batch is None and saved_batch is None and batch == optimizer._last_jacobian_batch
                and cache_limit == optimizer._cache_limit and bool(update.isfinite().all()))
    local_batch = getattr(hamiltonian, "_local_energy_batch", None)
    settings = {
        "key": key,
        "jacobian_batch": batch if verified else saved_batch,
        "trials": record.get("trials", []),
        "gram_backends": {str(shape[1]): name for (_, shape), name in optimizer._gram_backend_choices.items()},
        "local_energy_batch": None if local_batch is None else int(local_batch),
    }
    if settings != record:
        _write_record(path, settings, logger)
    return update


def first_cuda_measurement(config, state, hamiltonian, samples, logger):
    path, key = _cache_file(config, state, samples.spins.device, "measure")
    record = _read_record(path, key)
    evaluate = partial(hamiltonian.expectation, state, return_var=True)
    saved_batch = record.get("local_energy_batch")
    result = _evaluate_local_energy(state, hamiltonian, samples, evaluate, saved_batch, logger)
    batch = hamiltonian._local_energy_batch
    if batch is not None:
        settings = {"key": key, "local_energy_batch": int(batch)}
        if settings == record:
            logger.info(f"CUDA local-energy tuning loaded: {path}")
        else:
            _write_record(path, settings, logger)
    return result
