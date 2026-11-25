"""Run the CUDA batch matrix and validate finalists on the sampling loop."""

import gc
import json
import math
from dataclasses import asdict, replace
from itertools import product
from statistics import median
from time import perf_counter

import torch

from ._benchmark_cuda_memory import finish_cuda_step, start_cuda_step

import quantax_torch as qtx
from examples_torch import workflows
from examples_torch._cuda_tuning import _cache_file, _write_record
from examples_torch.run_pipeline import PRESETS, ROOT
from quantax_torch.operator.common_operators import _exchange_batch_candidates
from quantax_torch.optimizer._cuda_gram import time_gram_backends
from quantax_torch.utils import close_run, create_run
from quantax_torch.utils.logger import thermalization_progress


class _BatchFallback(RuntimeError):
    pass


def _timed(device, function, *args):
    torch.cuda.synchronize(device)
    start = perf_counter()
    result = function(*args)
    torch.cuda.synchronize(device)
    return perf_counter() - start, result


def _release_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def _median_times(rows):
    return {key: median(row[key] for row in rows) for key in rows[0]}


def _batches(config, state, stage):
    jacobian = {
        "train": (64, 250, 500, 1000), "train_symm": (32, 64, 125, 150), "measure": (None,),
    }[stage]
    if stage != "measure":
        jacobian = sorted({min(value, config.nsamples) for value in jacobian})
    base = state.forward_batch // state.nsymm
    if stage == "train":
        base = min(2048, base)
    local = [batch * state.nsymm for batch in _exchange_batch_candidates(state, base)]
    return list(product(jacobian, local))


def _setup(config, stage, checkpoint, thermal, sweep, logger):
    _, state = workflows._system(config, stage != "train", checkpoint)
    start = perf_counter()
    sampler = workflows._sampler(config, state, 2.0 if stage == "measure" else 1.0, thermal, sweep,
                                 thermalization_progress(logger, start))
    return state, sampler


def _check_batch(hamiltonian, state, local):
    if hamiltonian._local_energy_batch * state.nsymm != local:
        raise _BatchFallback("Local-energy batch differed from the requested value.")


def _trial(config, stage, batch, state, sampler, samples, warmup, steps, gram_choices, logger):
    jacobian, local = batch
    device = state.device
    fixed = replace(config, local_energy_forward_batch=local)
    hamiltonian = workflows._hamiltonian(fixed, device, stage != "train", state.inference_fn)
    optimizer = None if jacobian is None else qtx.optimizer.MinSR(
        state, hamiltonian, jacobian_batch=jacobian, logger=logger,
    )
    if optimizer is not None:
        optimizer._gram_backend_choices.update(gram_choices)
    full_step = samples is None
    torch.cuda.synchronize(device)
    rows = []
    memory_rows = []
    memory = None
    result = {"jacobian_batch": jacobian, "local_energy_forward_batch": local,
              "timings": rows, "cuda_memory": memory_rows}
    status = "ok"
    try:
        for i in range(warmup + steps):
            memory = start_cuda_step(device)
            times = {}
            if full_step:
                times["sampling"], samples = _timed(device, sampler.sweep)
            if optimizer is None:
                times["local_energy"], moments = _timed(device, hamiltonian.expectation, state, samples, True)
                if not all(map(math.isfinite, moments)):
                    raise FloatingPointError("Non-finite measurement.")
            else:
                times["local_energy"], Ebar = _timed(device, optimizer.get_Ebar, samples)
                retries = optimizer._jacobian_batch_limit, optimizer._cache_limit
                times["minsr"], update = _timed(device, optimizer._step_with_retry, samples, Ebar)
                if (optimizer._last_jacobian_batch != jacobian
                        or retries != (optimizer._jacobian_batch_limit, optimizer._cache_limit)):
                    raise _BatchFallback("MinSR retried a batch or changed its derivative cache limit.")
                if not bool(update.isfinite().all()):
                    raise FloatingPointError("Non-finite MinSR update.")
                if full_step:
                    times["update"], _ = _timed(device, state.update, update * config.step_size)
            _check_batch(hamiltonian, state, local)
            times["total"] = sum(times.values())
            rows.append(times)
            if optimizer is not None:
                memory["derivative_cache"] = optimizer._last_cache_plan
            memory_rows.append(finish_cuda_step(device, memory))
            memory = None
            detail = " | ".join(f"{name} {seconds:.3f} s" for name, seconds in times.items() if name != "total")
            logger.info(f"{stage} {batch} {'validation' if full_step else 'fixed'} "
                        f"{i + 1}/{warmup + steps}: {times['total']:.3f} s | {detail}")
    except torch.cuda.OutOfMemoryError as error:
        status, result["error"] = "out_of_memory", str(error)
    except _BatchFallback as error:
        status, result["error"] = "fallback", str(error)
    except FloatingPointError as error:
        status, result["error"] = "non_finite", str(error)
    if memory is not None:
        if optimizer is not None:
            memory["derivative_cache"] = optimizer._last_cache_plan
        memory_rows.append(finish_cuda_step(device, memory))
    result.update(status=status, warmup=warmup,
                  peak_bytes=max((row["allocated_peak_bytes"] for row in memory_rows), default=0),
                  peak_reserved_bytes=max((row["reserved_peak_bytes"] for row in memory_rows), default=0))
    if status == "ok":
        result["medians"] = _median_times(rows[warmup:])
        if optimizer is not None:
            gram_choices.update(optimizer._gram_backend_choices)
            result["gram_backends"] = {str(shape[1]): backend
                                       for (_, shape), backend in optimizer._gram_backend_choices.items()}
    elapsed = f" | median {result['medians']['total']:.3f} s" if status == "ok" else ""
    logger.info(f"{stage} {batch} {status}{elapsed} | peak {result['peak_bytes'] / 2**20:.0f} MiB")
    return result


def _rank(results):
    return sorted((row for row in results if row["status"] == "ok"), key=lambda row: row["medians"]["total"])


def _batch(row):
    return row["jacobian_batch"], row["local_energy_forward_batch"]


def _training_comparisons(config, stage, batch, state, samples, logger):
    jacobian, local = batch
    device = state.device
    fixed = replace(config, local_energy_forward_batch=local)
    hamiltonian = workflows._hamiltonian(fixed, device, stage != "train", state.inference_fn)
    optimizer = qtx.optimizer.MinSR(state, hamiltonian, jacobian_batch=jacobian, logger=logger)
    Ebar = optimizer.get_Ebar(samples)
    shape = (len(samples.spins),) * 2
    gram = {"status": "not_applicable"}
    if optimizer.vs_type == 0 and state.dtype == torch.float64 and shape[0] >= 512:
        blocks = optimizer._native_jacobian_blocks(samples.spins)
        try:
            _, raw = next(blocks)
        finally:
            blocks.close()
        if raw.shape[1] >= 4096:
            block = optimizer._normalized_block(raw.to(state.dtype), samples.reweight_factor)
            tile = 128 if shape[0] <= 1024 else 256 if shape[0] <= 4096 else 512
            output = torch.empty(shape, device=device, dtype=state.dtype)
            gram = {"status": "ok", "timings_ms": time_gram_backends(output, block, tile, warmup=2, steps=5)}
            gram["median_ms"] = {name: median(times) for name, times in gram["timings_ms"].items()}
            del output, block
        del raw
    matrix = optimizer.get_Tmat(samples, _native_cache=True)
    eigh = {}
    previous = torch.backends.cuda.preferred_linalg_library()
    try:
        for backend in ("default", "cusolver", "magma"):
            logger.info(f"{stage} eigensolver: {backend}")
            torch.cuda.reset_peak_memory_stats(device)
            try:
                torch.backends.cuda.preferred_linalg_library(backend)
                times = []
                for i in range(7):
                    seconds, vector = _timed(device, optimizer._solve, matrix, Ebar)
                    if not bool(vector.isfinite().all()):
                        raise FloatingPointError("Non-finite solution.")
                    if i >= 2:
                        times.append(seconds)
                eigh[backend] = {"status": "ok", "timings_seconds": times, "median_seconds": median(times)}
            except (RuntimeError, NotImplementedError, FloatingPointError) as error:
                eigh[backend] = {"status": "out_of_memory" if isinstance(error, torch.cuda.OutOfMemoryError)
                                 else "failed", "error": str(error)}
            eigh[backend]["peak_bytes"] = torch.cuda.max_memory_allocated(device)
    finally:
        torch.backends.cuda.preferred_linalg_library(previous)
    return {"gram": gram, "eigh": eigh}


def _sampling_comparisons(config, stage, checkpoint, logger):
    results = []
    for batch in (512, 1024, 2048):
        if batch > config.max_parallel:
            continue
        logger.info(f"{stage} sampling: {batch} orbit members/batch")
        fixed = replace(config, projected_forward_batch=batch)
        times = []

        def progress(step, total):
            torch.cuda.synchronize(fixed.device)
            times.append(perf_counter())

        try:
            _, state = workflows._system(fixed, True, checkpoint)
            torch.cuda.reset_peak_memory_stats(state.device)
            workflows._sampler(fixed, state, 2.0 if stage == "measure" else 1.0, 600, 0, progress)
            intervals = [end - start for start, end in zip(times[1:-1], times[2:], strict=True)]
            row = {"status": "ok", "timings_seconds": intervals, "median_seconds": median(intervals)}
        except torch.cuda.OutOfMemoryError as error:
            row = {"status": "out_of_memory", "error": str(error)}
        results.append(dict(row, projected_forward_batch=batch,
                            peak_bytes=torch.cuda.max_memory_allocated(fixed.device)))
        _release_cuda()
    return results


def _save_winner(config, stage, state, winner, logger):
    jacobian, local = _batch(winner)
    hamiltonian = workflows._hamiltonian(config, state.device, stage != "train", state.inference_fn)
    device = next(state.parameters()).device
    if not hamiltonian._set_local_energy_batch(state, device, local // state.nsymm):
        logger.warning(f"{stage} winner is available as explicit batch flags for this configuration.")
        return
    path, key = _cache_file(config, state, device, stage)
    record = {"key": key, "local_energy_batch": local // state.nsymm}
    if jacobian is not None:
        record.update(jacobian_batch=jacobian, gram_backends=winner["gram_backends"], trials=[])
    _write_record(path, record, logger)


@torch.no_grad()
def _run_stage(config, stage, checkpoint, thermal, sweep, warmup, steps, save_tuning, report, save, logger):
    logger.info(f"Starting {stage} matrix")
    thermal, sweep = config.sampling(thermal, sweep, measurement=stage == "measure")
    state, sampler = _setup(config, stage, checkpoint, thermal, sweep, logger)
    _, samples = _timed(state.device, sampler.sweep)
    maximum = state._maximum.clone()
    fixed = report["fixed"]
    gram_choices = {}
    for batch in _batches(config, state, stage):
        state._maximum.copy_(maximum)
        fixed.append(_trial(config, stage, batch, state, sampler, samples, warmup, steps, gram_choices, logger))
        save()
        _release_cuda()
    # Use the original fixed inputs for the secondary backend comparisons.
    if stage != "measure" and _rank(fixed):
        logger.info(f"{stage} fixed sample matrix complete")
    validation = report["validation"]
    del sampler
    _release_cuda()
    for finalist in _rank(fixed)[:2]:
        final_state, sampler = _setup(config, stage, checkpoint, thermal, sweep, logger)
        validation.append(_trial(config, stage, _batch(finalist), final_state, sampler, None, 2, 10,
                                 gram_choices, logger))
        save()
        del final_state, sampler
        _release_cuda()
    ranked = _rank(validation)
    if not ranked:
        return
    fastest = ranked[0]["medians"]["total"]
    winner = min((row for row in ranked if row["medians"]["total"] <= 1.05 * fastest),
                 key=lambda row: (row["peak_bytes"], row["medians"]["total"]))
    report["winner"] = winner
    logger.info(f"{stage} winner: {_batch(winner)}")
    if save_tuning:
        _save_winner(config, stage, state, winner, logger)
    if stage != "measure":
        state._maximum.copy_(maximum)
        try:
            report.update(_training_comparisons(config, stage, _batch(winner), state, samples, logger))
        except torch.cuda.OutOfMemoryError as error:
            report["backend_comparison"] = {"status": "out_of_memory", "error": str(error)}
    del samples, state
    _release_cuda()
    if stage != "train":
        report["sampling"] = _sampling_comparisons(config, stage, checkpoint, logger)


def _write_summary(run, summary):
    temporary = run / ".summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(run / "summary.json")


def run_matrix(config, param_file, thermal, sweep, warmup, steps, save_tuning):
    if param_file is None and config.L == 10 and config.kernel_size == 3 and config.use_sinh:
        param_file = next((ROOT / "params" / filename for depth, channels, filename in PRESETS.values()
                           if (depth, channels) == (config.depth, config.channels)), None)
    checkpoint = workflows._checkpoint(param_file) if param_file is not None else None
    run, logger, handler = create_run(config.output_dir, config.system, "minsr.torch.benchmark", "benchmark.log")
    print(run / "benchmark.log", flush=True)
    summary = {"config": asdict(config), "gpu": torch.cuda.get_device_name(config.device),
               "torch": torch.__version__, "checkpoint": checkpoint, "stages": {}, "status": "running"}

    def save():
        _write_summary(run, summary)

    previous = torch.backends.cuda.preferred_linalg_library()
    try:
        torch.backends.cuda.preferred_linalg_library("default")
        for stage in ("train", "train_symm", "measure"):
            report = {"fixed": [], "validation": [], "winner": None}
            summary["stages"][stage] = report
            save()
            _run_stage(config, stage, None if stage == "train" else checkpoint,
                       thermal, sweep, warmup, steps, save_tuning, report, save, logger)
            save()
            _release_cuda()
        incomplete = any(stage["winner"] is None for stage in summary["stages"].values())
        summary["status"] = "no_winner" if incomplete else "complete"
        save()
    except Exception:
        summary["status"] = "failed"
        save()
        logger.exception("CUDA batch matrix stopped")
        raise
    finally:
        torch.backends.cuda.preferred_linalg_library(previous)
        close_run(logger, handler)
    print(run / "summary.json", flush=True)
    return int(incomplete)
