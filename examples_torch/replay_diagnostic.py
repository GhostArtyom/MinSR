"""Recompute a MinSR diagnostic step on its saved configurations."""

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import torch

import quantax_torch as qtx
from quantax_torch.optimizer.solver import minsr_pinv_eig
from .diagnostics import _exception, _norm
from .run_pipeline import Config
from .workflows import _system


def _stat(tensor):
    finite = tensor.isfinite()
    values = tensor[finite]
    return {
        "finite": int(finite.sum().item()), "count": tensor.numel(),
        "max_abs_finite": values.abs().max().item() if values.numel() else None,
    }


def _compare(saved, replayed):
    replayed = replayed.to(device=saved.device)
    finite = saved.isfinite() & replayed.isfinite()
    delta = (saved[finite] - replayed[finite]).abs()
    return {
        "saved": _stat(saved), "replayed": _stat(replayed),
        "finite_masks_equal": torch.equal(saved.isfinite(), replayed.isfinite()),
        "exact_equal": torch.equal(saved, replayed),
        "max_abs_difference_finite": delta.max().item() if delta.numel() else None,
    }


def _json_number(value):
    if isinstance(value, dict):
        return {key: _json_number(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


@torch.no_grad()
def replay(snapshot, device=None):
    stage = snapshot["stage"]
    config = Config(**snapshot["config"])
    if device is not None:
        config = replace(config, device=device)
    symmetric = stage != "train"
    hamiltonian, state = _system(config, symmetric)
    for net, weights in zip(state.ansatz, snapshot["state_dicts_before_update"], strict=True):
        net.load_state_dict(weights)
    if (batch_size := snapshot["local_energy_batch"]) is not None and state.device.type == "cuda":
        hamiltonian._set_local_energy_batch(state, torch.empty(0, device=state.device).device, batch_size)
    if state.device.type == "cuda" and config.eigh_backend is not None:
        torch.backends.cuda.preferred_linalg_library(config.eigh_backend)
    batch = snapshot["batch_before_update"]
    report = {"stage": stage, "step": snapshot["step"], "device": str(state.device),
              "recorded_metrics": snapshot["metrics"]}
    if not batch:
        return report
    samples = qtx.sampler.Samples(*(batch[key].to(state.device)
                                   for key in ("spins", "wave_function", "reweight_factor")))
    amplitudes = state(samples.spins)
    report["wave_function"] = _compare(batch["wave_function"], amplitudes)

    def observe(local, *_):
        report["local_energy_recomputed"] = _stat(local)
        if "local_energy" in batch:
            report["local_energy"] = _compare(batch["local_energy"], local)
        maximum, index = local.abs().max(dim=0)
        report["replayed_max_local_energy"] = {
            "index": index.item(), "abs": maximum.item(), "local_energy": local[index].real.item(),
            "recorded_amplitude": samples.wave_function[index].abs().item(),
            "recomputed_amplitude": amplitudes[index].abs().item(),
            "reweight_factor": samples.reweight_factor[index].item(),
        }

    if stage == "measure":
        hamiltonian.expectation(state, samples, local_energy_observer=observe)
        return report

    saved_attempts = snapshot["solver_attempts"]
    default_solver = minsr_pinv_eig()

    def solve(gram, rhs):
        report["replayed_gram"] = _stat(gram)
        report["replayed_rhs"] = _stat(rhs)
        if saved_attempts:
            saved = saved_attempts[-1]
            if "gram" in saved:
                report["gram"] = _compare(saved["gram"], gram)
            if "rhs" in saved:
                report["rhs"] = _compare(saved["rhs"], rhs)
                report["rhs_norm"] = {"saved": _norm(saved["rhs"]), "replayed": _norm(rhs)}
        return default_solver(gram, rhs)

    optimizer = qtx.optimizer.MinSR(state, hamiltonian, solver=solve,
                                   jacobian_batch=snapshot["jacobian_batch"],
                                   local_energy_observer=observe)
    try:
        update = optimizer.get_step(samples) * config.step_size
    except (FloatingPointError, RuntimeError) as error:
        report["replay_error"] = _exception(error)
    else:
        saved = snapshot["scaled_step_argument"]
        if saved is not None:
            report["scaled_step"] = _compare(saved, update)
            report["scaled_parameter_step_norm"] = {"saved": _norm(saved[1:]), "replayed": _norm(update[1:])}
    if snapshot["update_applied"]:
        state._maximum.copy_(snapshot["maximum_before_update"])
        state.update(snapshot["scaled_step_argument"])
        report["wave_function_after_saved_update"] = _stat(state(samples.spins))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()
    snapshot = torch.load(args.snapshot, map_location="cpu", weights_only=True)
    print(json.dumps(_json_number(replay(snapshot, args.device)), indent=2, allow_nan=False))
