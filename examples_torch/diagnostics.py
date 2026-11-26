"""Record MinSR inputs and updates without reevaluating the optimizer."""

import csv
from collections import deque
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path

import torch

from quantax_torch.optimizer.solver import minsr_pinv_eig


WINDOW = 10
ENERGY_JUMP = 20.0
LOCAL_ENERGY_LIMIT = 1e4
UPDATE_NORM_LIMIT = 0.5
FIELDS = (
    "stage", "step", "status", "weighted_energy", "centered_variance", "unweighted_local_energy_mean",
    "max_abs_local_energy", "max_abs_local_energy_index", "min_nonzero_amplitude", "effective_samples",
    "parameter_norm", "update_norm", "rhs_norm", "events", "error",
)


def _cpu(tensor):
    return tensor.detach().to(device="cpu", copy=True)


def _norm(tensor):
    dtype = torch.complex128 if tensor.is_complex() else torch.float64
    return torch.linalg.vector_norm(tensor, dtype=dtype).item()


def _exception(error):
    return f"{type(error).__name__}: {error}"


class StepDiagnostics:
    def __init__(self, config, state, hamiltonian, run, stage, logger, file=None):
        self.config = asdict(config) | {"output_dir": str(config.output_dir)}
        self.state = state
        self.hamiltonian = hamiltonian
        self.stage = stage
        self.training = stage != "measure"
        self.logger = logger
        self.root = run / "diagnostics"
        self.rolling = deque()
        self.previous_energy = None
        self.default_solver = minsr_pinv_eig()
        self.parameter_names = [list(tree) for tree in state.named_parameter_dicts()]
        self.file = run / "diagnostics.csv" if file is None else Path(file)
        with self.file.open("a", newline="", encoding="utf-8") as stream:
            if stream.tell() == 0:
                csv.DictWriter(stream, FIELDS).writeheader()
        logger.info(f"Diagnostics: {self.file} | snapshots: {self.root}")
        logger.info(
            f"Diagnostic events: |energy jump| >= {ENERGY_JUMP:g}, "
            f"max |local energy| >= {LOCAL_ENERGY_LIMIT:g}, update norm >= {UPDATE_NORM_LIMIT:g}, "
            "non-finite values or step exceptions"
        )
        if not self.training:
            self.fixed_weights = self._weights()
            self.fixed_parameter_norm = self._parameter_norm(self.fixed_weights)

    def _weights(self):
        return [{key: _cpu(value) for key, value in net.state_dict().items()} for net in self.state.ansatz]

    def _parameter_norm(self, weights):
        parts = [tree[name].reshape(-1) for tree, names in zip(weights, self.parameter_names) for name in names]
        return _norm(torch.cat(parts))

    def begin(self, step):
        self.step = step
        self.samples = None
        self.energy = None
        self.scaled_step = None
        self.parameter_update = None
        self.maximum = None
        self.solver_attempts = []
        self.weights = self._weights() if self.training else self.fixed_weights

    def record_samples(self, samples):
        self.samples = samples

    def record_energy(self, local, weighted_mean, variance, unweighted_mean):
        self.energy = local, weighted_mean, variance, unweighted_mean

    def solve(self, gram, rhs):
        # Save the actual arguments even if the finite-value check or eigensolver raises.
        attempt = {}
        self.solver_attempts.append(attempt)
        try:
            return self.default_solver(gram, rhs)
        except Exception as error:
            attempt["error"] = _exception(error)
            raise
        finally:
            for name, value in (("gram", gram), ("rhs", rhs)):
                try:
                    attempt[name] = _cpu(value)
                except RuntimeError as error:
                    if "error" not in attempt:
                        raise
                    attempt[f"{name}_capture_error"] = _exception(error)

    def record_update(self, scaled_step):
        self.scaled_step = _cpu(scaled_step)
        self.maximum = _cpu(self.state._maximum)
        deltas = []
        offset = 1
        for weights, names in zip(self.weights, self.parameter_names):
            for name in names:
                value = weights[name]
                step = self.scaled_step[offset:offset + value.numel()]
                offset += value.numel()
                # Variational.update subtracts a delta rounded to the parameter dtype.
                deltas.append(-(step if value.is_complex() else step.real).to(value.dtype))
        self.parameter_update = torch.cat(deltas)

    def finish(self, *, error=None, update_applied=None, optimizer=None):
        row = dict(stage=self.stage, step=self.step, status="failed" if error else "completed")
        flags = []
        row["parameter_norm"] = (self._parameter_norm(self.weights) if self.training
                                 else self.fixed_parameter_norm)
        batch = {}
        if self.samples is not None:
            batch = {name: _cpu(getattr(self.samples, name))
                     for name in ("spins", "wave_function", "reweight_factor")}
            amplitudes = batch["wave_function"].abs()
            weights = batch["reweight_factor"]
            nonzero = amplitudes[amplitudes != 0]
            row["min_nonzero_amplitude"] = nonzero.min().item() if nonzero.numel() else None
            probability = weights.to(torch.float64) / len(weights)
            row["effective_samples"] = probability.square().sum().reciprocal().item()
            for name, values in (("wave_function", amplitudes), ("reweight_factor", weights)):
                if not bool(values.isfinite().all()):
                    flags.append(f"nonfinite_{name}")
        if self.energy is not None:
            local, energy, variance, unweighted = map(_cpu, self.energy)
            batch["local_energy"] = local
            maximum, index = local.abs().max(dim=0)
            row.update(
                weighted_energy=energy.item(), centered_variance=variance.item(),
                unweighted_local_energy_mean=unweighted.real.item(),
                max_abs_local_energy=maximum.item(), max_abs_local_energy_index=index.item(),
            )
            if not bool(local.isfinite().all()):
                flags.append("nonfinite_local_energy")
            if maximum.item() >= LOCAL_ENERGY_LIMIT:
                flags.append("local_energy_spike")
            if bool(energy.isfinite()):
                if self.previous_energy is not None and abs(energy.item() - self.previous_energy) >= ENERGY_JUMP:
                    flags.append("weighted_energy_jump")
                self.previous_energy = energy.item()
        if self.solver_attempts:
            attempt = self.solver_attempts[-1]
            if "rhs" in attempt:
                row["rhs_norm"] = _norm(attempt["rhs"])
            for attempt in self.solver_attempts:
                if "error" in attempt:
                    if "gram" in attempt and not bool(attempt["gram"].isfinite().all()):
                        flags.append("nonfinite_gram")
                    flags.append("solve_exception")
        if self.parameter_update is not None:
            row["update_norm"] = _norm(self.parameter_update)
            if not bool(self.scaled_step.isfinite().all()):
                flags.append("nonfinite_scaled_step")
            if row["update_norm"] >= UPDATE_NORM_LIMIT:
                flags.append("large_update")
        for key in ("weighted_energy", "centered_variance", "parameter_norm", "rhs_norm", "update_norm"):
            if key in row and not torch.isfinite(torch.as_tensor(row[key], dtype=torch.float64)).item():
                flags.append(f"nonfinite_{key}")
        if error is not None:
            flags.append("step_exception")
            row["error"] = _exception(error)
        row["events"] = "|".join(dict.fromkeys(flags))

        if self.training or flags:
            snapshot = {
                "version": 1, "stage": self.stage, "step": self.step, "config": self.config,
                "state_dicts_before_update": self.weights, "parameter_names": self.parameter_names,
                "batch_before_update": batch, "solver_attempts": self.solver_attempts,
                "scaled_step_argument": self.scaled_step, "maximum_before_update": self.maximum,
                "parameter_update": self.parameter_update, "update_applied": update_applied,
                "local_energy_batch": getattr(self.hamiltonian, "_local_energy_batch", None),
                "jacobian_batch": None if optimizer is None else optimizer._last_jacobian_batch,
                "metrics": row,
            }
            self._save(snapshot, bool(flags))
        with self.file.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, FIELDS).writerow(row)
        if flags:
            self.logger.warning(f"Diagnostic event at step {self.step}: {row['events']}")

    def _save(self, snapshot, event):
        name = f"step_{self.step:06d}.pt"
        if event:
            event_dir = self.root / "events" / f"step_{self.step:06d}"
            event_dir.mkdir(parents=True)
            for previous in self.rolling:
                (event_dir / previous.name).hardlink_to(previous)
        if self.training:
            directory = self.root / "rolling"
            directory.mkdir(parents=True, exist_ok=True)
        else:
            directory = event_dir
        path = directory / name
        temporary = path.with_suffix(".tmp")
        try:
            torch.save(snapshot, temporary)
            temporary.replace(path)
        finally:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
        if self.training:
            if event:
                (event_dir / name).hardlink_to(path)
            self.rolling.append(path)
            if len(self.rolling) > WINDOW:
                self.rolling.popleft().unlink()
