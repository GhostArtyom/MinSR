"""Run the PyTorch training, symmetry-projected training and measurement in separate processes."""

import argparse
import importlib
import math
import multiprocessing
from dataclasses import dataclass, replace
from pathlib import Path

# Edit these settings, then run: python -m examples_torch.run_pipeline
ROOT = Path(__file__).resolve().parents[1]
FROM_CHECKPOINT = None  # Set to a training checkpoint to skip the first stage.
# Keep lattice and network settings shared. A stage can override nsamples or
# max_parallel, or set thermal_steps and sweep_steps instead of the defaults.
STAGES = {
    "train": {"steps": 2000},
    "train_symm": {"steps": 1000},
    "measure": {"steps": 100},
}


@dataclass(frozen=True)
class Config:
    # Common settings for all three stages.
    L: int = 10
    J1: float = 1.0
    J2: float = 0.5
    depth: int = 16
    channels: int = 16
    kernel_size: int = 3
    use_sinh: bool = True
    nsamples: int = 1000
    step_size: float = 1e-3
    max_parallel: int = 32768
    projected_forward_batch: int | None = None  # None tunes local energy and uses 1024 for sampling.
    projected_periodic_conv: bool = True  # IEEE FP32 periodic convolution on supported CUDA GPUs.
    projected_local_energy_graph: bool = True  # Replay full batches of connected configurations.
    projected_exchange_graph: bool = True  # Replay compacted CUDA exchange proposals.
    ordinary_periodic_conv: bool = True  # For supported CUDA shapes; does not change the backward pass.
    jacobian_batch: int | None = None
    seed: int = 2347
    device: str = "cuda"
    output_dir: Path = ROOT / "runs" / "torch"
    eigh_backend: str | None = None

    def __post_init__(self):
        if self.L < 2 or self.L % 2:
            raise ValueError("This square-lattice example needs an even L with L >= 2.")
        if self.depth < 2 or self.depth % 2:
            raise ValueError("depth must be positive and even.")
        if min(self.channels, self.kernel_size, self.nsamples, self.max_parallel) < 1:
            raise ValueError("Network and batch sizes must be positive.")
        if self.jacobian_batch is not None and self.jacobian_batch < 1:
            raise ValueError("jacobian_batch must be positive.")
        if self.projected_forward_batch is not None and self.projected_forward_batch < 1:
            raise ValueError("projected_forward_batch must be positive.")
        if self.eigh_backend not in (None, "default", "cusolver", "magma"):
            raise ValueError("eigh_backend must be default, cusolver, or magma.")
        if not all(math.isfinite(x) for x in (self.J1, self.J2, self.step_size)) or self.step_size <= 0:
            raise ValueError("The couplings must be finite and step_size must be finite and positive.")

    @property
    def system(self):
        return f"j1j2_square_{self.L}x{self.L}_J1-{self.J1:g}_J2-{self.J2:g}"

    @property
    def network(self):
        return {key: getattr(self, key) for key in ("depth", "channels", "kernel_size", "use_sinh")}

    def sampling(self, thermal_steps, sweep_steps, *, measurement=False):
        thermal_default = 100 if measurement else 20
        sweep_default = 10 if measurement else 2
        thermal_steps = thermal_default * self.L**2 if thermal_steps is None else thermal_steps
        sweep_steps = sweep_default * self.L**2 if sweep_steps is None else sweep_steps
        if min(thermal_steps, sweep_steps) < 0:
            raise ValueError("Thermalization and sweep steps must be nonnegative.")
        return thermal_steps, sweep_steps

    def network_for_preset(self, index):
        depth, channels, filename = PRESETS[index]
        return replace(self, depth=depth, channels=channels), ROOT / "params" / filename


PRESETS = {
    1: (16, 16, "square10x10_1616.eqx"),
    2: (16, 32, "square10x10_1632.eqx"),
    3: (30, 64, "square10x10_3064.eqx"),
}


def add_common_options(parser):
    defaults = Config()
    for name in ("L", "J1", "J2", "depth", "channels", "kernel_size", "nsamples",
                 "step_size", "max_parallel",
                 "seed", "device", "output_dir"):
        value = getattr(defaults, name)
        kind = Path if isinstance(value, Path) else type(value)
        flag = name.lower().replace("_", "-")
        parser.add_argument(f"--{flag}", dest=name, type=kind, default=value)
    parser.add_argument("--projected-forward-batch", type=int, default=defaults.projected_forward_batch,
                        help="Fix the projected CUDA forward batch in orbit members. By default local energy tunes it.")
    parser.add_argument("--jacobian-batch", type=int, default=defaults.jacobian_batch,
                        help="Samples per MinSR derivative batch. Defaults to an estimate from GPU memory.")
    parser.add_argument("--eigh-backend", choices=("default", "cusolver", "magma"), default=defaults.eigh_backend,
                        help="Select the CUDA linear algebra backend for the training process.")
    parser.add_argument("--projected-periodic-conv", action=argparse.BooleanOptionalAction,
                        default=defaults.projected_periodic_conv)
    parser.add_argument("--projected-local-energy-graph", action=argparse.BooleanOptionalAction,
                        default=defaults.projected_local_energy_graph)
    parser.add_argument("--projected-exchange-graph", action=argparse.BooleanOptionalAction,
                        default=defaults.projected_exchange_graph)
    parser.add_argument("--ordinary-periodic-conv", action=argparse.BooleanOptionalAction,
                        default=defaults.ordinary_periodic_conv)
    parser.add_argument("--use-sinh", action=argparse.BooleanOptionalAction, default=defaults.use_sinh)


def add_sampling_options(parser):
    parser.add_argument("--thermal-steps", type=int)
    parser.add_argument("--sweep-steps", type=int)


def _stage_options(stage, config, steps):
    overrides = STAGES[stage].copy()
    length = overrides.pop("steps")
    steps = length if steps is None else steps
    if steps < 1:
        raise ValueError(f"{stage}: steps must be positive.")
    thermal = overrides.pop("thermal_steps", None)
    sweep = overrides.pop("sweep_steps", None)
    config = replace(config, **overrides) if overrides else config
    thermal, sweep = config.sampling(thermal, sweep, measurement=stage == "measure")
    return config, {"steps": steps, "thermal_steps": thermal, "sweep_steps": sweep}


def _worker(stage, connection, checkpoint, config, options):
    module = importlib.import_module(f"examples_torch.{stage}")
    steps = options["steps"]
    with connection:
        if stage == "train":
            run = module.main(config=config, param_file=checkpoint, **options)
            result = run / f"params_{steps}.pt"
        elif stage == "train_symm":
            run = module.main(checkpoint, config=config, **options)
            result = run / f"params_symm_{steps}.pt"
        else:
            run = module.main(checkpoint, config=config, **options)
            result = run / "summary.json"
        if not result.is_file():
            raise RuntimeError(f"{stage} did not produce {result}.")
        connection.send(result)


def _run_stage(stage, checkpoint, config, options):
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(stage, sender, checkpoint, config, options))
    with receiver, sender:
        print(f"Starting {stage}...", flush=True)
        process.start()
        sender.close()
        try:
            try:
                result = receiver.recv()
            except EOFError:
                result = None
            process.join()
            if process.exitcode or result is None:
                raise RuntimeError(f"{stage} failed (exit code {process.exitcode}).")
            return result
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
            process.close()


def main(config=None, *, steps=None, symm_steps=None, measure_steps=None, from_checkpoint=FROM_CHECKPOINT,
         initial_weights=None):
    if initial_weights is not None and from_checkpoint is not None:
        raise ValueError("Choose either initial weights or a finished ordinary training checkpoint.")
    config = Config() if config is None else config
    stages = {
        stage: _stage_options(stage, config, length)
        for stage, length in (("train", steps), ("train_symm", symm_steps), ("measure", measure_steps))
    }
    if from_checkpoint is None:
        checkpoint = _run_stage("train", initial_weights, *stages["train"])
        print(f"Training checkpoint: {checkpoint}", flush=True)
    else:
        checkpoint = Path(from_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Training weights do not exist: {checkpoint}")
    checkpoint = _run_stage("train_symm", checkpoint, *stages["train_symm"])
    print(f"Symmetry checkpoint: {checkpoint}", flush=True)
    summary = _run_stage("measure", checkpoint, *stages["measure"])
    print(f"Measurement summary: {summary}", flush=True)
    return summary.parent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_options(parser)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--symm-steps", type=int)
    parser.add_argument("--measure-steps", type=int)
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--from-checkpoint", type=Path, default=FROM_CHECKPOINT,
                         help="Skip ordinary training and start with its final checkpoint.")
    weights.add_argument("--initial-weights", type=Path,
                         help="Load these weights before the first ordinary training step.")
    args = vars(parser.parse_args())
    controls = {key: args.pop(key) for key in (
        "steps", "symm_steps", "measure_steps", "from_checkpoint", "initial_weights",
    )}
    main(Config(**args), **controls)
