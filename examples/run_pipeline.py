"""Run training, symmetry continuation, and measurement in order."""

import argparse
import importlib
import multiprocessing
import sys
from pathlib import Path


# Edit this block, then run: python -m examples.run_pipeline
FROM_CHECKPOINT = None  # Set to a training checkpoint to skip the first stage.
COMMON = {
    "L": 10,
    "J1": 1.0,
    "J2": 0.5,
    "DEPTH": 16,
    "CHANNELS": 16,
    "KERNEL_SIZE": 3,
    "USE_SINH": True,
    "NSAMPLES": 1000,  # 10000
    "STEP_SIZE": 1e-3,  # Training and symmetry continuation only.
    "MAX_PARALLEL": 32768,
    "OUTPUT_DIR": Path(__file__).resolve().parents[1] / "runs",
}
# Keep lattice and network settings shared. Override NSAMPLES or MAX_PARALLEL here if needed.
STAGES = {
    "train": {
        "STEPS": 2000,  # 20000
        "JACOBIAN_BATCH": None,
    },
    "train_symm": {
        "STEPS": 1000,  # 10000
        "JACOBIAN_BATCH": None,  # 250 limits derivative memory for 10000 samples.
    },
    "measure": {"STEPS": 100},
}


def _measurement_sampling(L):
    return {"THERMAL_STEPS": 100 * L**2, "SWEEP_STEPS": 10 * L**2}


def get_stage_config(stage, **overrides):
    """Combine the common defaults, stage defaults, and explicit overrides."""
    config = {**COMMON, **STAGES[stage]}
    if stage == "measure":
        config.pop("STEP_SIZE")
        config = {**_measurement_sampling(overrides.get("L", config["L"])), **config}
    return {**config, **overrides}


def _parse_bool(value):
    value = value.lower()
    if value not in ("true", "false"):
        raise argparse.ArgumentTypeError("expected true or false")
    return value == "true"


def add_config_arguments(parser, stage):
    """Expose existing configuration keys as optional standalone arguments."""
    types = {"J1": float, "J2": float, "STEP_SIZE": float, "USE_SINH": _parse_bool, "JACOBIAN_BATCH": int}
    for key, value in get_stage_config(stage).items():
        name = key.lower().replace("_", "-")
        help_text = f"Default: {value}"
        if stage == "measure" and key in ("DEPTH", "CHANNELS"):
            help_text = f"Custom network (requires --param-file); configured value: {value}"
        parser.add_argument(
            f"--{name}", f"--{key}", dest=key, metavar=key,
            type=types.get(key, type(value)),
            default=argparse.SUPPRESS, help=help_text,
        )


def _stage_worker(stage, checkpoint, result, config):
    # Import JAX only in the child so each stage gets fresh process-wide state.
    module = importlib.import_module(f"examples.{stage}")

    with result:
        if stage == "measure":
            run_dir = module.main(checkpoint, **config)
            result.send((run_dir, None))
            return

        if stage == "train_symm":
            run_dir = module.main(checkpoint, **config)
        else:
            run_dir = module.main(**config)

        suffix = "_symm" if stage == "train_symm" else ""
        final_checkpoint = run_dir / f"params{suffix}_{config['STEPS']}.eqx"
        if not final_checkpoint.is_file():
            raise RuntimeError(f"Final checkpoint not found: {final_checkpoint}")
        result.send((run_dir, final_checkpoint))


def _run_stage(stage, checkpoint):
    context = multiprocessing.get_context("spawn")
    received, send = context.Pipe(duplex=False)
    with received, send:
        process = context.Process(
            target=_stage_worker, args=(stage, checkpoint, send, get_stage_config(stage)),
        )
        print(f"Starting {stage}...", flush=True)
        process.start()
        send.close()
        try:
            try:
                output = received.recv()
            except EOFError:
                output = None
            process.join()
            if process.exitcode != 0 or output is None:
                raise RuntimeError(f"{stage} failed (exit code {process.exitcode}); no later stage was started.")
            return output
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
            process.close()


def main():
    if FROM_CHECKPOINT is None:
        _, checkpoint = _run_stage("train", None)
        print(f"Training checkpoint: {checkpoint}", flush=True)
    else:
        checkpoint = Path(FROM_CHECKPOINT).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Training checkpoint does not exist: {checkpoint}")

    _, checkpoint = _run_stage("train_symm", checkpoint)
    print(f"Symmetry checkpoint: {checkpoint}", flush=True)
    measured_dir, _ = _run_stage("measure", checkpoint)
    summary = measured_dir / "summary.json"
    if not summary.is_file():
        raise RuntimeError(f"Measurement summary not found: {summary}")
    print(f"Measurement summary: {summary}", flush=True)
    return measured_dir


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit("Edit the settings in run_pipeline.py and run it without arguments.")
    main()
