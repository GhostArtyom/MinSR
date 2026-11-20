"""Run the PyTorch training, symmetry continuation and measurement in separate processes."""

import argparse
import importlib
import multiprocessing
from pathlib import Path

from .config import Config, add_common_options


def _worker(stage, connection, config, checkpoint, steps):
    module = importlib.import_module(f"examples_torch.{stage}")
    with connection:
        if stage == "train":
            run = module.main(config=config, steps=steps)
            result = run / f"params_{steps}.pt"
        elif stage == "train_symm":
            run = module.main(checkpoint, config=config, steps=steps)
            result = run / f"params_symm_{steps}.pt"
        else:
            run = module.main(checkpoint, config=config, steps=steps)
            result = run / "summary.json"
        if not result.is_file():
            raise RuntimeError(f"{stage} did not produce {result}.")
        connection.send(result)


def _run_stage(stage, config, checkpoint, steps):
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(stage, sender, config, checkpoint, steps))
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


def main(config=None, *, steps=2000, symm_steps=1000, measure_steps=100, from_checkpoint=None):
    config = Config() if config is None else config
    if min(steps, symm_steps, measure_steps) < 1:
        raise ValueError("All stage lengths must be positive.")
    if from_checkpoint is None:
        checkpoint = _run_stage("train", config, None, steps)
        print(f"Training checkpoint: {checkpoint}", flush=True)
    else:
        checkpoint = Path(from_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Training weights do not exist: {checkpoint}")
    checkpoint = _run_stage("train_symm", config, checkpoint, symm_steps)
    print(f"Symmetry checkpoint: {checkpoint}", flush=True)
    summary = _run_stage("measure", config, checkpoint, measure_steps)
    print(f"Measurement summary: {summary}", flush=True)
    return summary.parent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_options(parser)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--symm-steps", type=int, default=1000)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument("--from-checkpoint", type=Path)
    args = vars(parser.parse_args())
    controls = {key: args.pop(key) for key in ("steps", "symm_steps", "measure_steps", "from_checkpoint")}
    main(Config(**args), **controls)
