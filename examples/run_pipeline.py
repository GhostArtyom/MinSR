"""Run training, symmetry continuation, and measurement in order."""

import argparse
import importlib
import multiprocessing
from pathlib import Path


SHARED_SETTINGS = ("L", "J1", "J2", "KERNEL_SIZE", "USE_SINH")


def _stage_worker(stage, checkpoint, result):
    # Import JAX only in the child so each stage gets fresh process-wide state.
    from examples import train

    module = importlib.import_module(f"examples.{stage}")
    for name in SHARED_SETTINGS:
        setattr(module, name, getattr(train, name))

    with result:
        if stage == "measure":
            run_dir = module.main(checkpoint, network_config=(train.DEPTH, train.CHANNELS))
            result.send((run_dir, None))
            return

        if module.STEPS < 1:
            raise ValueError(f"{stage}.STEPS must be at least 1.")
        if stage == "train_symm":
            module.DEPTH, module.CHANNELS = train.DEPTH, train.CHANNELS
            run_dir = module.main(checkpoint)
        else:
            run_dir = module.main()

        # Look only in the directory returned by this stage, never in older runs.
        checkpoints = list(run_dir.glob(f"params_*_{module.STEPS}.eqx"))
        if len(checkpoints) != 1 or not checkpoints[0].is_file():
            raise RuntimeError(f"Expected one final checkpoint at step {module.STEPS} in {run_dir}.")
        result.send((run_dir, checkpoints[0]))


def _run_stage(stage, checkpoint):
    context = multiprocessing.get_context("spawn")
    received, send = context.Pipe(duplex=False)
    with received, send:
        process = context.Process(
            target=_stage_worker, args=(stage, checkpoint, send),
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


def main(from_checkpoint=None):
    if from_checkpoint is not None:
        from_checkpoint = Path(from_checkpoint).expanduser().resolve()
        if not from_checkpoint.is_file():
            raise FileNotFoundError(f"Training checkpoint does not exist: {from_checkpoint}")

    checkpoint = from_checkpoint
    if checkpoint is None:
        _, checkpoint = _run_stage("train", None)
        print(f"Training checkpoint: {checkpoint}", flush=True)

    _, checkpoint = _run_stage("train_symm", checkpoint)
    print(f"Symmetry checkpoint: {checkpoint}", flush=True)
    measured_dir, _ = _run_stage("measure", checkpoint)
    summary, = measured_dir.glob("summary_*.json")
    print(f"Measurement summary: {summary}", flush=True)
    return measured_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-checkpoint", type=Path,
        help="Skip train.py and start symmetry continuation from a compatible checkpoint.",
    )
    args = parser.parse_args()
    try:
        main(args.from_checkpoint)
    except (FileNotFoundError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Pipeline interrupted.\n")
