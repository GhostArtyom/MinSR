"""Prepare a PyTorch CUDA tuning record for one stage without updating weights."""

import argparse
from pathlib import Path
from time import perf_counter

from .run_pipeline import Config, add_common_options, add_sampling_options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "train_symm", "measure"))
    parser.add_argument("--param-file", type=Path, help="Use representative weights instead of CUDA initialization.")
    add_common_options(parser)
    add_sampling_options(parser)
    args = vars(parser.parse_args())
    stage, checkpoint, thermal, sweep = (
        args.pop(key) for key in ("stage", "param_file", "thermal_steps", "sweep_steps")
    )

    import torch
    import quantax_torch as qtx
    from quantax_torch.utils import close_run, create_run
    from quantax_torch.utils.logger import thermalization_progress
    from ._cuda_tuning import first_cuda_measurement, first_cuda_step
    from .workflows import _checkpoint, _header, _log_cuda_paths, _sampler, _system

    config = Config(**args)
    if torch.device(config.device).type != "cuda" or not torch.cuda.is_available() or torch.version.cuda is None:
        parser.error("CUDA tuning requires an NVIDIA CUDA device.")
    checkpoint = None if checkpoint is None else _checkpoint(checkpoint)
    symmetric = stage != "train"
    measurement = stage == "measure"
    thermal, sweep = config.sampling(thermal, sweep, measurement=measurement)
    start = perf_counter()
    log_file = f"tuning_{stage}.log"
    run, logger, handler = create_run(config.output_dir, config.system, "minsr.torch.tune", log_file)
    print(run / log_file, flush=True)
    try:
        hamiltonian, state = _system(config, symmetric, checkpoint)
        if not measurement and config.eigh_backend is not None:
            torch.backends.cuda.preferred_linalg_library(config.eigh_backend)
        reweight = 2.0 if measurement else 1.0
        _header(logger, config, state, 1, thermal, sweep, symmetric, reweight, checkpoint)
        sampler = _sampler(config, state, reweight, thermal, sweep, thermalization_progress(logger, start))
        samples = sampler.sweep()
        optimizer = None
        if measurement:
            first_cuda_measurement(config, state, hamiltonian, samples, logger, benchmark=True)
        else:
            optimizer = qtx.optimizer.MinSR(state, hamiltonian, jacobian_batch=config.jacobian_batch, logger=logger)
            update = first_cuda_step(config, optimizer, samples, symmetric, logger, benchmark=True)
            if not bool(update.isfinite().all()):
                raise FloatingPointError("Non-finite MinSR update during CUDA tuning.")
        _log_cuda_paths(logger, config, state, sampler, hamiltonian, optimizer)
    except Exception:
        logger.exception("CUDA tuning failed")
        raise
    finally:
        close_run(logger, handler)


if __name__ == "__main__":
    main()
