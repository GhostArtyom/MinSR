import argparse
from pathlib import Path

from .run_pipeline import Config, add_common_options, add_sampling_options, add_training_options


def main(param_file, config=None, *, steps=1000, thermal_steps=None, sweep_steps=None, diagnostics_file=None):
    from .workflows import train

    return train(Config() if config is None else config, steps=steps, param_file=param_file, symmetric=True,
                 thermal_steps=thermal_steps, sweep_steps=sweep_steps, diagnostics_file=diagnostics_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MinSR training with symmetry projection from an initial checkpoint.")
    add_common_options(parser)
    add_sampling_options(parser)
    add_training_options(parser)
    parser.add_argument("--param-file", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    args = vars(parser.parse_args())
    controls = {key: args.pop(key) for key in ("param_file", "steps", "thermal_steps", "sweep_steps")}
    print(main(config=Config(**args), **controls))
