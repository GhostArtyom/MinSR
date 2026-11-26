import argparse
from pathlib import Path

from .run_pipeline import Config, PRESETS, add_common_options, add_sampling_options


def main(param_file=None, config=None, *, network_idx=None, steps=100,
         thermal_steps=None, sweep_steps=None, diagnostics_file=None):
    from .workflows import measure

    config = Config() if config is None else config
    if network_idx is not None or param_file is None:
        network_idx = 1 if network_idx is None else network_idx
        config, bundled_weights = config.network_for_preset(network_idx)
        if param_file is None:
            param_file = bundled_weights
    return measure(config, steps=steps, param_file=param_file, preset=network_idx,
                   thermal_steps=thermal_steps, sweep_steps=sweep_steps, diagnostics_file=diagnostics_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure energy and variance at fixed weights.")
    add_common_options(parser)
    add_sampling_options(parser)
    parser.add_argument("--param-file", type=Path)
    parser.add_argument("--network-idx", type=int, choices=PRESETS)
    parser.add_argument("--steps", type=int, default=100)
    args = vars(parser.parse_args())
    controls = {key: args.pop(key) for key in ("param_file", "network_idx", "steps", "thermal_steps", "sweep_steps")}
    print(main(config=Config(**args), **controls))
