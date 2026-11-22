import argparse

from .run_pipeline import Config, add_common_options, add_sampling_options


def main(config=None, *, steps=2000, thermal_steps=None, sweep_steps=None, param_file=None):
    from .workflows import train

    return train(Config() if config is None else config, steps=steps, param_file=param_file,
                 thermal_steps=thermal_steps, sweep_steps=sweep_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PyTorch neural quantum state with MinSR.")
    add_common_options(parser)
    add_sampling_options(parser)
    parser.add_argument("--steps", type=int, default=2000)
    args = vars(parser.parse_args())
    controls = {key: args.pop(key) for key in ("steps", "thermal_steps", "sweep_steps")}
    print(main(Config(**args), **controls))
