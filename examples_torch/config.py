import argparse
from dataclasses import dataclass, replace
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRESETS = {
    1: (16, 16, "square10x10_1616.eqx"),
    2: (16, 32, "square10x10_1632.eqx"),
    3: (30, 64, "square10x10_3064.eqx"),
}


@dataclass(frozen=True)
class Config:
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
    jacobian_batch: int = 8
    seed: int = 2347
    device: str = "cuda"
    output_dir: Path = ROOT / "runs" / "torch"

    def __post_init__(self):
        if self.L < 2 or self.L % 2:
            raise ValueError("This square-lattice example needs an even L with L >= 2.")
        if self.depth < 2 or self.depth % 2:
            raise ValueError("depth must be positive and even.")
        if min(self.channels, self.kernel_size, self.nsamples, self.max_parallel, self.jacobian_batch) < 1:
            raise ValueError("Network and batch sizes must be positive.")
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


def add_common_options(parser):
    defaults = Config()
    for name in ("L", "J1", "J2", "depth", "channels", "kernel_size", "nsamples",
                 "step_size", "max_parallel", "jacobian_batch", "seed", "device", "output_dir"):
        value = getattr(defaults, name)
        kind = Path if isinstance(value, Path) else type(value)
        flag = name.lower().replace("_", "-")
        parser.add_argument(f"--{flag}", dest=name, type=kind, default=value)
    parser.add_argument("--use-sinh", action=argparse.BooleanOptionalAction, default=defaults.use_sinh)


def add_sampling_options(parser):
    parser.add_argument("--thermal-steps", type=int)
    parser.add_argument("--sweep-steps", type=int)
