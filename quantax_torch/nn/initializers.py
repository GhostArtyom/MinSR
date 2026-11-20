import math

import torch
from torch.nn import functional as F

from ..global_defs import get_generator


def variance_scaling(scale, mode, distribution):
    def initialize(tensor, generator=None):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
        divisor = {"fan_in": fan_in, "fan_out": fan_out, "fan_avg": (fan_in + fan_out) / 2}[mode]
        generator = get_generator(tensor.device) if generator is None else generator
        variance = scale / divisor
        if distribution == "uniform" and not tensor.is_complex():
            limit = math.sqrt(3 * variance)
            return torch.nn.init.uniform_(tensor, -limit, limit, generator=generator)
        if distribution == "truncated_normal" and not tensor.is_complex():
            std = math.sqrt(variance) / 0.8796256610342398
            return torch.nn.init.trunc_normal_(tensor, std=std, a=-2 * std, b=2 * std, generator=generator)
        if distribution not in ("uniform", "normal", "truncated_normal"):
            raise ValueError(f"Unsupported distribution {distribution!r}.")
        return torch.nn.init.normal_(tensor, std=math.sqrt(variance), generator=generator)

    return initialize


lecun_normal = variance_scaling(1, "fan_in", "truncated_normal")
lecun_uniform = variance_scaling(1, "fan_in", "uniform")
glorot_normal = variance_scaling(1, "fan_avg", "truncated_normal")
glorot_uniform = variance_scaling(1, "fan_avg", "uniform")
he_normal = variance_scaling(2, "fan_in", "truncated_normal")
he_uniform = variance_scaling(2, "fan_in", "uniform")


def value_pad(value):
    def initialize(tensor, generator=None):
        value_tensor = torch.as_tensor(value, device=tensor.device, dtype=tensor.dtype)
        if tensor.ndim != value_tensor.ndim:
            raise ValueError("Only values with the same rank can be padded.")
        pads = []
        for target, source in zip(reversed(tensor.shape), reversed(value_tensor.shape)):
            pads.extend(((target - 1) // 2 - (source - 1) // 2, target // 2 - source // 2))
        with torch.no_grad():
            tensor.copy_(F.pad(value_tensor, pads))
        return tensor

    return initialize
