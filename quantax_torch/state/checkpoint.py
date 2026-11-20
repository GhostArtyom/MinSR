from pathlib import Path

import numpy as np
import torch


def load_checkpoint(file):
    file = Path(file)
    if file.suffix != ".pt":
        raise ValueError("Torch checkpoints use .pt. Convert Equinox checkpoints first.")
    data = torch.load(file, map_location="cpu", weights_only=True)
    if data.get("version") != 1:
        raise ValueError("Unsupported checkpoint version.")
    return data


def load_res_sum_eqx(file, net):
    """Read the NumPy array stream of a checkout-compatible ResSum checkpoint."""
    weights = {}
    with Path(file).open("rb") as stream:
        for name, target in net.state_dict().items():
            array = np.load(stream, allow_pickle=False)
            value = torch.from_numpy(array)
            if name.endswith(".bias"):
                value = value.flatten()
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(f"{file}: {name} has {tuple(value.shape)} and {value.dtype}, "
                                 f"expected {tuple(target.shape)} and {target.dtype}.")
            weights[name] = value
        if stream.read(1):
            raise ValueError(f"{file}: unexpected extra Equinox leaves.")
    net.load_state_dict(weights, strict=True)
