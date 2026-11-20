"""Batched per-sample reverse pass for 2D ResSum blocks."""

import math

import torch
from torch.nn import functional as F

from ..nn import Conv, Sequential
from ..nn._cuda_support import supports_tuned_kernels
from ..nn.modules import _circular_pad
from ..nn.sum_nets import ResBlock


def _unhooked(module):
    return not (module._forward_hooks or module._forward_pre_hooks)


def _supported_conv(conv, height, width):
    return (
        type(conv) is Conv and conv.weight.ndim == 4 and conv.weight.dtype == torch.float32
        and _unhooked(conv)
        and conv.padding == "CIRCULAR" and conv.groups == 1 and conv.stride in (1, (1, 1))
        and conv.dilation == (1, 1)
        and all(max(pair) <= size for pair, size in zip(conv._circular_pads, (height, width)))
    )


def _pad(x, conv, *, transpose=False):
    pairs = conv._circular_pads
    return _circular_pad(x, tuple(pair[::-1] for pair in pairs) if transpose else pairs)


def _params_grad(conv, activate, delta, weight_grad, orbit=1):
    grads = []
    if conv.weight.requires_grad:
        # Each column is one periodic input patch. A batched GEMM reduces
        # across lattice sites, leaving a separate kernel gradient per sample.
        if weight_grad is None:
            patches = F.unfold(_pad(activate, conv), conv.kernel_size)
            grads.append(torch.bmm(delta.flatten(2), patches.transpose(1, 2)).flatten(1))
        else:
            grads.append(weight_grad(conv, activate, delta, orbit))
    if conv.bias is not None and conv.bias.requires_grad:
        if orbit == 1:
            grads.append(delta.flatten(2).sum(dim=-1))
        else:
            grads.append(delta.unflatten(0, (-1, orbit)).sum(dim=(1, 3, 4)))
    return grads


def _input_grad(conv, delta):
    # The adjoint of a periodic cross-correlation is another periodic
    # cross-correlation, with reversed kernel and complementary padding.
    weight = conv.weight.transpose(0, 1).flip((-2, -1))
    return F.conv2d(_pad(delta, conv, transpose=True), weight)


def _reverse(block, x, delta, propagate, weight_grad, orbit):
    activations = []
    for layer in block:
        scale = math.sqrt(2 if layer.nblock == 0 else layer.nblock + 1)
        y = x / scale
        y = y.relu() if layer.nblock else y
        hidden = layer.conv1(y).relu()
        activations.append((y, hidden, scale))
        if len(activations) < len(block):
            x = x + layer.conv2(hidden)

    gradients = []
    for i in reversed(range(len(block))):
        layer = block[i]
        y, hidden, scale = activations[i]
        d_hidden = _input_grad(layer.conv2, delta) * (hidden > 0)
        gradients.append(_params_grad(layer.conv1, y, d_hidden, weight_grad, orbit)
                         + _params_grad(layer.conv2, hidden, delta, weight_grad, orbit))
        if propagate or i > 0:
            d_residual = delta if y.shape[1] == delta.shape[1] else delta.sum(dim=1, keepdim=True)
            d_input = _input_grad(layer.conv1, d_hidden) / scale
            if layer.nblock:
                d_input = d_input * (y > 0)
            delta = d_residual + d_input
    return torch.cat([g for layer_grads in reversed(gradients) for g in layer_grads], dim=1), delta


def batched_res_sum_vjp(block, x, delta, propagate, max_parallel):
    if (
        x.device.type != "cuda" or x.ndim != 5 or not 1 <= x.shape[1] <= 16 or x.dtype != torch.float32
        or type(block) is not Sequential or not len(block) or not _unhooked(block)
        or not all(
            type(layer) is ResBlock and _unhooked(layer)
            and all(_supported_conv(conv, *x.shape[-2:]) for conv in (layer.conv1, layer.conv2))
            for layer in block
        )
    ):
        return None
    orbit = x.shape[1]
    sites = x.shape[-2] * x.shape[-1]
    convolutions = [conv for layer in block for conv in (layer.conv1, layer.conv2)]
    weight_grad = None
    if (sites <= 100 and delta.dtype == x.dtype and not torch.is_autocast_enabled("cuda")
            and all(max(conv.kernel_size) <= 5 and conv.weight.shape[0] <= 64 for conv in convolutions)
            and supports_tuned_kernels(x.device)):
        from ._res_sum_cuda import periodic_weight_grad

        weight_grad = periodic_weight_grad
    elif orbit > 1:
        # A projected sample needs a sum across all orbit members, not one
        # parameter-gradient row per convolution input.
        return None
    if orbit == 1:
        # Retain the previous chunk size, including for the unfold fallback.
        batch = max(1, 2**24 // (sites * max(conv.weight[0].numel() for conv in convolutions)))
    else:
        # Keep the convolution activations bounded as the orbit grows.
        width = max(conv.weight.shape[0] for conv in convolutions)
        largest = sites * orbit * width
        batch = max(1, min(128, 2**24 // largest))
        if width >= 32 and sites * width >= 2048:
            # Large projected feature maps run faster in smaller reverse
            # chunks. This does not change the inference or forward cap.
            batch = min(batch, max(1, 512 // orbit))
    batch = min(batch, len(x) if max_parallel is None else max(1, max_parallel // orbit))
    grads, upstream = [], []
    for features, cotangent in zip(x.split(batch), delta.split(batch), strict=True):
        grad, d_input = _reverse(block, features.flatten(0, 1), cotangent.flatten(0, 1),
                                 propagate, weight_grad, orbit)
        grads.append(grad)
        if propagate:
            upstream.append(d_input.unflatten(0, features.shape[:2]))
    return torch.cat(grads), torch.cat(upstream) if propagate else None
