"""Batched, per-sample reverse pass for unprojected 2D ResSum blocks."""

import math

import torch
from torch.nn import functional as F

from ..nn import Conv, Sequential
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


def _params_grad(conv, activate, delta):
    grads = []
    delta = delta.flatten(2)
    if conv.weight.requires_grad:
        # Each column is one periodic input patch. A batched GEMM reduces
        # across lattice sites, leaving a separate kernel gradient per sample.
        patches = F.unfold(_pad(activate, conv), conv.kernel_size)
        grads.append(torch.bmm(delta, patches.transpose(1, 2)).flatten(1))
    if conv.bias is not None and conv.bias.requires_grad:
        grads.append(delta.sum(dim=-1))
    return grads


def _input_grad(conv, delta):
    # The adjoint of a periodic cross-correlation is another periodic
    # cross-correlation, with reversed kernel and complementary padding.
    weight = conv.weight.transpose(0, 1).flip((-2, -1))
    return F.conv2d(_pad(delta, conv, transpose=True), weight)


def _reverse(block, x, delta, propagate):
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
        gradients.append(_params_grad(layer.conv1, y, d_hidden) + _params_grad(layer.conv2, hidden, delta))
        if propagate or i > 0:
            d_residual = delta if y.shape[1] == delta.shape[1] else delta.sum(dim=1, keepdim=True)
            d_input = _input_grad(layer.conv1, d_hidden) / scale
            if layer.nblock:
                d_input = d_input * (y > 0)
            delta = d_residual + d_input
    return torch.cat([g for layer_grads in reversed(gradients) for g in layer_grads], dim=1), delta


def batched_res_sum_vjp(block, x, delta, propagate, max_parallel):
    if (
        x.device.type != "cuda" or x.ndim != 5 or x.shape[1] != 1 or x.dtype != torch.float32
        or type(block) is not Sequential or not len(block) or not _unhooked(block)
        or not all(
            type(layer) is ResBlock and _unhooked(layer)
            and all(_supported_conv(conv, *x.shape[-2:]) for conv in (layer.conv1, layer.conv2))
            for layer in block
        )
    ):
        return None
    # Bound the largest temporary (unfolded patches) rather than using the
    # much smaller chunk needed by the generic per-sample autodiff pass.
    patch_size = x.shape[-2] * x.shape[-1] * max(
        conv.weight[0].numel() for layer in block for conv in (layer.conv1, layer.conv2)
    )
    batch = min(len(x) if max_parallel is None else max_parallel, max(1, 2**24 // patch_size))
    grads, upstream = [], []
    for features, cotangent in zip(x[:, 0].split(batch), delta[:, 0].split(batch), strict=True):
        grad, d_input = _reverse(block, features, cotangent, propagate)
        grads.append(grad)
        if propagate:
            upstream.append(d_input[:, None])
    return torch.cat(grads), torch.cat(upstream) if propagate else None
