"""Optional CUDA kernels for ResSum inference; convolution uses IEEE FP32."""

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .activation import Exp, Scale, SinhShift
from .modules import Conv, Sequential
from .nqs_layers import ConvSymmetrize, ReshapeConv
from .sum_nets import ResBlock


@triton.jit
def _conv_tap(X, W, acc, site, out, lattice, row, col, npositions, k, valid_out,
              H: tl.constexpr, WIDE: tl.constexpr, C: tl.constexpr,
              KW: tl.constexpr, TOP: tl.constexpr, LEFT: tl.constexpr, tiles: tl.constexpr):
    tap = k // tiles
    channel = k % tiles * 16 + tl.arange(0, 16)
    h = (row + tap // KW + (H - TOP % H)) % H
    w = (col + tap % KW + (WIDE - LEFT % WIDE)) % WIDE
    pos = ((lattice * H + h) * WIDE + w) * C
    valid_channel = channel < C if C % 16 else tl.full((16,), True, tl.int1)
    values = tl.load(X + pos[:, None] + channel[None, :], (site[:, None] < npositions) & valid_channel[None, :], 0)
    weights = tl.load(W + (tap * C + channel[:, None]) * C + out[None, :],
                      valid_channel[:, None] & valid_out[None, :], 0)
    return tl.dot(values, weights, acc, input_precision="ieee")


@triton.jit
def _active_inputs(COUNT, start, ORBIT: tl.constexpr):
    return tl.maximum(tl.load(COUNT) - start, 0) * ORBIT


@triton.jit(do_not_specialize=["start", "npositions"])
def _periodic_conv(X, W, B, Y, RES, Y_NEXT, next_scale, COUNT, start, npositions,
                   ORBIT: tl.constexpr, H: tl.constexpr, WIDE: tl.constexpr,
                   C: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
                   TOP: tl.constexpr, LEFT: tl.constexpr, BN: tl.constexpr,
                   HAS_BIAS: tl.constexpr, RELU: tl.constexpr, RES_CHANNELS: tl.constexpr,
                   PREPARE_NEXT: tl.constexpr):
    # One implicit GEMM row per output site. Wrap during the load instead of
    # materializing a padded tensor. Walk one 16-channel tile of one tap at a time.
    if ORBIT:
        npositions = tl.minimum(npositions, _active_inputs(COUNT, start, ORBIT) * H * WIDE)
    active = tl.program_id(0) * 64 < npositions if ORBIT else True
    if active:
        site = tl.program_id(0) * 64 + tl.arange(0, 64)
        out = tl.program_id(1) * BN + tl.arange(0, BN)
        lattice = site // (H * WIDE)
        row = site // WIDE % H
        col = site % WIDE
        tiles: tl.constexpr = (C + 15) // 16
        valid_out = out < C if C % BN else tl.full((BN,), True, tl.int1)
        acc = tl.full((64, BN), 0, tl.float32)
        # Specialize the short masked tiles of a 2×2 kernel. Regular channel
        # tiles and larger kernels benchmarked better with the runtime loop.
        if KH == 2 and KW == 2 and C > 16 and C < 32:
            for k in tl.static_range(KH * KW * tiles):
                acc = _conv_tap(X, W, acc, site, out, lattice, row, col, npositions, k,
                                valid_out, H, WIDE, C, KW, TOP, LEFT, tiles)
        else:
            for k in range(KH * KW * tiles):
                acc = _conv_tap(X, W, acc, site, out, lattice, row, col, npositions, k,
                                valid_out, H, WIDE, C, KW, TOP, LEFT, tiles)
        if HAS_BIAS:
            acc += tl.load(B + out, valid_out, 0)[None, :]
        offset = site[:, None] * C + out[None, :]
        mask = (site[:, None] < npositions) & valid_out[None, :]
        if RES_CHANNELS == 1:
            acc = tl.load(RES + site, site < npositions, 0)[:, None] + acc
        elif RES_CHANNELS:
            acc = tl.load(RES + offset, mask, 0) + acc
        if RELU:
            acc = tl.where(acc < 0.0, 0.0, acc)
        tl.store(Y + offset, acc, mask)
        if PREPARE_NEXT:
            value = acc * next_scale
            tl.store(Y_NEXT + offset, tl.where(value < 0.0, 0.0, value), mask)


@triton.jit(do_not_specialize=["start", "npositions"])
def _periodic_stem(X, W, B, Y, COUNT, start, npositions, ORBIT: tl.constexpr, H: tl.constexpr, WIDE: tl.constexpr,
                   C: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
                   TOP: tl.constexpr, LEFT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                   HAS_BIAS: tl.constexpr):
    # One input channel: walk the taps directly instead of padding the
    # input or issuing a dot product with a nearly empty channel tile.
    if ORBIT:
        npositions = tl.minimum(npositions, _active_inputs(COUNT, start, ORBIT) * H * WIDE)
    active = tl.program_id(0) * BM < npositions if ORBIT else True
    if active:
        site = tl.program_id(0) * BM + tl.arange(0, BM)
        out = tl.program_id(1) * BN + tl.arange(0, BN)
        row = site // WIDE % H
        col = site % WIDE
        base = site // (H * WIDE) * (H * WIDE)
        acc = tl.full((BM, BN), 0, tl.float32)
        for tap in tl.static_range(KH * KW):
            h = (row + tap // KW + (H - TOP % H)) % H
            w = (col + tap % KW + (WIDE - LEFT % WIDE)) % WIDE
            value = tl.load(X + base + h * WIDE + w, site < npositions, 0)
            weight = tl.load(W + out * (KH * KW) + tap, out < C, 0)
            acc = tl.fma(value[:, None], weight[None, :], acc)
        if HAS_BIAS:
            acc += tl.load(B + out, out < C, 0)[None, :]
        acc = tl.where(acc < 0.0, 0.0, acc)
        tl.store(Y + site[:, None] * C + out[None, :], acc,
                 (site[:, None] < npositions) & (out[None, :] < C))


def _stem(conv, x, active):
    nbatch, _, height, width = x.shape
    channels = conv.weight.shape[0]
    kh, kw = conv.kernel_size
    (top, _), (left, _) = conv._circular_pads
    result = torch.empty((nbatch, height, width, channels), device=x.device, dtype=x.dtype)
    if channels <= 16 and nbatch >= 512:
        bm, bn, warps = 128, 16, 4
    else:
        bm, bn, warps = 64, min(32, triton.next_power_of_2(channels)), 2
    npositions = nbatch * height * width
    _periodic_stem[(triton.cdiv(npositions, bm), triton.cdiv(channels, bn))](
        x.contiguous(), conv.weight.contiguous(), conv.bias, result, *active[:2], npositions, active[2],
        height, width, channels, kh, kw, top, left, bm, bn, conv.bias is not None, num_warps=warps,
    )
    return result.permute(0, 3, 1, 2)


def _convolve(conv, x, *, relu=False, residual=None, next_denominator=None, active=(None, 0, 0)):
    compatible = not (
        type(conv) is not Conv or x.ndim != 4 or not len(x) or conv.weight.ndim != 4
        or x.dtype != torch.float32 or conv.weight.dtype != torch.float32
        or conv.padding != "CIRCULAR" or conv.stride not in (1, (1, 1)) or conv.dilation != (1, 1)
        or conv.groups != 1 or conv.weight.shape[1] != x.shape[1]
        or (conv.bias is not None and (conv.bias.dtype != torch.float32 or not conv.bias.is_contiguous()))
        or conv._forward_hooks or conv._forward_pre_hooks
        or (residual is not None and (residual.ndim != x.ndim or residual.shape[0] != len(x)
            or residual.shape[2:] != x.shape[2:] or residual.shape[1] not in (1, x.shape[1])
            or residual.dtype != x.dtype or residual.device != x.device))
    )
    if (compatible and x.shape[1] == 1 and 1 < conv.weight.shape[0] <= 64 and relu
            and residual is None and next_denominator is None and math.prod(conv.kernel_size) <= 25):
        return _stem(conv, x, active)
    if not compatible or conv.weight.shape[0] != x.shape[1]:
        result = conv(x)
        if residual is not None:
            result = residual + result
        if relu:
            result = result.relu()
        return result if next_denominator is None else (result, (result / next_denominator).relu())
    channels, height, width = x.shape[1:]
    kh, kw = conv.kernel_size
    (top, _), (left, _) = conv._circular_pads
    features = x.permute(0, 2, 3, 1).contiguous()
    # Read live weights after every update; there is no stale packed copy.
    weights = conv.weight.permute(2, 3, 1, 0).contiguous()
    result = torch.empty_like(features)
    skip = residual.permute(0, 2, 3, 1).contiguous() if residual is not None else None
    prepared = torch.empty_like(features) if next_denominator is not None else None
    next_scale = 1.0 / next_denominator if next_denominator is not None else 0.0
    npositions = len(x) * height * width
    bn = min(64, max(16, triton.next_power_of_2(channels)))
    _periodic_conv[(triton.cdiv(npositions, 64), triton.cdiv(channels, bn))](
        features, weights, conv.bias, result, skip, prepared, next_scale,
        *active[:2], npositions, active[2], height, width, channels,
        kh, kw, top, left, bn, conv.bias is not None, relu,
        0 if residual is None else residual.shape[1], prepared is not None, num_warps=2,
    )
    result = result.permute(0, 3, 1, 2)
    return result if prepared is None else (result, prepared.permute(0, 3, 1, 2))


@triton.jit(do_not_specialize=["start"])
def _reduce_features(X, SCALE, THETA0, EIGVAL, Y, COUNT, start, ORBIT: tl.constexpr, SITES: tl.constexpr, CHANNELS: tl.constexpr,
                     BLOCK: tl.constexpr, SINH_SHIFT: tl.constexpr):
    batch = tl.program_id(0)
    active = batch < _active_inputs(COUNT, start, ORBIT) if ORBIT else True
    if active:
        offset = tl.arange(0, BLOCK)
        valid = offset < SITES * CHANNELS
        value = tl.load(X + batch * SITES * CHANNELS + offset, valid, 0)
        # Scale rounds in FP32 before the activation promotes to FP64, matching
        # the two original modules. Keep both exponents (no algebraic rewrite).
        x = (value * tl.load(SCALE).to(tl.float32)).to(tl.float64)
        theta0 = tl.load(THETA0)
        activated = libdevice.exp(x + theta0)
        if SINH_SHIFT:
            activated = (activated - libdevice.exp(-x + theta0)) * 0.5 + libdevice.exp(theta0)
        # The input is NHWC in memory. The original reduction weights every
        # channel by the same table of spatial eigenvalues.
        weight = tl.load(EIGVAL + offset // CHANNELS, valid, 0)
        output = tl.sum(tl.where(valid, activated * weight, 0.0), 0) / (SITES * CHANNELS)
        tl.store(Y + batch, output)
    else:
        tl.store(Y + batch, 0.0)


def _finish_res_sum(x, layers, active=(None, 0, 0)):
    if len(layers) != 3:
        return None
    scale, activation, reduction = layers
    if (type(scale) is not Scale or type(activation) not in (SinhShift, Exp)
            or type(reduction) is not ConvSymmetrize
            or any(layer._forward_hooks or layer._forward_pre_hooks for layer in layers)
            or x.ndim != 4 or not len(x) or x.dtype != torch.float32
            or not x.is_contiguous(memory_format=torch.channels_last)
            or reduction.feature_dims != 3):
        return None
    nbatch, channels, height, width = x.shape
    sites = height * width
    if (sites * channels > 8192
            or scale.scale.ndim != 0 or scale.scale.dtype != torch.float64
            or activation.theta0.ndim != 0 or activation.theta0.dtype != torch.float64
            or reduction.eigval.shape != (sites,) or reduction.eigval.dtype != torch.float64
            or not reduction.eigval.is_contiguous()):
        return None
    out = torch.empty(nbatch, device=x.device, dtype=torch.float64)
    _reduce_features[(nbatch,)](
        x, scale.scale, activation.theta0, reduction.eigval, out, *active, sites, channels,
        triton.next_power_of_2(sites * channels), type(activation) is SinhShift,
        num_warps=2, enable_fp_fusion=False,
    )
    return out


def _fused_blocks(layers):
    blocks = layers[1:-3]
    if (blocks and type(layers[0]) is ReshapeConv
            and all(type(block) is ResBlock and block.nblock == i
                    and not (block._forward_hooks or block._forward_pre_hooks)
                    for i, block in enumerate(blocks))):
        return blocks
    return ()


def _run_blocks(x, blocks, active):
    y = x / blocks[0]._input_scale
    for i, block in enumerate(blocks):
        hidden = _convolve(block.conv1, y, relu=True, active=active)
        if i + 1 == len(blocks):
            return _convolve(block.conv2, hidden, residual=x, active=active)
        # Keep the unscaled residual and write the next block's scaled,
        # rectified input in the same kernel.
        x, y = _convolve(block.conv2, hidden, residual=x, active=active,
                         next_denominator=blocks[i + 1]._input_scale)


def res_sum_forward(net, spins):
    if (torch.is_grad_enabled() or torch.is_autocast_enabled("cuda")
            or not spins.is_cuda or type(net) is not Sequential
            or net._forward_hooks or net._forward_pre_hooks):
        return net(spins)
    layers = net.layers

    def run(x, subset):
        for layer in subset:
            x = layer(x, convolve=_convolve) if type(layer) is ResBlock else layer(x)
        return x

    blocks = _fused_blocks(layers)
    with torch.cuda.device(spins.device):
        features = (_run_blocks(layers[0](spins), blocks, (None, 0, 0)) if len(blocks) > 1
                    else run(spins, layers[:-3]))
        result = _finish_res_sum(features, layers[-3:])
        return run(features, layers[-3:]) if result is None else result


def supports_counted_res_sum(net):
    """Only capture networks whose convolutions and reduction can all skip inactive rows."""
    if type(net) is not Sequential or net._forward_hooks or net._forward_pre_hooks:
        return False
    layers = net.layers
    blocks = _fused_blocks(layers)
    if not blocks:
        return False
    reshape = layers[0]
    if len(reshape.shape) != 3 or reshape.shape[-1] != 1 or reshape.dtype != torch.float32:
        return False
    if any(layer._forward_hooks or layer._forward_pre_hooks for layer in (reshape, *layers[-3:])):
        return False
    conv = blocks[0].conv2
    if type(conv) is not Conv or conv.weight.ndim != 4:
        return False
    channels = conv.weight.shape[0]
    sites = math.prod(reshape.shape[:2])
    if not 1 <= channels <= 64 or sites * channels > 8192:
        return False
    for i, block in enumerate(blocks):
        for layer, ninput in ((block.conv1, channels if i else 1), (block.conv2, channels)):
            if (type(layer) is not Conv or layer.weight.ndim != 4
                    or layer.weight.shape[:2] != (channels, ninput) or layer.weight.dtype != torch.float32
                    or layer.padding != "CIRCULAR" or layer.stride not in (1, (1, 1))
                    or layer.dilation != (1, 1) or layer.groups != 1 or math.prod(layer.kernel_size) > 25
                    or layer._forward_hooks or layer._forward_pre_hooks
                    or (layer.bias is not None and (layer.bias.dtype != torch.float32 or not layer.bias.is_contiguous()))):
                return False
    scale, activation, reduction = layers[-3:]
    return (type(scale) is Scale and type(activation) in (SinhShift, Exp)
            and type(reduction) is ConvSymmetrize and reduction.feature_dims == 3
            and scale.scale.ndim == 0 and scale.scale.dtype == torch.float64
            and activation.theta0.ndim == 0 and activation.theta0.dtype == torch.float64
            and reduction.eigval.shape == (sites,) and reduction.eigval.dtype == torch.float64
            and reduction.eigval.is_contiguous())


def counted_res_sum_forward(net, spins, count, start, orbit):
    """Evaluate a capacity-sized chunk; return zero for padded network inputs."""
    layers = net.layers
    active = (count, start, orbit)
    with torch.cuda.device(spins.device):
        x = _run_blocks(layers[0](spins), layers[1:-3], active)
        return _finish_res_sum(x, layers[-3:], active)
