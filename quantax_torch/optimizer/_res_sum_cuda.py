"""Per-sample periodic convolution weight gradients, without an im2col tensor."""

import torch
import triton
import triton.language as tl


@triton.jit
def _weight_grad(X, D, Y, H: tl.constexpr, W: tl.constexpr, CI: tl.constexpr, CO: tl.constexpr,
                 KH: tl.constexpr, KW: tl.constexpr, TOP: tl.constexpr, LEFT: tl.constexpr,
                 XS: tl.constexpr, XC: tl.constexpr, XH: tl.constexpr, XW: tl.constexpr,
                 DS: tl.constexpr, DC: tl.constexpr, DH: tl.constexpr, DW: tl.constexpr, ORBIT: tl.constexpr):
    sample = tl.program_id(0)
    out = tl.program_id(1) * 16 + tl.arange(0, 16)
    groups: tl.constexpr = (CI + 15) // 16
    tile = tl.program_id(2)
    tap = tile // groups
    channel = tile % groups * 16 + tl.arange(0, 16)
    accum = tl.full((16, 16), 0, tl.float32)
    # One coefficient per (out-channel, in-channel, tap). Reduce lattice
    # sites and all symmetry members of this sample; never mix Gram rows.
    for step in range((ORBIT * H * W + 31) // 32):
        pos = step * 32 + tl.arange(0, 32)
        member = sample * ORBIT + pos // (H * W) if ORBIT > 1 else sample + tl.full((32,), 0, tl.int32)
        h, w = pos // W % H, pos % W
        ih = (h + tap // KW + (H - TOP % H)) % H
        iw = (w + tap % KW + (W - LEFT % W)) % W
        cotangent = tl.load(D + member[None, :] * DS + out[:, None] * DC + h[None, :] * DH + w[None, :] * DW,
                            (out[:, None] < CO) & (pos[None, :] < ORBIT * H * W), 0.0)
        x = tl.load(X + member[:, None] * XS + ih[:, None] * XH + iw[:, None] * XW + channel[None, :] * XC,
                    (pos[:, None] < ORBIT * H * W) & (channel[None, :] < CI), 0.0)
        accum = tl.dot(cotangent, x, accum, input_precision="ieee")
    offset = ((sample * CO + out[:, None]) * CI + channel[None, :]) * (KH * KW) + tap
    tl.store(Y + offset, accum, (out[:, None] < CO) & (channel[None, :] < CI))


def periodic_weight_grad(conv, x, delta, orbit=1):
    batch, ci, height, width = x.shape
    co = delta.shape[1]
    kh, kw = conv.kernel_size
    (top, _), (left, _) = conv._circular_pads
    batch //= orbit
    out = torch.empty((batch, co * ci * kh * kw), device=x.device, dtype=x.dtype)
    with torch.cuda.device(x.device):
        _weight_grad[(batch, triton.cdiv(co, 16), triton.cdiv(ci, 16) * kh * kw)](
            x, delta, out, height, width, ci, co, kh, kw, top, left,
            *x.stride(), *delta.stride(), orbit, num_warps=2,
        )
    return out
