import math
from itertools import islice

import torch
from torch import distributed
from torch.func import functional_call, vjp, vmap

from ._cuda_gram import add_gram_lower_
from ._res_sum import batched_res_sum_vjp
from .solver import minsr_pinv_eig
from .tdvp import TDVP, _stack_components
from ..nn import Sequential
from ..sampler import Samples


_DERIVATIVE_CACHE_BYTES = 320 * 2**20


def _gram_tile(T):
    # On the tested consumer CUDA architecture, FP64 GEMM is expensive
    # enough to benefit from not computing both halves of a large Gram.
    n = T.shape[0]
    if (512 <= n <= 4096 and T.dtype == torch.float64 and T.is_cuda and torch.version.cuda is not None
            and torch.cuda.get_device_capability(T.device) == (12, 0)):
        return 128 if n <= 1024 else 256
    return 0


def _active_params(module):
    return {name: p.detach() for name, p in module.named_parameters() if p.requires_grad}


def _block_layers(net):
    if not _active_params(net):
        return Sequential([]), [], net
    if not isinstance(net, Sequential):
        return Sequential([]), [net], Sequential([])
    active = [i for i, layer in enumerate(net) if _active_params(layer)]
    width = math.isqrt(len(active))
    start = active[0]
    prefix = net[:start]
    blocks = []
    for index in range(width - 1, len(active) + width - 1, width):
        end = active[min(index, len(active) - 1)] + 1
        blocks.append(net[start:end])
        start = end
    return prefix, blocks, net[start:]


def _on_orbits(module, x, params=None):
    def call(values):
        return module(values) if params is None else functional_call(module, params, (values,))

    if isinstance(module, Sequential):
        return call(x)
    return vmap(call)(x)


class MinSR(TDVP):
    """Compute the sample Gram matrix and step over parameter blocks."""

    def __init__(self, state, hamiltonian, solver=None, *, jacobian_batch=8, process_group=None):
        super().__init__(state, hamiltonian, solver=minsr_pinv_eig() if solver is None else solver)
        if self.vs_type == 1:
            raise ValueError("MinSR does not support non-holomorphic complex parameters.")
        if jacobian_batch < 1:
            raise ValueError("jacobian_batch must be positive.")
        self.jacobian_batch = jacobian_batch
        self.process_group = process_group
        self._layers = tuple(_block_layers(net) for net in state.ansatz)
        self._complex_params = any(p.is_complex() for p in state.parameters())

    @torch.no_grad()
    def _prepare(self, spins):
        """Save only the inputs to the parameter blocks and the final output cotangents."""
        self.state._raw(spins[:1])
        inputs = self.state._inputs(spins)
        shapes = [x.shape[1:-1] for x in inputs]
        positions = []
        endpoints = []
        for (prefix, blocks, tail), x in zip(self._layers, inputs):
            orbit_size = math.prod(x.shape[1:-1])
            chunks = []
            ends = []
            forward_batch = self.state.max_parallel
            max_samples = len(spins) if forward_batch is None else max(1, forward_batch // orbit_size)
            for s in x.reshape(len(spins), orbit_size, -1).split(max_samples):
                def apply(module, values):
                    result = _on_orbits(module, values.flatten(0, 1))
                    return result.unflatten(0, values.shape[:2])

                step_inputs = []
                values = apply(prefix, s)
                for block in blocks:
                    step_inputs.append(values)
                    values = apply(block, values)
                if not blocks:
                    values = apply(tail, values)
                chunks.append(step_inputs)
                ends.append(values)
            positions.append([torch.cat(group) if len(group) > 1 else group[0] for group in zip(*chunks)])
            endpoint = torch.cat(ends) if len(ends) > 1 else ends[0]
            endpoints.append(endpoint)

        def cotangents(*values):
            def projected(*before):
                outputs = []
                for (_, blocks, tail), value, shape in zip(self._layers, before, shapes):
                    wf = _on_orbits(tail, value) if blocks else value
                    outputs.append(wf.reshape(shape))
                psi = self.state.output_fn(tuple(outputs))
                return psi / psi.detach()

            psi, pullback = vjp(projected, *values)
            real = pullback(torch.ones_like(psi))
            if self.vs_type == 2:
                imag = pullback(torch.full_like(psi, 1j))
                return real, imag
            return real, ()

        real, imag = vmap(cotangents, chunk_size=self.jacobian_batch)(*endpoints)
        return positions, real, imag

    @torch.no_grad()
    def _block_vjp(self, block, x, delta_real, delta_imag, propagate):
        if delta_imag is None and not self._complex_params:
            result = batched_res_sum_vjp(block, x, delta_real, propagate, self.state.max_parallel)
            if result is not None:
                jac, upstream = result
                return jac, upstream, None
        params = _active_params(block)

        def backward_one(x, real, imag=None):
            if propagate:
                _, pullback = vjp(lambda p, s: _on_orbits(block, s, p), params, x)
            else:
                _, pullback = vjp(lambda p: _on_orbits(block, x, p), params)
            back_real = pullback(real)
            grad_real = torch.cat([p.flatten() for p in back_real[0].values()])
            upstream_real = back_real[1] if propagate else torch.empty(0, device=x.device)
            if imag is None:
                grad = grad_real.conj() if self._complex_params else grad_real
                return grad, upstream_real
            back_imag = pullback(imag)
            grad_imag = torch.cat([p.flatten() for p in back_imag[0].values()])
            upstream_imag = back_imag[1] if propagate else torch.empty(0, device=x.device)
            return torch.complex(grad_real, grad_imag), upstream_real, upstream_imag

        batched = vmap(backward_one, chunk_size=self.jacobian_batch)
        if delta_imag is None:
            jac, real = batched(x, delta_real)
            return jac, real, None
        jac, real, imag = batched(x, delta_real, delta_imag)
        return jac, real, imag

    def _native_jacobian_blocks(self, spins):
        spins = torch.as_tensor(spins, device=self.state.device)
        positions, real, imag = self._prepare(spins)
        block_index = sum(len(blocks) for _, blocks, _ in self._layers)
        for i in reversed(range(len(self._layers))):
            _, blocks, _ = self._layers[i]
            r = real[i]
            im = imag[i] if imag else None
            for j in reversed(range(len(blocks))):
                block_index -= 1
                jac, r, im = self._block_vjp(blocks[j], positions[i][j], r, im, propagate=j > 0)
                yield block_index, jac

    def jacobian_blocks(self, spins):
        """Yield state-dtype logarithmic derivatives in reverse network order."""
        for i, jac in self._native_jacobian_blocks(spins):
            yield i, jac.to(self.state.dtype)

    def _normalized_block(self, jac, weight):
        centered = self._center_jacobian(jac, weight)
        return _stack_components(centered) if self.vs_type == 2 else centered

    def get_Tmat(self, samples, *, _blocks=None, _native_cache=False, _retain_before=None):
        size = samples.nsamples * (2 if self.vs_type == 2 else 1)
        dtype = samples.wave_function.real.dtype if self.vs_type == 2 else self.state.dtype
        T = torch.zeros((size, size), device=self.state.device, dtype=dtype)
        means = []
        group = self.process_group
        rank = 0 if group is None else distributed.get_rank(group)
        world = 1 if group is None else distributed.get_world_size(group)
        tile = _gram_tile(T) if group is None and self.vs_type == 0 else 0
        mirror = False
        lower_only = False
        blocks = self._native_jacobian_blocks if _native_cache else self.jacobian_blocks
        for i, raw in blocks(samples.spins):
            jac = raw.to(self.state.dtype) if _native_cache else raw
            means.append((jac * samples.reweight_factor[:, None]).mean(dim=0))
            if i % world == rank:
                block = self._normalized_block(jac, samples.reweight_factor)
                if tile and block.shape[1] >= 4096:
                    if 512 <= size <= 1024 and add_gram_lower_(T, block):
                        lower_only = True
                    else:
                        # Below-diagonal tile columns plus full diagonal tiles.
                        for start in range(0, size, tile):
                            end = min(start + tile, size)
                            T[start:, start:end].addmm_(block[start:], block[start:end].mT)
                        mirror = True
                else:
                    T.addmm_(block, block.mH)
                if _blocks is not None and (_retain_before is None or i < _retain_before):
                    _blocks.append(raw if _native_cache else block)
        if not means:
            raise ValueError("MinSR requires trainable parameters.")
        if lower_only:
            T = T.tril() + T.tril(-1).mT
        elif mirror:
            # Reconstruct the missing off-diagonal tiles in-place, including
            # contributions from small blocks accumulated with full GEMMs.
            for start in range(tile, size, tile):
                end = min(start + tile, size)
                T[:start, start:end].copy_(T[start:end, :start].mT)
        if group is not None:
            distributed.all_reduce(T, group=group)
        self._Omean = torch.cat(means[::-1])
        return T

    def _adjoint_blocks(self, samples, v, limit=None):
        blocks = []
        weight = samples.reweight_factor
        for _, jac in islice(self.jacobian_blocks(samples.spins), limit):
            block = self._normalized_block(jac, weight)
            blocks.append(block.mH @ v)
        return blocks[::-1]

    def Ohvp(self, samples, v):
        return torch.cat(self._adjoint_blocks(samples, v))

    def solve(self, samples, Tmat, Ebar):
        rhs = _stack_components(Ebar) if self.vs_type == 2 else Ebar
        return self.Ohvp(samples, self._solve(Tmat, rhs))

    def _gather_samples(self, samples):
        group = self.process_group
        if group is None:
            return samples
        world = distributed.get_world_size(group)

        def gather(x):
            outputs = [torch.empty_like(x) for _ in range(world)]
            distributed.all_gather(outputs, x.contiguous(), group=group)
            return torch.cat(outputs)

        weights = gather(samples._unnormalized_weight)
        return Samples(gather(samples.spins), gather(samples.wave_function), weights / weights.mean())

    def _cache_plan(self, samples, element_size):
        if (type(self) is not MinSR or self.process_group is not None or self.state.device.type != "cuda"
                or self.vs_type != 0 or self._complex_params):
            return None, None
        params = tuple(self.state.parameters())
        elements = samples.nsamples * sum(p.numel() for p in params)
        budget = _DERIVATIVE_CACHE_BYTES
        if elements * element_size <= budget:
            return "normalized", None
        if self.state.dtype != torch.float64 or not all(p.dtype == torch.float32 for p in params):
            return None, None
        if elements * 4 <= budget:
            return "native", None

        # Prefer the input-side prefix: the second pass can then stop before
        # those layers, avoiding their input as well as weight gradients.
        retained = 0
        for _, blocks, _ in self._layers:
            for block in blocks:
                budget -= samples.nsamples * 4 * sum(p.numel() for p in block.parameters() if p.requires_grad)
                if budget < 0:
                    return ("native", retained) if retained else (None, None)
                retained += 1
        return "native", None

    @torch.no_grad()
    def get_step(self, samples):
        samples = self._gather_samples(samples)
        Ebar = self.get_Ebar(samples)
        cache, retain_before = self._cache_plan(samples, Ebar.element_size())
        if cache is None:
            T = self.get_Tmat(samples)
            step = self.solve(samples, T, Ebar)
        else:
            blocks = []
            native = cache == "native"
            T = self.get_Tmat(samples, _blocks=blocks, _native_cache=native, _retain_before=retain_before)
            v = self._solve(T, Ebar)
            step_blocks = []
            while blocks:
                block = blocks.pop()
                if native:
                    # The usual VJP already produced FP32 derivatives. Only
                    # the centering and solve use FP64; redo the centering at
                    # that precision instead of saving a promoted tensor.
                    block = self._normalized_block(block.to(self.state.dtype), samples.reweight_factor)
                step_blocks.append(block.mH @ v)
                del block
            if retain_before is not None:
                total_blocks = sum(len(blocks) for _, blocks, _ in self._layers)
                step_blocks.extend(self._adjoint_blocks(samples, v, total_blocks - retain_before))
            step = torch.cat(step_blocks)
        return self.build_step(step, self._Omean, self.energy)
