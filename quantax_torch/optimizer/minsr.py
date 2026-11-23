import logging
import math
import re
from itertools import islice
from operator import index

import torch
from torch import distributed
from torch.func import functional_call, vjp, vmap

from ._cuda_gram import add_gram_lower_, add_gram_tiled_, mirror_gram_, tune_gram
from ._res_sum import batched_res_sum_vjp
from .solver import minsr_pinv_eig
from .tdvp import TDVP, _stack_components
from ..nn import Sequential
from ..sampler import Samples


_DERIVATIVE_CACHE_BYTES = 320 * 2**20


def _cuda_memory(device):
    """Return allocator headroom and physical capacity in bytes."""
    free, total = torch.cuda.mem_get_info(device)
    with torch.cuda.device(device):
        device_index = torch.cuda.current_device()
        limit = int(total * torch.cuda.get_per_process_memory_fraction(device_index))
    stats = torch.cuda.memory_stats(device)
    allocated = stats.get("allocated_bytes.all.current", 0)
    reserved = stats.get("reserved_bytes.all.current", 0)
    fragmented = stats.get("inactive_split_bytes.all.current", 0)
    reusable = max(0, reserved - allocated - fragmented)
    return max(0, min(free + reusable, limit - allocated - fragmented)), total


def _retry_batch(batch_size, message, device):
    request = re.search(r"allocate (\d+(?:\.\d+)?) (bytes|[KMGT]iB)\b", message)
    if request is not None:
        unit = ("bytes", "KiB", "MiB", "GiB", "TiB").index(request[2])
        requested = float(request[1]) * 1024**unit
        try:
            available, _ = _cuda_memory(device)
        except RuntimeError:
            return batch_size // 2
        if 0 < available < requested:
            # Estimate how many samples fit with 20% headroom. The
            # failed allocation may also include fixed-size buffers.
            return max(1, int(0.8 * batch_size * available / requested))
    return batch_size // 2


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
    """Compute the sample Gram matrix and step over parameter blocks.

    ``jacobian_batch`` bounds the number of sampled configurations in a
    derivative batch. When omitted, CUDA chooses it from allocator headroom
    and estimated temporary storage. Gram and solve use all samples.
    """

    def __init__(self, state, hamiltonian, solver=None, *, jacobian_batch=None, process_group=None, logger=None):
        super().__init__(state, hamiltonian, solver=minsr_pinv_eig() if solver is None else solver)
        if self.vs_type == 1:
            raise ValueError("MinSR does not support non-holomorphic complex parameters.")
        if jacobian_batch is not None:
            jacobian_batch = index(jacobian_batch)
            if jacobian_batch < 1:
                raise ValueError("jacobian_batch must be positive.")
        self.jacobian_batch = jacobian_batch
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self.process_group = process_group
        self._layers = tuple(_block_layers(net) for net in state.ansatz)
        self._complex_params = any(p.is_complex() for p in state.parameters())
        self._vjp_backends = set()
        self._vjp_backend_choices = {}
        self._gram_backends = set()
        self._gram_backend_choices = {}
        self._last_cache_plan = (None, None)
        self._cache_limit = None
        self._jacobian_batch_limit = None
        self._last_jacobian_batch = None

    def _derivative_batch(self, spins):
        cap = min(len(spins), self._jacobian_batch_limit or len(spins))
        if self.jacobian_batch is not None:
            return min(cap, self.jacobian_batch)
        if not spins.is_cuda:
            return cap
        try:
            available, total = _cuda_memory(spins.device)
        except RuntimeError:
            return 1
        params = tuple(self.state.parameters())
        dtype_bytes = max((param.element_size() for param in params), default=spins.element_size())
        total_params = sum(param.numel() for param in params)
        blocks = [block for _, layers, _ in self._layers for block in layers]
        widest_block = max((sum(p.numel() for p in block.parameters() if p.requires_grad) for block in blocks),
                           default=total_params)
        channels = max((p.shape[0] for p in params if p.ndim >= 3), default=1)
        orbit = self.state.nsymm
        feature_bytes = orbit * self.state.nsites * channels * dtype_bytes
        # Saved block boundaries, cotangents, concatenation and live convolution tensors.
        per_sample = max(1, 2 * (len(blocks) + 4) * feature_bytes + widest_block * dtype_bytes)
        scalar_bytes = self.state.dtype.itemsize
        # The Gram is allocated by get_Tmat before we get here. Reserve derivative
        # outputs and FP64 centering temporaries that can coexist with activations.
        fixed = len(spins) * (total_params * max(dtype_bytes, scalar_bytes)
                              + 2 * widest_block * scalar_bytes)
        headroom = max(0, available - fixed - total // 10)
        return max(1, min(cap, headroom // (3 * per_sample)))

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

        real, imag = vmap(cotangents)(*endpoints)
        return positions, real, imag

    @torch.no_grad()
    def _block_vjp(self, block, x, delta_real, delta_imag, propagate):
        if delta_imag is None and not self._complex_params:
            result = batched_res_sum_vjp(block, x, delta_real, propagate, self.state.max_parallel,
                                         self._vjp_backend_choices)
            if result is not None:
                jac, upstream, periodic = result
                self._vjp_backends.add("periodic" if periodic else "unfold")
                return jac, upstream, None
        self._vjp_backends.add("autograd")
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

        batched = vmap(backward_one)
        if delta_imag is None:
            jac, real = batched(x, delta_real)
            return jac, real, None
        jac, real, imag = batched(x, delta_real, delta_imag)
        return jac, real, imag

    def _differentiate(self, spins, first_block=0):
        positions, real, imag = self._prepare(spins)
        block_index = sum(len(blocks) for _, blocks, _ in self._layers)
        for i in reversed(range(len(self._layers))):
            _, blocks, _ = self._layers[i]
            r = real[i]
            im = imag[i] if imag else None
            for j in reversed(range(len(blocks))):
                block_index -= 1
                if block_index < first_block:
                    return
                jac, r, im = self._block_vjp(blocks[j], positions[i][j], r, im,
                                             propagate=j > 0 and block_index > first_block)
                yield block_index, jac

    def _native_jacobian_blocks(self, spins, first_block=0):
        spins = torch.as_tensor(spins, device=self.state.device)
        nsamples = len(spins)
        batch_size = self._derivative_batch(spins)
        self._last_jacobian_batch = batch_size
        if batch_size >= nsamples:
            yield from self._differentiate(spins, first_block)
            return

        # Keep only this sample batch's activations. Write its derivatives
        # into full-size parameter blocks before centering across samples.
        blocks = {}
        for start in range(0, nsamples, batch_size):
            end = min(start + batch_size, nsamples)
            for i, jac in self._differentiate(spins[start:end], first_block):
                if i not in blocks:
                    blocks[i] = jac.new_empty((nsamples, jac.shape[1]))
                blocks[i][start:end].copy_(jac)
        for i in tuple(blocks):
            yield i, blocks.pop(i)

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
        tile = 128 if size <= 1024 else 256 if size <= 4096 else 512
        tune = (group is None and self.vs_type == 0 and T.is_cuda and dtype == torch.float64 and size >= 512
                and not torch.are_deterministic_algorithms_enabled()
                and not (torch.is_grad_enabled() and samples.reweight_factor.requires_grad))
        if tune:
            with torch.cuda.device(T.device):
                tune = not torch.cuda.is_current_stream_capturing()
        backend = "full GEMM"
        mirror = False
        lower_only = False
        blocks = self._native_jacobian_blocks if _native_cache else self.jacobian_blocks
        for i, raw in blocks(samples.spins):
            jac = raw.to(self.state.dtype) if _native_cache else raw
            means.append((jac * samples.reweight_factor[:, None]).mean(dim=0))
            if i % world == rank:
                block = self._normalized_block(jac, samples.reweight_factor)
                wide = block.shape[1] >= 4096
                if tune and len(means) == 1 and wide:
                    key = (T.device, block.shape)
                    if key not in self._gram_backend_choices:
                        with torch.cuda.device(T.device):
                            self._gram_backend_choices[key] = tune_gram(T, block, tile)
                    backend = self._gram_backend_choices[key]
                if wide and backend == "cuBLAS SYRK" and add_gram_lower_(T, block):
                    lower_only = True
                    self._gram_backends.add("cuBLAS SYRK")
                elif wide and backend == "tiled GEMM":
                    add_gram_tiled_(T, block, tile)
                    mirror = True
                    self._gram_backends.add("tiled GEMM")
                else:
                    T.addmm_(block, block.mH)
                    self._gram_backends.add("full GEMM")
                if _blocks is not None and (_retain_before is None or i < _retain_before):
                    _blocks.append(raw if _native_cache else block)
        if not means:
            raise ValueError("MinSR requires trainable parameters.")
        if lower_only or mirror:
            mirror_gram_(T, tile, lower_only)
        if group is not None:
            distributed.all_reduce(T, group=group)
        self._Omean = torch.cat(means[::-1])
        return T

    def _adjoint_blocks(self, samples, v, limit=None):
        if limit == 0:
            return []
        blocks = []
        weight = samples.reweight_factor
        if limit is None or type(self) is not MinSR:
            jacobians = islice(self.jacobian_blocks(samples.spins), limit)
        else:
            first = sum(len(blocks) for _, blocks, _ in self._layers) - limit
            jacobians = self._native_jacobian_blocks(samples.spins, first)
        for _, jac in jacobians:
            jac = jac.to(self.state.dtype)
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

    def _cache_budget(self, samples, element_size):
        device = samples.spins.device
        try:
            available, total = _cuda_memory(device)
        except RuntimeError:
            return _DERIVATIVE_CACHE_BYTES
        # Account for the Gram, eigensolver copies/workspace, and allocator headroom.
        solve_reserve = 3 * samples.nsamples**2 * element_size + max(512 * 2**20, total // 10)
        return max(0, int(available) - solve_reserve)

    def _cache_plan(self, samples, element_size):
        if (type(self) is not MinSR or self.process_group is not None or self.state.device.type != "cuda"
                or self.vs_type != 0 or self._complex_params):
            return None, None
        params = tuple(self.state.parameters())
        elements = samples.nsamples * sum(p.numel() for p in params)
        budget = self._cache_budget(samples, element_size)
        if self._cache_limit is not None:
            budget = min(budget, self._cache_limit)
        if elements * element_size <= min(budget, _DERIVATIVE_CACHE_BYTES):
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
        if self.state.device.type != "cuda" or self.process_group is not None:
            return self._step_from_energy(samples, Ebar)

        initial_batch_limit = self._jacobian_batch_limit
        initial_cache_limit = self._cache_limit
        try:
            while True:
                try:
                    return self._step_from_energy(samples, Ebar)
                except torch.cuda.OutOfMemoryError as error:
                    current = self._last_jacobian_batch or min(
                        self._jacobian_batch_limit or samples.nsamples,
                        self.jacobian_batch or samples.nsamples, samples.nsamples,
                    )
                    expanded_cache = self._cache_limit is None and self._last_cache_plan[0] == "native"
                    if current <= 1 and not expanded_cache:
                        raise
                    message = str(error)
                # Release references held by the failed attempt before
                # returning unused CUDA allocator blocks to the driver.
                torch.cuda.empty_cache()
                if expanded_cache:
                    self._cache_limit = _DERIVATIVE_CACHE_BYTES
                    self._logger.warning("MinSR CUDA out of memory. Retrying with a smaller derivative cache.")
                    continue
                self._jacobian_batch_limit = _retry_batch(current, message, Ebar.device)
                self._logger.warning(
                    f"MinSR CUDA out of memory. Retrying with Jacobian batch={self._jacobian_batch_limit} (was {current})"
                )
        except BaseException:
            self._jacobian_batch_limit = initial_batch_limit
            self._cache_limit = initial_cache_limit
            raise

    def _step_from_energy(self, samples, Ebar):
        self._last_jacobian_batch = None
        self._vjp_backends.clear()
        self._gram_backends.clear()
        cache, retain_before = self._cache_plan(samples, Ebar.element_size())
        self._last_cache_plan = cache, retain_before
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

    def log_backends(self, logger):
        vjp = ", ".join(sorted(self._vjp_backends))
        gram = ", ".join(sorted(self._gram_backends))
        cache, retained = self._last_cache_plan
        cache_status = cache or "off"
        if retained is not None:
            cache_status += f" ({retained} blocks)"
        batch = self._last_jacobian_batch
        detail = f" | {batch} samples/batch" if batch is not None else ""
        logger.info(f"CUDA MinSR derivatives: {vjp} | cache={cache_status}{detail}")
        logger.info(f"CUDA MinSR Gram: {gram}")
