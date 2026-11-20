import numpy as np
import torch

from .sampler import Sampler, SamplerStatus
from ..utils.spins import rand_spins


class Metropolis(Sampler):
    def __init__(self, state, nsamples, reweight=2.0, thermal_steps=None,
                 sweep_steps=None, initial_spins=None, *, generator=None):
        super().__init__(state, nsamples, reweight, generator=generator)
        self.thermal_steps = 20 * self.nsites if thermal_steps is None else thermal_steps
        self.sweep_steps = 2 * self.nsites if sweep_steps is None else sweep_steps
        if self.thermal_steps < 0 or self.sweep_steps < 0:
            raise ValueError("Sampling steps must be nonnegative.")
        self.initial_spins = initial_spins
        self._rows = torch.arange(nsamples, device=state.device)
        self.reset()

    @torch.no_grad()
    def reset(self):
        if self.initial_spins is None:
            spins = rand_spins(self.nsamples, self.state.total_sz, nsites=self.nsites,
                               device=self.state.device, generator=self.generator)
        else:
            spins = torch.as_tensor(self.initial_spins, dtype=torch.int8, device=self.state.device)
            spins = spins.expand(self.nsamples, -1) if spins.ndim == 1 else spins.reshape(self.nsamples, self.nsites)
            if spins.shape[-1] != self.nsites or not bool(((spins == 1) | (spins == -1)).all()):
                raise ValueError("Initial spins must be a batch of -1/+1 configurations.")
            if self.state.total_sz is not None and not bool((spins.sum(dim=1) == self.state.total_sz).all()):
                raise ValueError("Initial spins do not match the magnetization sector.")
        # The reference sampler makes one initial proposal before thermalization.
        spins, proposal = self._propose(spins)
        self._status = SamplerStatus(spins, propose_prob=proposal)
        self.sweep(self.thermal_steps)

    @staticmethod
    def _update(old, new, uniform):
        accepted = (new.prob * old.propose_prob > old.prob * new.propose_prob * uniform) | (old.prob == 0)
        return SamplerStatus(
            torch.where(accepted[:, None], new.spins, old.spins),
            torch.where(accepted, new.wave_function, old.wave_function),
            torch.where(accepted, new.prob, old.prob),
            torch.where(accepted, new.propose_prob, old.propose_prob),
        )

    def _advance(self, status, uniform, positions=None):
        if positions is None:
            proposal, proposal_prob = self._propose(status.spins)
        else:
            proposal, proposal_prob = self._propose(status.spins, positions)
        wf = self.state(proposal)
        return self._update(status, SamplerStatus(proposal, wf, wf.abs().pow(self.reweight), proposal_prob), uniform)

    def _proposals(self, status, uniforms):
        for uniform in uniforms:
            status = self._advance(status, uniform)
        return status

    @torch.no_grad()
    def sweep(self, nsweeps=None):
        nsweeps = self.sweep_steps if nsweeps is None else nsweeps
        if nsweeps < 0:
            raise ValueError("nsweeps must be nonnegative.")
        spins = self.current_spins
        wf = self.state(spins)
        prob = wf.abs().pow(self.reweight)
        status = SamplerStatus(spins, wf, prob, self._status.propose_prob)
        uniforms = 1 - torch.rand((nsweeps, self.nsamples), generator=self.generator,
                                  dtype=prob.dtype, device=prob.device)
        status = self._proposals(status, uniforms)
        self._status = status
        return self._samples(status.spins, status.wave_function)

    def _uniform_proposal(self, spins):
        return torch.ones(self.nsamples, device=spins.device, dtype=torch.float64)


class LocalFlip(Metropolis):
    def _propose(self, spins, positions=None):
        if positions is None:
            positions = torch.randint(self.nsites, (self.nsamples,), device=spins.device, generator=self.generator)
        result = spins.clone()
        result[self._rows, positions] *= -1
        return result, self._uniform_proposal(spins)


class NeighborExchange(Metropolis):
    """Exchange sampler with optional graph replay and compaction of changed proposals.

    With both flags set, supported projected CUDA networks replay compacted
    proposals; other networks use eager compaction.
    """

    def __init__(self, state, nsamples, reweight=2.0, thermal_steps=None,
                 sweep_steps=None, initial_spins=None, n_neighbor=1, *, generator=None,
                 cuda_graph=False, skip_unchanged=False):
        if state.total_sz is None:
            raise ValueError("NeighborExchange requires a fixed magnetization sector.")
        shells = [n_neighbor] if isinstance(n_neighbor, int) else n_neighbor
        bonds = np.concatenate(state.symm.sites.get_neighbor(shells), axis=0)
        self.neighbors = torch.as_tensor(bonds, dtype=torch.long, device=state.device)
        self._use_cuda_graph = cuda_graph and state.device.type == "cuda"
        self._skip_unchanged = skip_unchanged
        self._cuda_graph = None
        self._compact_key = None
        super().__init__(state, nsamples, reweight, thermal_steps, sweep_steps,
                         initial_spins, generator=generator)

    def _pairs(self, spins, positions):
        if positions is None:
            positions = torch.randint(len(self.neighbors), (self.nsamples,), device=spins.device, generator=self.generator)
        return self.neighbors[positions]

    def _exchange(self, spins, pairs):
        rows = self._rows[:len(spins), None]
        result = spins.clone()
        result[rows, pairs] = spins[rows, pairs.flip(dims=(1,))]
        return result

    def _propose(self, spins, positions=None):
        return self._exchange(spins, self._pairs(spins, positions)), self._uniform_proposal(spins)

    def _advance(self, status, uniform, positions=None):
        if not self._skip_unchanged:
            return super()._advance(status, uniform, positions)
        spins = status.spins
        pairs = self._pairs(spins, positions)
        left, right = pairs.unbind(dim=1)
        # Equal-spin swaps are self-transitions. Their wave functions were
        # already evaluated under the current parameters in this sweep.
        (changed,) = torch.nonzero(spins[self._rows, left] != spins[self._rows, right], as_tuple=True)
        if not changed.numel():
            return status
        old = SamplerStatus(spins[changed], status.wave_function[changed],
                            status.prob[changed], status.propose_prob[changed])
        proposal = self._exchange(old.spins, pairs[changed])
        wf = self.state(proposal)
        new = SamplerStatus(proposal, wf, wf.abs().pow(self.reweight), torch.ones_like(old.propose_prob))
        updated = self._update(old, new, uniform[changed])
        return SamplerStatus(
            spins.index_copy(0, changed, updated.spins),
            status.wave_function.index_copy(0, changed, updated.wave_function),
            status.prob.index_copy(0, changed, updated.prob),
            status.propose_prob.index_copy(0, changed, updated.propose_prob),
        )

    def _compact_backend(self):
        state = self.state
        if (not callable(getattr(torch, "nonzero_static", None)) or state.nsymm <= 1
                or getattr(state, "inference_fn", None) is None or torch.are_deterministic_algorithms_enabled()
                or torch.is_autocast_enabled("cuda")):
            return None
        from ..nn._cuda_support import supports_tuned_kernels
        from ..state import Variational
        from ..symmetry import Symmetry

        if (type(state) is not Variational or type(state.symm) is not Symmetry
                or not (state._default_input and state._default_output)
                or not supports_tuned_kernels(state.device)):
            return None
        from ..nn._periodic_cuda import counted_res_sum_forward, res_sum_forward, supports_counted_res_sum

        if state.inference_fn is not res_sum_forward or not supports_counted_res_sum(state.ansatz[0]):
            return None
        cap = state.forward_batch
        batch = self.nsamples if cap is None else cap // state.nsymm
        # Bound the capture size: one subgraph per possible inference chunk.
        if batch <= 0 or self.nsamples > 32 * batch or min(batch, self.nsamples) * state.nsymm > 2048:
            return None
        return counted_res_sum_forward, batch

    def _advance_compact(self, status, uniform, positions, forward, batch):
        state = self.state
        spins = status.spins
        pairs = self._pairs(spins, positions)
        left, right = pairs.unbind(dim=1)
        changed = spins[self._rows, left] != spins[self._rows, right]
        count = changed.sum(dtype=torch.int32)
        # Keep storage addresses and launch sizes fixed. The convolution
        # kernels skip padding using the true count, read on the device.
        chosen = torch.nonzero_static(changed, size=self.nsamples, fill_value=-1).flatten().clamp_min_(0)
        proposal = self._exchange(spins[chosen], pairs[chosen])
        values = []
        for offset, chunk in enumerate(proposal.split(batch)):
            (inputs,) = state._inputs(chunk)
            wf = forward(state.ansatz[0], inputs.flatten(0, 1), count, offset * batch, state.nsymm)
            values.append(state._output((wf.reshape(inputs.shape[:-1]),)))
        wave_function = torch.cat(values)
        state._record_maximum(wave_function)

        # Undo compaction in chain order without an index write to a padded
        # row. An equal-spin exchange must leave that chain untouched.
        inverse = (changed.cumsum(0) - 1).clamp_min_(0)
        wave = torch.where(changed, wave_function[inverse], status.wave_function)
        prob = torch.where(changed, wave.abs().pow(self.reweight), status.prob)
        next_spins = torch.where(changed[:, None], proposal[inverse], spins)
        proposal_prob = torch.where(changed, 1.0, status.propose_prob)
        return self._update(status, SamplerStatus(next_spins, wave, prob, proposal_prob), uniform)

    def _proposals(self, status, uniforms):
        if not self._use_cuda_graph or not len(uniforms):
            return super()._proposals(status, uniforms)
        if self._skip_unchanged:
            backend = self._compact_backend()
            if backend is None:
                self._cuda_graph = self._compact_key = None
                return super()._proposals(status, uniforms)
            forward, batch = backend
            state = self.state
            net = state.ansatz[0]
            # Updates in-place keep the captured addresses valid. Rebuild
            # after changes to the network storage or to the batch split.
            tensors = (self.neighbors, self._rows, state._maximum, *net.parameters(), *net.buffers(),
                       *state.symm._on_device(status.spins.device))
            key = (batch, state.nsymm, id(net), id(state.symm), id(state.input_fn), id(state.output_fn),
                   id(self.generator), self.reweight,
                   tuple((x.data_ptr(), x.shape, x.stride(), x.dtype) for x in tensors))
            if self._cuda_graph is None or self._compact_key != key:
                def advance(old, uniform, positions=None):
                    return self._advance_compact(old, uniform, positions, forward, batch)

                self._cuda_graph = _ExchangeGraph(self, status, uniforms[0], advance=advance)
                self._compact_key = key
        elif self._cuda_graph is None:
            self._cuda_graph = _ExchangeGraph(self, status, uniforms[0])
        return self._cuda_graph(status, uniforms)


class _ExchangeGraph:
    """Replay one exchange proposal, with fixed chain buffers and fresh CUDA RNG."""

    _fields = ("spins", "wave_function", "prob", "propose_prob")

    def __init__(self, sampler, status, uniform, *, advance=None):
        advance = sampler._advance if advance is None else advance
        self.status = SamplerStatus(*(getattr(status, name).clone() for name in self._fields))
        self.uniform = uniform.clone()
        device = status.spins.device
        positions = torch.zeros(sampler.nsamples, device=device, dtype=torch.long)
        maximum = sampler.state._maximum.clone()
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            # Warm up cuDNN and the other kernels without drawing a proposal
            # or adding the warmup amplitude to the scale estimate.
            advance(self.status, self.uniform, positions)
            sampler.state._maximum.copy_(maximum)
        torch.cuda.current_stream(device).wait_stream(stream)
        with torch.cuda.device(device):
            self.graph = torch.cuda.CUDAGraph()
            self.graph.register_generator_state(sampler.generator)
            with torch.cuda.graph(self.graph, stream=stream):
                self._copy(advance(self.status, self.uniform))

    def _copy(self, status):
        for name in self._fields:
            getattr(self.status, name).copy_(getattr(status, name))

    def __call__(self, status, uniforms):
        self._copy(status)
        for uniform in uniforms:
            self.uniform.copy_(uniform)
            self.graph.replay()
        # Samples returned to the caller must not alias the replay buffers.
        return SamplerStatus(*(getattr(self.status, name).clone() for name in self._fields))
