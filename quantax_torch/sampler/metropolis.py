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
    """Exchange sampler; graph replay requires a fixed, in-place-updated CUDA state."""

    def __init__(self, state, nsamples, reweight=2.0, thermal_steps=None,
                 sweep_steps=None, initial_spins=None, n_neighbor=1, *, generator=None, cuda_graph=False):
        if state.total_sz is None:
            raise ValueError("NeighborExchange requires a fixed magnetization sector.")
        shells = [n_neighbor] if isinstance(n_neighbor, int) else n_neighbor
        bonds = np.concatenate(state.symm.sites.get_neighbor(shells), axis=0)
        self.neighbors = torch.as_tensor(bonds, dtype=torch.long, device=state.device)
        self._use_cuda_graph = cuda_graph and state.device.type == "cuda"
        self._cuda_graph = None
        super().__init__(state, nsamples, reweight, thermal_steps, sweep_steps,
                         initial_spins, generator=generator)

    def _propose(self, spins, positions=None):
        if positions is None:
            positions = torch.randint(len(self.neighbors), (self.nsamples,), device=spins.device, generator=self.generator)
        pairs = self.neighbors[positions]
        rows = self._rows[:, None]
        result = spins.clone()
        result[rows, pairs] = spins[rows, pairs.flip(dims=(1,))]
        return result, self._uniform_proposal(spins)

    def _proposals(self, status, uniforms):
        if not self._use_cuda_graph or not len(uniforms):
            return super()._proposals(status, uniforms)
        if self._cuda_graph is None:
            self._cuda_graph = _ExchangeGraph(self, status, uniforms[0])
        return self._cuda_graph(status, uniforms)


class _ExchangeGraph:
    """Replay one exchange proposal, with fixed chain buffers and fresh CUDA RNG."""

    _fields = ("spins", "wave_function", "prob", "propose_prob")

    def __init__(self, sampler, status, uniform):
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
            sampler._advance(self.status, self.uniform, positions)
            sampler.state._maximum.copy_(maximum)
        torch.cuda.current_stream(device).wait_stream(stream)
        with torch.cuda.device(device):
            self.graph = torch.cuda.CUDAGraph()
            self.graph.register_generator_state(sampler.generator)
            with torch.cuda.graph(self.graph, stream=stream):
                self._copy(sampler._advance(self.status, self.uniform))

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
