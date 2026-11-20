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
        for uniform in uniforms:
            proposal, proposal_prob = self._propose(status.spins)
            proposed_wf = self.state(proposal)
            new = SamplerStatus(proposal, proposed_wf, proposed_wf.abs().pow(self.reweight), proposal_prob)
            status = self._update(status, new, uniform)
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
    def __init__(self, state, nsamples, reweight=2.0, thermal_steps=None,
                 sweep_steps=None, initial_spins=None, n_neighbor=1, *, generator=None):
        if state.total_sz is None:
            raise ValueError("NeighborExchange requires a fixed magnetization sector.")
        shells = [n_neighbor] if isinstance(n_neighbor, int) else n_neighbor
        bonds = np.concatenate(state.symm.sites.get_neighbor(shells), axis=0)
        self.neighbors = torch.as_tensor(bonds, dtype=torch.long, device=state.device)
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
