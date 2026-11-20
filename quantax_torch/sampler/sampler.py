from dataclasses import dataclass
from numbers import Real

import torch

from ..global_defs import get_generator
from ..utils.spins import ints_to_array, rand_spins


class Samples:
    def __init__(self, spins, wave_function, reweight=2.0):
        self.wave_function = torch.as_tensor(wave_function)
        self.spins = torch.as_tensor(spins, device=self.wave_function.device)
        if isinstance(reweight, Real):
            factor = self.wave_function.abs().pow(2 - reweight)
            self.reweight_factor = factor / factor.mean()
            self._unnormalized_weight = factor
        else:
            self.reweight_factor = torch.as_tensor(reweight, device=self.spins.device)
            self._unnormalized_weight = self.reweight_factor
        if self.spins.ndim != 2 or self.wave_function.shape != self.spins.shape[:1]:
            raise ValueError("Expected a batch of spins and one wave function value per sample.")
        if self.reweight_factor.shape != self.wave_function.shape:
            raise ValueError("Expected one reweighting factor per sample.")

    @property
    def nsamples(self):
        return len(self.spins)

    def __getitem__(self, index):
        spins = self.spins[index]
        return Samples(spins.reshape(-1, self.spins.shape[-1]), self.wave_function[index].reshape(-1),
                       self.reweight_factor[index].reshape(-1))


@dataclass
class SamplerStatus:
    spins: torch.Tensor | None = None
    wave_function: torch.Tensor | None = None
    prob: torch.Tensor | None = None
    propose_prob: torch.Tensor | None = None


class Sampler:
    def __init__(self, state, nsamples, reweight=2.0, *, generator=None):
        if nsamples < 1:
            raise ValueError("nsamples must be positive.")
        self.state = state
        self.nsamples = nsamples
        self.reweight = reweight
        self.generator = get_generator(state.device) if generator is None else generator
        self._status = SamplerStatus()

    @property
    def nsites(self):
        return self.state.nsites

    @property
    def current_spins(self):
        return self._status.spins

    @property
    def current_wf(self):
        return self._status.wave_function

    @property
    def current_prob(self):
        return self._status.prob

    def _samples(self, spins, wf):
        samples = Samples(spins, wf, self.reweight)
        self._status = SamplerStatus(spins, wf, wf.abs().pow(self.reweight), self._status.propose_prob)
        return samples


class ExactSampler(Sampler):
    def __init__(self, state, nsamples, reweight=2.0, symm=None, *, generator=None):
        super().__init__(state, nsamples, reweight, generator=generator)
        self.symm = state.symm if symm is None else symm

    @torch.no_grad()
    def sweep(self):
        dense = self.state.todense(self.symm)
        weights = dense.wave_function.abs().pow(self.reweight)
        indices = torch.multinomial(weights, self.nsamples, replacement=True, generator=self.generator)
        ints = self.symm.basis.states[indices.cpu().numpy()]
        spins = torch.as_tensor(ints_to_array(ints, self.nsites), device=self.state.device)
        orbit = self.symm.get_symm_spins(spins)
        selection = torch.randint(orbit.shape[1], (self.nsamples,), device=spins.device, generator=self.generator)
        spins = orbit[torch.arange(self.nsamples, device=spins.device), selection]
        return self._samples(spins, self.state(spins))


class RandomSampler(Sampler):
    def __init__(self, state, nsamples, *, generator=None):
        super().__init__(state, nsamples, 0.0, generator=generator)

    @torch.no_grad()
    def sweep(self):
        spins = rand_spins(self.nsamples, self.state.total_sz, nsites=self.nsites,
                           device=self.state.device, generator=self.generator)
        return self._samples(spins, self.state(spins))
