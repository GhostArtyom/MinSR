import numpy as np
import torch

from ..global_defs import get_default_dtype
from ..symmetry import Identity
from ..utils.spins import array_to_ints, ints_to_array


class State:
    def __init__(self, symm=None):
        self.symm = Identity() if symm is None else symm
        self.dtype = get_default_dtype()
        self.device = torch.device("cpu")

    @property
    def nsites(self):
        return self.symm.nsites

    @property
    def nsymm(self):
        return self.symm.nsymm

    @property
    def total_sz(self):
        return self.symm.total_sz

    @property
    def basis(self):
        return self.symm.basis

    def __getitem__(self, integers):
        return self(ints_to_array(integers, self.nsites))

    def __array__(self, dtype=None):
        return np.asarray(self.todense().wave_function.detach().cpu().numpy(), dtype=dtype)

    def todense(self, symm=None):
        symm = self.symm if symm is None else symm
        basis = symm.basis
        basis.make()
        wf = self[basis.states]
        factors = basis.get_amp(basis.states.copy())
        if not wf.is_complex():
            factors = factors.real
        return DenseState(wf / torch.as_tensor(factors, device=wf.device), symm)

    def norm(self, ord=None):
        wf = self.todense().wave_function
        return torch.linalg.vector_norm(wf, ord=2 if ord is None else ord).item()

    def _overlap(self, other, normalized):
        if not isinstance(other, State):
            return NotImplemented
        symm = self.symm if self.symm is other.symm else Identity()
        left = self.todense(symm).wave_function
        right = other.todense(symm).wave_function.to(left.device)
        if normalized:
            left = left / left.norm()
            right = right / right.norm()
        return (left.conj() * right).sum().item()

    def __matmul__(self, other):
        return self._overlap(other, False)

    def overlap(self, other):
        return self._overlap(other, True)


class DenseState(State):
    def __init__(self, wave_function, symm=None):
        super().__init__(symm)
        self.basis.make()
        self.wave_function = torch.as_tensor(wave_function).flatten()
        self.device = self.wave_function.device
        self.dtype = self.wave_function.dtype
        if self.wave_function.numel() != self.basis.states.size:
            raise ValueError("Wave function size does not match the symmetry basis.")

    def todense(self, symm=None):
        if symm is None or symm is self.symm:
            return self
        return super().todense(symm)

    def normalize(self):
        self.wave_function = self.wave_function / self.wave_function.norm()
        return self

    def __getitem__(self, integers):
        integers = np.asarray(integers, dtype=self.basis.dtype, order="C")
        representatives = integers.flatten()
        factors = self.basis.get_amp(representatives, mode="full_basis")
        if not self.wave_function.is_complex() and np.allclose(factors.imag, 0):
            factors = factors.real
        states = self.basis.states[::-1]
        indices = np.searchsorted(states, representatives)
        if not states.size:
            return torch.zeros(integers.shape, device=self.device, dtype=self.dtype)
        indices = np.minimum(indices, states.size - 1)
        indices, masks = states.size - 1 - indices, states[indices] == representatives
        values = self.wave_function[torch.as_tensor(indices, device=self.device)]
        values = values * torch.as_tensor(factors * masks, device=self.device)
        return values.reshape(integers.shape)

    def __call__(self, spins, ref_states=None):
        return self[array_to_ints(spins)].reshape(spins.shape[:-1])

    def __add__(self, other):
        if not isinstance(other, DenseState) or self.symm is not other.symm:
            return NotImplemented
        return DenseState(self.wave_function + other.wave_function, self.symm)

    def __sub__(self, other):
        return self + (-1) * other

    def __mul__(self, other):
        return DenseState(self.wave_function * other, self.symm)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return DenseState(self.wave_function / other, self.symm)

    def __rtruediv__(self, other):
        return DenseState(other / self.wave_function, self.symm)
