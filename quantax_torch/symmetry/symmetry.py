import numpy as np
import torch

from ..global_defs import get_default_dtype, get_sites


def _characters(sector, order):
    if not isinstance(sector, (int, np.integer)):
        if len(sector) != order:
            raise ValueError("Explicit characters must match the generator order.")
        return np.asarray(sector)
    if not 0 <= sector < order:
        raise ValueError(f"Sector {sector} out of range for order {order}.")
    if sector == 0:
        return np.ones(order)
    if 2 * sector == order:
        return (-1.0) ** np.arange(order)
    if not get_default_dtype().is_complex:
        raise ValueError("Complex characters require a complex default dtype.")
    return np.exp(-2j * np.pi * sector * np.arange(order) / order)


class Symmetry:
    def __init__(self, generator=None, sector=0, spin_inversion=0, total_sz=None,
                 perm=None, eigval=None):
        self.sites = get_sites()
        self.nsites = self.sites.nsites
        identity = np.arange(self.nsites)
        self._generators = np.atleast_2d(identity if generator is None else generator)
        if any(not np.array_equal(np.sort(g), identity) for g in self._generators):
            raise ValueError("Each symmetry generator must permute the sites.")
        self._sectors = [sector] if len(self._generators) == 1 else list(sector)
        if spin_inversion not in (0, 1, -1):
            raise ValueError("Spin inversion must be 0, 1 or -1.")
        if total_sz is not None and (abs(total_sz) > self.nsites or (self.nsites + total_sz) % 2):
            raise ValueError("total_sz does not match the number of sites.")
        self.spin_inversion = spin_inversion
        self.total_sz = total_sz
        if perm is None:
            perm, eigval = identity[None], np.ones(1)
            for g, s in zip(self._generators, self._sectors, strict=True):
                powers = [identity]
                current = g
                while not np.array_equal(current, identity):
                    powers.append(current)
                    current = current[g]
                perm = perm[:, np.stack(powers)].reshape(-1, self.nsites)
                eigval = np.outer(eigval, _characters(s, len(powers))).ravel()
        self._perm = perm
        self._eigval = eigval
        self._device_maps = {}
        self._basis = None

    @property
    def eigval(self):
        return torch.as_tensor(self._eigval, dtype=get_default_dtype())

    @property
    def nsymm(self):
        return self._eigval.size * (2 if self.spin_inversion else 1)

    @property
    def Nup(self):
        return None if self.total_sz is None else (self.nsites + self.total_sz) // 2

    @property
    def basis(self):
        if self._basis is None:
            try:
                from quspin.basis import spin_basis_general
            except ModuleNotFoundError as error:
                if error.name != "quspin":
                    raise
                raise ModuleNotFoundError("Dense symmetry bases require the optional QuSpin wheel.") from error
            blocks = {}
            for i, (g, sector) in enumerate(zip(self._generators, self._sectors)):
                if not np.array_equal(g, np.arange(g.size)):
                    if not isinstance(sector, (int, np.integer)):
                        raise ValueError("QuSpin bases require integer symmetry sectors.")
                    blocks[f"block{i}"] = (g, sector)
            if self.spin_inversion:
                blocks["inversion"] = (-np.arange(self.nsites) - 1, int(self.spin_inversion == -1))
            self._basis = spin_basis_general(
                self.nsites, self.Nup, pauli=-1, make_basis=False, **blocks,
            )
        return self._basis

    def _on_device(self, device):
        if device not in self._device_maps:
            # The cache is populated outside the autodiff transforms.
            perm = torch.as_tensor(self._perm, dtype=torch.long, device=device)
            eigval = self._eigval
            if self.spin_inversion:
                eigval = np.concatenate([eigval, self.spin_inversion * eigval])
            weight = torch.as_tensor(eigval / eigval.size, device=device)
            self._device_maps[device] = perm, weight
        return self._device_maps[device]

    def get_symm_spins(self, spins):
        perm, _ = self._on_device(spins.device)
        orbit = spins[..., perm]
        return torch.cat([orbit, -orbit], dim=-2) if self.spin_inversion else orbit

    def symmetrize(self, inputs):
        _, weight = self._on_device(inputs.device)
        return (inputs * weight.to(inputs.dtype)).sum(dim=-1)

    def __add__(self, other):
        if self.sites is not other.sites:
            raise ValueError("Cannot combine symmetries on different lattices.")
        if self.spin_inversion and other.spin_inversion and self.spin_inversion != other.spin_inversion:
            raise ValueError("Cannot combine different spin inversion parities.")
        if self.total_sz is not None and other.total_sz is not None and self.total_sz != other.total_sz:
            raise ValueError("Cannot combine different magnetization sectors.")
        total_sz = self.total_sz if self.total_sz is not None else other.total_sz
        return Symmetry(
            np.concatenate([self._generators, other._generators]),
            [*self._sectors, *other._sectors],
            self.spin_inversion or other.spin_inversion,
            total_sz,
            self._perm[:, other._perm].reshape(-1, self.nsites),
            np.outer(self._eigval, other._eigval).ravel(),
        )

    def __call__(self, state):
        from ..state import DenseState, Variational

        if isinstance(state, DenseState):
            return state.todense(self)
        if not isinstance(state, Variational):
            raise TypeError("Expected a quantum state.")
        return Variational(
            state.ansatz, symm=state.symm + self, max_parallel=state.max_parallel,
            input_fn=lambda s: state.input_fn(self.get_symm_spins(s)),
            output_fn=lambda x: self.symmetrize(state.output_fn(x)),
        )
