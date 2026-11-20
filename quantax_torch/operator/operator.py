from collections import defaultdict
from numbers import Number

import numpy as np
import torch
from scipy.linalg import eigh

from ..global_defs import get_default_dtype, get_sites
from ..sampler import Samples
from ..state import DenseState
from ..symmetry import Identity
from ..utils.spins import array_to_ints


class Operator:
    """Sum of local operator strings in the checkout's Pauli convention."""

    def __init__(self, op_list):
        self.sites = get_sites()
        self.dtype = get_default_dtype()
        self.op_list = [[chars, [list(row) for row in terms]] for chars, terms in op_list if terms]
        self._compiled = {}
        self._quspin_ops = {}

    def _groups(self, device):
        if device not in self._compiled:
            self._compiled[device] = [
                (chars.replace("i", "I"), torch.tensor([t[0] for t in terms], device=device, dtype=self.dtype),
                 torch.tensor([t[1:] for t in terms], device=device, dtype=torch.long))
                for chars, terms in self.op_list
            ]
        return self._compiled[device]

    @property
    def expression(self):
        return " ".join(f"{coef:+} {chars}{tuple(sites)}"
                        for chars, terms in self.op_list for coef, *sites in terms)

    def __repr__(self):
        return self.expression

    def _derived(self, terms):
        operator = Operator(terms)
        operator.sites, operator.dtype = self.sites, self.dtype
        return operator

    def __add__(self, other):
        if isinstance(other, Number):
            return self if other == 0 else self + self._derived([["I", [[other, 0]]]])
        if not isinstance(other, Operator):
            return NotImplemented
        if self.sites is not other.sites:
            raise ValueError("Cannot combine operators on different sites.")
        groups = defaultdict(list)
        for chars, terms in self.op_list + other.op_list:
            groups[chars].extend(terms)
        return self._derived(groups.items())

    __radd__ = __add__

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if isinstance(other, Number):
            return self._derived([(chars, [[other * coeff, *idx] for coeff, *idx in terms])
                                  for chars, terms in self.op_list])
        if not isinstance(other, Operator):
            return NotImplemented
        if self.sites is not other.sites:
            raise ValueError("Cannot combine operators on different sites.")
        groups = defaultdict(list)
        for left, left_terms in self.op_list:
            for right, right_terms in other.op_list:
                groups[left + right].extend([a * b, *i, *j] for a, *i in left_terms for b, *j in right_terms)
        return self._derived(groups.items())

    __rmul__ = __mul__

    def __truediv__(self, value):
        return self * (1 / value)

    def apply_diag(self, spins):
        spins = torch.as_tensor(spins)
        result = torch.zeros(len(spins), dtype=self.dtype, device=spins.device)
        for chars, coeff, idx in self._groups(spins.device):
            if set(chars) <= {"I", "z"}:
                for column, char in enumerate(chars):
                    if char == "z":
                        coeff = coeff * spins[:, idx[:, column]]
                result += coeff.expand(len(spins), -1).sum(dim=-1)
        return result

    def _connected(self, spins, row_elements=False, term_batch=128):
        """Yield source sample indices, connected spins, and nonzero matrix elements."""
        for chars, coefficients, sites in self._groups(spins.device):
            if set(chars) <= {"I", "z"}:
                continue
            for coeff, idx in zip(coefficients.split(term_batch), sites.split(term_batch)):
                values = coeff.expand(len(spins), -1)
                # Track flips per site so operator products can revisit the same site.
                flips = []
                columns = range(len(chars)) if row_elements else reversed(range(len(chars)))
                for column in columns:
                    char = chars[column]
                    site = idx[:, column]
                    spin = spins[:, site]
                    for flipped in flips:
                        spin = torch.where(site == flipped, -spin, spin)
                    if char == "z":
                        values = values * spin
                    elif char == "y":
                        values = values * ((-1j if row_elements else 1j) * spin)
                    elif char in ("+", "-"):
                        required = 1 if (char == "+") == row_elements else -1
                        values = values * (spin == required)
                    elif char not in ("I", "x"):
                        raise ValueError(f"Unsupported spin operator {char!r}.")
                    if char in "xy+-":
                        flips.append(site)
                sample, term = torch.nonzero(values, as_tuple=True)
                connected = spins[sample]
                rows = torch.arange(len(sample), device=spins.device)
                for site in flips:
                    connected[rows, site[term]] *= -1
                yield sample, connected, values[sample, term]

    def apply_off_diag(self, spins, return_basis_ints=False):
        spins = torch.as_tensor(spins)
        chunks = list(self._connected(spins))
        if chunks:
            sample, connected, values = map(torch.cat, zip(*chunks))
        else:
            sample = torch.empty(0, device=spins.device, dtype=torch.long)
            connected = spins[:0]
            values = torch.empty(0, device=spins.device, dtype=self.dtype)
        if return_basis_ints:
            connected = array_to_ints(connected)
        return sample, connected, values

    @torch.no_grad()
    def psiOloc(self, state, samples):
        if isinstance(samples, Samples):
            spins, psi = samples.spins, samples.wave_function
        else:
            spins = torch.as_tensor(samples, device=state.device)
            psi = state(spins)
        result = self.apply_diag(spins) * psi
        has_connections = False
        for sample, connected, values in self._connected(spins, row_elements=True):
            if len(sample):
                has_connections = True
                contribution = state(connected) * values
                result.index_add_(0, sample, contribution.to(result.dtype))
        if has_connections and getattr(state, "max_parallel", None) is not None:
            # The JAX connected-state batch includes all-up padding in its scale estimate.
            state(torch.ones_like(spins[:1]))
        return result

    def Oloc(self, state, samples):
        if not isinstance(samples, Samples):
            spins = torch.as_tensor(samples, device=state.device)
            samples = Samples(spins, state(spins))
        return self.psiOloc(state, samples) / samples.wave_function

    def expectation(self, state, samples, return_var=False):
        weights = samples.reweight_factor if isinstance(samples, Samples) else 1.0
        local = self.Oloc(state, samples)
        energy = (weights * local).mean().real.item()
        if return_var:
            return energy, (weights * local.abs().square()).mean().real.item()
        return energy

    def get_quspin_op(self, symm=None):
        symm = Identity() if symm is None else symm
        if symm not in self._quspin_ops:
            basis = symm.basis
            from quspin.operators import quantum_LinearOperator

            basis.make()
            dtype = np.dtype(str(self.dtype).removeprefix("torch."))
            terms = [[chars.replace("i", "I"), values] for chars, values in self.op_list]
            self._quspin_ops[symm] = quantum_LinearOperator(terms, basis=basis, dtype=dtype)
        return self._quspin_ops[symm]

    def todense(self, symm=None):
        op = self.get_quspin_op(symm)
        return op.matmat(np.eye(op.shape[1], dtype=op.dtype))

    def __array__(self, dtype=None):
        return np.asarray(self.todense(), dtype=dtype)

    def diagonalize(self, symm=None, k=1):
        op = self.get_quspin_op(symm)
        if k == "full":
            return eigh(self.todense(symm))
        if k >= op.shape[0]:
            values, vectors = eigh(self.todense(symm))
            return values[:k], vectors[:, :k]
        return op.eigsh(k=k, which="SA")

    def __matmul__(self, state):
        dense = state.todense()
        op = self.get_quspin_op(dense.symm)
        psi = dense.wave_function.detach().cpu().numpy()
        return DenseState(torch.as_tensor(op.matvec(psi), device=state.device), dense.symm)

    __rmatmul__ = __matmul__
