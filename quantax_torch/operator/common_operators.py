from numbers import Number

import torch

from .operator import Operator
from ..global_defs import get_sites
from ..sampler import Samples


class _Heisenberg(Operator):
    @torch.no_grad()
    def psiOloc(self, state, samples):
        if not isinstance(samples, Samples):
            spins = torch.as_tensor(samples, device=state.device)
            samples = Samples(spins, state(spins))
        else:
            spins = samples.spins
        psi = samples.wave_function

        # Both exchange strings connect the same configurations; exactly one
        # is nonzero on each antiparallel bond. Compact the full mask once.
        groups = {name: (coeff, idx) for name, coeff, idx in self._groups(spins.device)}
        if "+-" not in groups:
            return super().psiOloc(state, samples)
        coupling, bonds = groups["zz"]
        if len(spins) * len(bonds) > 2**22:
            # Do not turn an arbitrarily large local estimator into a single
            # dense sample-by-bond allocation.
            return super().psiOloc(state, samples)
        exchange, _ = groups["+-"]
        left, right = bonds.unbind(dim=1)
        a, b = spins[:, left], spins[:, right]
        result = ((a * b * coupling).sum(dim=1) * psi)
        rows, terms = torch.nonzero((a != b) & (exchange != 0), as_tuple=True)
        max_parallel = getattr(state, "max_parallel", None)
        chunk = 32768 if max_parallel is None else max_parallel // state.nsymm
        if spins.is_cuda and state.nsymm == 1:
            # A smaller convolution batch keeps the padded feature maps
            # smaller; max_parallel is a ceiling, not a required batch size.
            chunk = min(chunk, 2048)
        start_indices = torch.arange(min(chunk, rows.numel()), device=spins.device)
        for src, term in zip(rows.split(chunk), terms.split(chunk), strict=True):
            if not src.numel():
                break
            connected = spins[src]
            idx = start_indices[:src.numel()]
            connected[idx, left[term]] *= -1
            connected[idx, right[term]] *= -1
            contribution = state(connected) * exchange[term]
            result.index_add_(0, src, contribution.to(result.dtype))
        if rows.numel() and max_parallel is not None:
            # Keep the scale estimate's all-up padding from the generic path.
            state(torch.ones_like(spins[:1]))
        return result


def Heisenberg(J=1.0, n_neighbor=1, msr=False):
    J = [J] if isinstance(J, Number) else J
    shells = [n_neighbor] if isinstance(n_neighbor, int) else n_neighbor
    if len(J) != len(shells):
        raise ValueError("J and n_neighbor must have the same length.")
    neighbors = get_sites().get_neighbor(shells)
    diag = []
    exchange = []
    for shell, coupling, bonds in zip(shells, J, neighbors):
        sign = -1 if msr and shell == 1 else 1
        diag.extend([coupling, i, j] for i, j in bonds)
        exchange.extend([2 * sign * coupling, i, j] for i, j in bonds)
    cls = _Heisenberg if all(i != j for _, i, j in diag) else Operator
    return cls([["zz", diag], ["+-", exchange], ["-+", exchange]])


def Ising(h=0.0, J=1.0):
    sites = get_sites()
    return Operator([
        ["zz", [[-J, i, j] for i, j in sites.get_neighbor()]],
        ["x", [[-h, i] for i in range(sites.nsites)]],
    ])
