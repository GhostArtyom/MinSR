from __future__ import annotations
from copy import deepcopy
from typing import Sequence, Union
from numbers import Number
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from . import Operator, sigma_x, sigma_p, sigma_m, sigma_z
from ..global_defs import get_default_dtype, get_sites
from ..sampler import Samples
from ..state import Variational


@eqx.filter_jit
def _heisenberg_local(forward, ansatz, spins, wf, bonds, zz, exchange, batch_size):
    i, j = bonds.T
    product = spins[:, i] * spins[:, j]
    diagonal = jnp.sum(zz * product, axis=1) * wf
    active = (product == -1) & (jnp.abs(exchange) > 1e-8)
    count = jnp.count_nonzero(active)
    # Compact the exchanges on device. The extra entries allow a fixed-size
    # final window without copying the active count to the host.
    indices = jnp.nonzero(active.ravel(), size=active.size + batch_size)[0]
    lanes = jnp.arange(batch_size)
    nbonds = bonds.shape[0]

    def step(carry):
        offset, result, maximum = carry
        selected = jax.lax.dynamic_slice_in_dim(indices, offset, batch_size)
        sample, bond = selected // nbonds, selected % nbonds
        connected = spins[sample]
        left, right = i[bond], j[bond]
        connected = connected.at[lanes, left].set(spins[sample, right])
        connected = connected.at[lanes, right].set(spins[sample, left])
        valid = (offset + lanes) < count
        psi = forward(ansatz, jnp.where(valid[:, None], connected, 1))
        values = jnp.where(valid, exchange[bond] * psi, 0)
        result = result.at[sample].add(values)
        maximum = jnp.maximum(maximum, jnp.max(jnp.abs(psi)))
        return offset + batch_size, result, maximum

    # Match the generic path's all-up padding, used for wavefunction rescaling.
    _, result, maximum = jax.lax.while_loop(
        lambda carry: (count > 0) & (carry[0] <= count),
        step, (0, diagonal, jnp.zeros((), dtype=wf.real.dtype)),
    )
    return result, maximum, count


class _DeviceHeisenberg(Operator):
    def __init__(self, op_list, neighbors, couplings, signs):
        super().__init__(op_list)
        self._bonds = jnp.asarray(np.concatenate(neighbors, axis=0), dtype=jnp.int32)
        self._zz = jnp.asarray([
            J for pairs, J in zip(neighbors, couplings) for _ in range(len(pairs))
        ], dtype=get_default_dtype())
        self._exchange = jnp.asarray([
            2 * J * sign for pairs, J, sign in zip(neighbors, couplings, signs) for _ in range(len(pairs))
        ], dtype=get_default_dtype())

    @property
    def op_list(self):
        # Operator algebra edits its input lists. Keep the bond table and this
        # reference representation in sync when forming a new operator.
        return deepcopy(super().op_list)

    def __imul__(self, other):
        return self * other

    def psiOloc(self, state, samples):
        if (
            not isinstance(state, Variational) or jax.local_device_count() != 1
            or state.max_parallel is None or self._bonds.shape[0] == 0
        ):
            return super().psiOloc(state, samples)
        if isinstance(samples, Samples):
            spins, wf = samples.spins, samples.wave_function
        else:
            spins = jnp.asarray(samples)
            wf = state(spins)
        dtype = get_default_dtype()
        batch_size = state.max_parallel // state.nsymm
        result, maximum, count = _heisenberg_local(
            state._forward_batch, state.ansatz, spins, wf, self._bonds,
            self._zz.astype(dtype), self._exchange.astype(dtype), batch_size,
        )
        state._record_maximum(maximum)
        self._connectivity = count / spins.shape[0]
        return result


def Heisenberg(
    J: Union[Number, Sequence[Number]] = 1.0,
    n_neighbor: Union[int, Sequence[int]] = 1,
    msr: bool = False,
    *,
    device_local: bool = False,
) -> Operator:
    """Build a Pauli-scale Heisenberg operator, optionally evaluating local energies on device."""
    sites = get_sites()
    if isinstance(J, Number):
        J = [J]
    if isinstance(n_neighbor, Number):
        n_neighbor = [n_neighbor]
    if len(J) != len(n_neighbor):
        raise ValueError("The 'J' and 'n_neighbor' should have the same length.")
    neighbors = [np.asarray(pairs) for pairs in sites.get_neighbor(n_neighbor)]

    def hij(i, j, sign):
        hx = 2 * sign * (sigma_p(i) * sigma_m(j) + sigma_m(i) * sigma_p(j))
        hz = sigma_z(i) * sigma_z(j)
        return hx + hz

    signs = [-1 if msr and shell == 1 else 1 for shell in n_neighbor]
    H = 0
    for neighbors_i, coupling, sign in zip(neighbors, J, signs):
        H = H + coupling * sum(hij(i, j, sign) for i, j in neighbors_i)
    if device_local and not sites.is_fermion:
        return _DeviceHeisenberg(H.op_list, neighbors, J, signs)
    return H


def Ising(
    h: Number = 0.0,
    J: Number = 1.0,
) -> Operator:
    sites = get_sites()
    H = -h * sum(sigma_x(i) for i in range(sites.nsites))
    neighbors = sites.get_neighbor()
    H += -J * sum(sigma_z(i) * sigma_z(j) for i, j in neighbors)
    return H
