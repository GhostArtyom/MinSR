import numpy as np
import torch

from ..global_defs import get_generator, get_lattice, get_sites


def ints_to_array(basis_ints, N=None):
    from quspin.tools import misc

    N = get_sites().nsites if N is None else N
    return misc.ints_to_array(basis_ints, N).astype(np.int8) * 2 - 1


def array_to_ints(state_array):
    from quspin.tools import misc

    if isinstance(state_array, torch.Tensor):
        state_array = state_array.detach().cpu().numpy()
    return misc.array_to_ints((np.asarray(state_array) > 0).astype(np.uint8)).flatten()


def neel(bipartiteA=True):
    xyz = get_lattice().xyz_from_index
    spins = torch.from_numpy((1 - 2 * (xyz.sum(axis=1) % 2)).astype(np.int8))
    return spins if bipartiteA else -spins


def stripe(alternate_dim=1):
    xyz = get_lattice().xyz_from_index
    return torch.from_numpy((1 - 2 * (xyz[:, alternate_dim] % 2)).astype(np.int8))


def Sqz_factor(*q):
    sites = get_sites()
    factor = np.exp(-1j * (sites.coord @ q)) / (2 * np.sqrt(sites.nsites))
    if np.allclose(factor.imag, 0):
        factor = factor.real

    def evaluate(spin):
        coefficients = torch.as_tensor(factor, device=spin.device)
        return (coefficients * spin).sum(dim=-1)

    return evaluate


def rand_spins(ns, total_sz=None, *, nsites=None, device="cpu", generator=None):
    nsites = get_sites().nsites if nsites is None else nsites
    generator = get_generator(device) if generator is None else generator
    shape = (ns, nsites)
    if total_sz is None:
        return torch.randint(0, 2, shape, dtype=torch.int8, device=device, generator=generator) * 2 - 1
    if abs(total_sz) > nsites or (nsites - total_sz) % 2:
        raise ValueError("total_sz does not match the number of sites.")
    order = torch.rand(shape, device=device, generator=generator).argsort(dim=-1)
    return 1 - 2 * (order < (nsites - total_sz) // 2).to(torch.int8)
