from numbers import Number

from .operator import Operator
from ..global_defs import get_sites


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
    return Operator([["zz", diag], ["+-", exchange], ["-+", exchange]])


def Ising(h=0.0, J=1.0):
    sites = get_sites()
    return Operator([
        ["zz", [[-J, i, j] for i, j in sites.get_neighbor()]],
        ["x", [[-h, i] for i in range(sites.nsites)]],
    ])
