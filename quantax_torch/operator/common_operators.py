from numbers import Number

import torch

from .operator import Operator
from ..global_defs import get_sites
from ..sampler import Samples
from ..state import Variational


def _replay_exchange(accumulate, rows, terms, chunk, result, maximum, *, workspaces=None):
    """Replay four full inference batches at once for this local-energy call."""
    batches = 4
    count = batches * chunk
    src = rows[:count].clone().view(batches, chunk)
    term = terms[:count].clone().view_as(src)
    device = rows.device
    with torch.cuda.device(device):
        current = torch.cuda.current_stream(device)
        previous = None if workspaces is None else workspaces.get(current)
        stream = torch.cuda.Stream(device=device) if previous is None else previous[0]
        pool = None if previous is None else previous[1].pool()
        saved_maximum = maximum.clone()
        scratch = torch.zeros_like(result)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            # Warm up library kernels without including an extra batch in
            # either the estimator or the wave function scale estimate.
            accumulate(scratch, src[0], term[0])
            maximum.copy_(saved_maximum)
        current.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        # The convenience context synchronizes the device and empties the
        # allocator cache. That penalizes the MinSR allocations after Oloc
        # when a new graph is captured on every step.
        with torch.cuda.stream(stream):
            graph.capture_begin(pool=pool)
            try:
                for source, bond in zip(src.unbind(), term.unbind(), strict=True):
                    accumulate(result, source, bond)
            finally:
                graph.capture_end()
        end = rows.numel() // count * count
        for i in range(0, end, count):
            src.copy_(rows[i:i + count].view_as(src))
            term.copy_(terms[i:i + count].view_as(term))
            graph.replay()
        # Keep the pool alive until the next capture on this caller stream.
        # Replays are ordered and result was allocated outside the pool.
        if workspaces is not None:
            workspaces[current] = stream, graph
    return end


class _Heisenberg(Operator):
    def __init__(self, terms, *, local_energy_graph, reuse_local_energy_pool=False):
        super().__init__(terms)
        self.local_energy_graph = local_energy_graph
        self._local_energy_graph_used = False
        self._local_energy_workspaces = {} if reuse_local_energy_pool else None

    @torch.no_grad()
    def psiOloc(self, state, samples):
        self._local_energy_graph_used = False
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
        forward_batch = getattr(state, "forward_batch", max_parallel)
        chunk = 32768 if forward_batch is None else forward_batch // state.nsymm
        if spins.is_cuda and state.nsymm == 1:
            # A smaller convolution batch keeps the padded feature maps
            # smaller; max_parallel is a ceiling, not a required batch size.
            chunk = min(chunk, 2048)
        start_indices = torch.arange(min(chunk, rows.numel()), device=spins.device)

        def accumulate(out, src, term):
            connected = spins[src]
            idx = start_indices[:src.numel()]
            connected[idx, left[term]] *= -1
            connected[idx, right[term]] *= -1
            contribution = state(connected) * exchange[term]
            out.index_add_(0, src, contribution.to(out.dtype))

        start = 0
        if (self.local_energy_graph and spins.is_cuda and type(state) is Variational
                and state.nsymm > 1 and state.inference_fn is not None
                and state._default_input and state._default_output
                and rows.numel() >= 32 * chunk and not torch.are_deterministic_algorithms_enabled()):
            start = _replay_exchange(accumulate, rows, terms, chunk, result, state._maximum,
                                     workspaces=self._local_energy_workspaces)
            self._local_energy_graph_used = start > 0
        for src, term in zip(rows[start:].split(chunk), terms[start:].split(chunk), strict=True):
            if src.numel():
                accumulate(result, src, term)
        if rows.numel() and max_parallel is not None:
            # Keep the scale estimate's all-up padding from the generic path.
            state(torch.ones_like(spins[:1]))
        return result


def Heisenberg(J=1.0, n_neighbor=1, msr=False, *, local_energy_graph=False, reuse_local_energy_pool=False):
    """Build a Heisenberg operator; graph replay requires a capture-safe inference backend."""
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
    terms = [["zz", diag], ["+-", exchange], ["-+", exchange]]
    if all(i != j for _, i, j in diag):
        return _Heisenberg(terms, local_energy_graph=local_energy_graph, reuse_local_energy_pool=reuse_local_energy_pool)
    return Operator(terms)


def Ising(h=0.0, J=1.0):
    sites = get_sites()
    return Operator([
        ["zz", [[-J, i, j] for i, j in sites.get_neighbor()]],
        ["x", [[-h, i] for i in range(sites.nsites)]],
    ])
