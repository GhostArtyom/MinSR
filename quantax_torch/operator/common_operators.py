import sys
from numbers import Number

import torch

from .operator import Operator
from ..global_defs import get_sites
from ..sampler import Samples
from ..state import Variational
from ..symmetry import Symmetry


_MAX_EXCHANGE_MASK_ELEMENTS = 2**22


def _exchange_batch_candidates(state, base):
    if state.nsymm == 1:
        return tuple(sorted({max(1, base // 4), max(1, base // 2), base}))
    limit = min(4096, state.max_parallel or 4096) // state.nsymm
    return tuple(batch for batch in (base, 2 * base, 4 * base) if batch <= max(base, limit))


def _tune_exchange_batch(state, accumulate, configs, rows, terms, result, base, *, replay):
    orbit = state.nsymm
    minimum_batches = 32 if replay else 8
    candidates = [batch for batch in _exchange_batch_candidates(state, base) if len(rows) >= minimum_batches * batch]
    if len(candidates) < 2:
        return base
    width = state.ansatz[0].config["channels"]
    free, _ = torch.cuda.mem_get_info(configs.device)
    # Keep tuning allocations small relative to the free device memory.
    candidates = [batch for batch in candidates
                  if batch == base or 32 * batch * orbit * state.nsites * width <= free // 4]
    if len(candidates) < 2:
        return base
    multiple = 4 * max(candidates)
    count = min(len(rows), (256 if replay else 8) * base) // multiple * multiple
    rows, terms = rows[:count], terms[:count]
    current = torch.cuda.current_stream(configs.device)
    saved_maximum = state._maximum.clone()
    scratch = torch.zeros_like(result)
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    rates = {}
    try:
        for batch in candidates:
            scratch.zero_()
            if replay:
                probe = _LocalEnergyGraph(accumulate, configs, scratch, rows, terms, batch, state._maximum,
                                          current, None)
                start.record()
                probe.replay(configs, scratch, rows, terms)
            else:
                indices = torch.arange(batch, device=rows.device)
                accumulate(configs, scratch, rows[:batch], terms[:batch], indices)
                scratch.zero_()
                start.record()
                for offset in range(0, count, batch):
                    accumulate(configs, scratch, rows[offset:offset + batch], terms[offset:offset + batch], indices)
            stop.record()
            stop.synchronize()
            rates[batch] = start.elapsed_time(stop)
            if replay:
                del probe
    finally:
        state._maximum.copy_(saved_maximum)
    best = min(rates, key=rates.get)
    return base if rates[best] > 0.98 * rates[base] else best


def _periodic_graph_net(state):
    periodic = sys.modules.get("quantax_torch.nn._periodic_cuda")
    hooks = torch.nn.modules.module
    if (periodic is not None and state.inference_fn is periodic.res_sum_forward
            and type(state.symm) is Symmetry and len(state.ansatz) == 1
            and not torch.is_autocast_enabled("cuda")
            and not (hooks._global_forward_hooks or hooks._global_forward_pre_hooks)
            and periodic.supports_counted_res_sum(state.ansatz[0])):
        return state.ansatz[0]
    return None


class _LocalEnergyGraph:
    def __init__(self, accumulate, configs, result, rows, terms, chunk, maximum, current, previous):
        self.configs = configs.clone()
        self.result = torch.zeros_like(result)
        self.src = rows[:4 * chunk].clone().view(4, chunk)
        self.term = terms[:4 * chunk].clone().view_as(self.src)
        indices = torch.arange(chunk, device=rows.device)
        self.indices = indices
        self.stream = torch.cuda.Stream(device=rows.device) if previous is None else previous.stream
        pool = None if previous is None else previous.graph.pool()
        saved_maximum = maximum.clone()
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            # Warm up without including an extra batch in the estimator or scale.
            accumulate(self.configs, self.result, self.src[0], self.term[0], indices)
            maximum.copy_(saved_maximum)
        current.wait_stream(self.stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(self.stream):
            self.graph.capture_begin(pool=pool)
            try:
                for source, bond in zip(self.src.unbind(), self.term.unbind(), strict=True):
                    accumulate(self.configs, self.result, source, bond, indices)
            finally:
                self.graph.capture_end()

    def replay(self, configs, result, rows, terms):
        active = len(configs)
        self.configs[:active].copy_(configs)
        self.result[:active].copy_(result)
        count = self.src.numel()
        end = rows.numel() // count * count
        for offset in range(0, end, count):
            self.src.view(-1).copy_(rows[offset:offset + count])
            self.term.view(-1).copy_(terms[offset:offset + count])
            self.graph.replay()
        result.copy_(self.result[:active])
        return end


def _replay_exchange(accumulate, configs, rows, terms, chunk, result, state, bonds, *, workspaces=None, net=None):
    with torch.cuda.device(rows.device):
        current = torch.cuda.current_stream(rows.device)
        key = None
        previous = None
        if workspaces is not None:
            previous = workspaces.get(current)
            if net is not None:
                tensors = (state._maximum, *bonds, *state.symm._on_device(rows.device),
                           *net.parameters(), *net.buffers())
                key = (configs.shape[1:], configs.dtype, result.dtype, chunk, id(state), id(state.symm),
                       id(state.input_fn), id(state.output_fn), state.forward_batch, id(net),
                       torch.is_inference_mode_enabled(),
                       tuple((t.data_ptr(), t.shape, t.stride(), t.dtype) for t in tensors))
        if (previous is not None and key is not None and previous[0] == key
                and len(previous[1].configs) >= len(configs)):
            workspace = previous[1]
        else:
            workspace = _LocalEnergyGraph(accumulate, configs, result, rows, terms, chunk, state._maximum,
                                          current, None if previous is None else previous[1])
            if workspaces is not None:
                workspace.tensors = tensors if key is not None else ()
                workspaces[current] = key, workspace
        return workspace.replay(configs, result, rows, terms)


class _Heisenberg(Operator):
    def __init__(self, terms, *, local_energy_graph, reuse_local_energy_pool, tune_local_energy,
                 local_energy_forward_batch):
        super().__init__(terms)
        self.local_energy_graph = local_energy_graph
        self._local_energy_graph_used = False
        self._local_energy_workspaces = {} if reuse_local_energy_pool else None
        self._tune_local_energy = tune_local_energy and local_energy_forward_batch is None
        self._fixed_forward_batch = local_energy_forward_batch
        self._local_energy_batch = None
        self._batch_key = None

    @staticmethod
    def _local_energy_key(state, device):
        return device, id(state.ansatz[0]), state.nsymm, state.forward_batch, state.max_parallel

    def _set_local_energy_batch(self, state, device, batch):
        if not self._tune_local_energy or state.forward_batch is None or torch.are_deterministic_algorithms_enabled():
            return False
        base = state.forward_batch // state.nsymm
        if state.nsymm == 1:
            base = min(base, 2048)
        elif _periodic_graph_net(state) is None:
            return False
        if batch not in _exchange_batch_candidates(state, base):
            return False
        self._batch_key = self._local_energy_key(state, device)
        self._local_energy_batch = batch
        return True

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
        # is nonzero on each antiparallel bond.
        groups = {name: (coeff, idx) for name, coeff, idx in self._groups(spins.device)}
        if "+-" not in groups:
            return super().psiOloc(state, samples)
        coupling, bonds = groups["zz"]
        exchange, _ = groups["+-"]
        left, right = bonds.unbind(dim=1)
        max_parallel = getattr(state, "max_parallel", None)
        forward_batch = getattr(state, "forward_batch", max_parallel)
        chunk = 32768 if forward_batch is None else forward_batch // state.nsymm
        if spins.is_cuda and state.nsymm == 1:
            # A smaller convolution batch keeps the padded feature maps
            # smaller; max_parallel is a ceiling, not a required batch size.
            chunk = min(chunk, 2048)
        fixed = self._fixed_forward_batch
        if fixed is not None:
            if (type(state) is not Variational or fixed < state.nsymm or fixed % state.nsymm
                    or (max_parallel is not None and fixed > max_parallel)):
                raise ValueError("The local-energy batch must fit max_parallel and contain whole orbits.")
            chunk = fixed // state.nsymm
            self._local_energy_batch = chunk

        # Compact exchanges in sample blocks so a large sample count can still
        # use replay while the temporary sample-by-bond arrays stay bounded.
        sample_batch = max(1, _MAX_EXCHANGE_MASK_ELEMENTS // len(bonds))
        result = torch.empty_like(psi, dtype=torch.promote_types(psi.dtype, coupling.dtype))
        replay = (self.local_energy_graph and spins.is_cuda and type(state) is Variational
                  and state.nsymm > 1 and state.inference_fn is not None
                  and state._default_input and state._default_output
                  and not torch.are_deterministic_algorithms_enabled())
        net = _periodic_graph_net(state) if replay else None
        ordinary = (spins.is_cuda and state.nsymm == 1 and type(state) is Variational
                    and not torch.are_deterministic_algorithms_enabled())
        tune = (self._tune_local_energy and (ordinary or net is not None)
                and getattr(state.ansatz[0], "config", None) is not None)
        if tune:
            key = self._local_energy_key(state, spins.device)
            if self._batch_key != key:
                self._local_energy_batch = None
                self._batch_key = key
            base = chunk
            tune_after = (64 if replay else 8) * base
        has_exchange = False
        for configs, wf, out in zip(spins.split(sample_batch), psi.split(sample_batch),
                                    result.split(sample_batch), strict=True):
            a, b = configs[:, left], configs[:, right]
            out.copy_((a * b * coupling).sum(dim=1) * wf)
            rows, terms = torch.nonzero((a != b) & (exchange != 0), as_tuple=True)
            if not rows.numel():
                continue
            has_exchange = True

            def accumulate(configs, out, src, term, indices):
                connected = configs[src]
                idx = indices[:src.numel()]
                connected[idx, left[term]] *= -1
                connected[idx, right[term]] *= -1
                if fixed is not None or (tune and replay):
                    psi_connected = state._inference(connected)
                    state._record_maximum(psi_connected)
                else:
                    psi_connected = state(connected)
                contribution = psi_connected * exchange[term]
                out.index_add_(0, src, contribution.to(out.dtype))

            if tune:
                if self._local_energy_batch is None and rows.numel() >= tune_after:
                    try:
                        with torch.cuda.device(spins.device):
                            self._local_energy_batch = _tune_exchange_batch(
                                state, accumulate, configs, rows, terms, out, base, replay=replay
                            )
                    except torch.cuda.OutOfMemoryError:
                        self._local_energy_batch = base
                chunk = self._local_energy_batch or base
            start_indices = torch.arange(min(chunk, rows.numel()), device=spins.device)
            start = 0
            if replay and rows.numel() >= 32 * chunk:
                start = _replay_exchange(accumulate, configs, rows, terms, chunk, out, state, (bonds, exchange),
                                         workspaces=self._local_energy_workspaces, net=net)
                self._local_energy_graph_used = True
            for offset in range(start, rows.numel(), chunk):
                accumulate(configs, out, rows[offset:offset + chunk], terms[offset:offset + chunk], start_indices)
        if has_exchange and max_parallel is not None:
            # Keep the scale estimate's all-up padding from the generic path.
            state(torch.ones_like(spins[:1]))
        return result


def Heisenberg(J=1.0, n_neighbor=1, msr=False, *, local_energy_graph=False, reuse_local_energy_pool=False,
               tune_local_energy=False, local_energy_forward_batch=None):
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
        return _Heisenberg(terms, local_energy_graph=local_energy_graph, reuse_local_energy_pool=reuse_local_energy_pool,
                           tune_local_energy=tune_local_energy, local_energy_forward_batch=local_energy_forward_batch)
    if local_energy_forward_batch is not None:
        raise ValueError("A fixed local-energy batch requires bonds between distinct sites.")
    return Operator(terms)


def Ising(h=0.0, J=1.0):
    sites = get_sites()
    return Operator([
        ["zz", [[-J, i, j] for i, j in sites.get_neighbor()]],
        ["x", [[-h, i] for i in range(sites.nsites)]],
    ])
