import copy
from pathlib import Path
from warnings import warn

import torch
from torch import nn
from torch.func import functional_call, vjp, vmap

from .state import State
from ..nn import Lambda, Sequential, Theta0Layer


def _evaluate(net, inputs, params=None, inference_fn=None):
    shape = inputs.shape[:-1]
    spins = inputs.reshape(-1, inputs.shape[-1])

    def forward(x):
        if params is not None:
            return functional_call(net, params, (x,))
        return net(x) if inference_fn is None else inference_fn(net, x)

    values = forward(spins) if isinstance(net, Sequential) else vmap(forward)(spins)
    return values.reshape(shape)


class Variational(State):
    """A variational state with an optional inference-only ``(net, inputs)`` callable."""

    def __init__(self, ansatz, param_file=None, symm=None, max_parallel=None,
                 input_fn=None, output_fn=None, *, device=None, inference_batch=None, inference_fn=None):
        super().__init__(symm)
        self.ansatz = (ansatz,) if isinstance(ansatz, nn.Module) else tuple(ansatz)
        if not self.ansatz or (len(self.ansatz) > 1 and (input_fn is None or output_fn is None)):
            raise ValueError("Multiple networks require an input function and an output function.")
        if max_parallel is not None and max_parallel < self.nsymm:
            raise ValueError("max_parallel must fit at least one symmetry orbit.")
        if inference_batch is not None and inference_batch < self.nsymm:
            raise ValueError("inference_batch must fit at least one symmetry orbit.")
        self.max_parallel = max_parallel
        self.inference_batch = inference_batch
        self.inference_fn = inference_fn
        self._default_input = input_fn is None
        self._default_output = output_fn is None
        self.input_fn = (lambda s: (self.symm.get_symm_spins(s),)) if input_fn is None else input_fn
        self.output_fn = (lambda x: self.symm.symmetrize(x[0])) if output_fn is None else output_fn
        if device is None:
            tensors = list(self.ansatz[0].parameters()) + list(self.ansatz[0].buffers())
            device = tensors[0].device if tensors else "cpu"
        self.device = torch.device(device)
        for net in self.ansatz:
            net.to(self.device)
        for_each = [net.holomorphic for net in self.ansatz if hasattr(net, "holomorphic")]
        self.holomorphic = bool(for_each) and all(for_each)
        self._maximum = torch.zeros((), device=self.device, dtype=torch.float64)
        self._vs_type = None
        self._params_copy = None
        if param_file is not None:
            self.load(param_file)

    def named_parameter_dicts(self):
        return tuple({key: value.detach() for key, value in net.named_parameters() if value.requires_grad}
                     for net in self.ansatz)

    def parameters(self):
        return (p for net in self.ansatz for p in net.parameters() if p.requires_grad)

    @property
    def nparams(self):
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def vs_type(self):
        if self._vs_type is None:
            with torch.no_grad():
                psi = self._raw(torch.ones((1, self.nsites), device=self.device, dtype=torch.int8))
            complex_params = any(p.is_complex() for p in self.parameters())
            if complex_params and not psi.is_complex():
                raise ValueError("Complex parameters require complex wave functions.")
            self._vs_type = (0 if self.holomorphic else 1) if complex_params else (2 if psi.is_complex() else 0)
        return self._vs_type

    def _inputs(self, spins):
        if self._default_input:
            return self.input_fn(spins)
        return vmap(self.input_fn)(spins)

    def _output(self, values):
        if self._default_output:
            return self.output_fn(values)
        return vmap(self.output_fn)(values)

    def _evaluate_spins(self, spins, inference_fn=None):
        values = tuple(_evaluate(net, x, inference_fn=inference_fn)
                       for net, x in zip(self.ansatz, self._inputs(spins), strict=True))
        return self._output(values)

    def _raw(self, spins):
        return self._evaluate_spins(spins)

    def _inference(self, spins):
        # Derivatives use _raw and the original modules, not this backend.
        return self._evaluate_spins(spins, self.inference_fn)

    @property
    def forward_batch(self):
        """Raw network inputs per inference batch, counting symmetry members."""
        return min((cap for cap in (self.max_parallel, self.inference_batch) if cap is not None), default=None)

    @torch.no_grad()
    def __call__(self, spins, ref_states=None):
        spins = torch.as_tensor(spins, device=self.device)
        shape = spins.shape[:-1]
        spins = spins.reshape(-1, self.nsites)
        if not len(spins):
            return torch.empty(shape, dtype=self.dtype, device=self.device)
        cap = self.forward_batch
        batch = len(spins) if cap is None else cap // self.nsymm
        evaluate = self._raw if self.inference_fn is None else self._inference
        psi = torch.cat([evaluate(s) for s in spins.split(batch)]).reshape(shape)
        self._record_maximum(psi)
        return psi

    def _record_maximum(self, psi):
        maximum = psi.abs().amax().to(self._maximum.dtype)
        self._maximum.copy_(torch.where(maximum.isfinite(), torch.maximum(maximum, self._maximum), self._maximum))

    @torch.no_grad()
    def jacobian(self, spins, *, chunk_size=16, logarithmic=True):
        params = self.named_parameter_dicts()
        spins = torch.as_tensor(spins, device=self.device)
        typ = self.vs_type
        complex_params = any(p.is_complex() for p in self.parameters())
        if not self.nparams:
            return torch.empty((len(spins), 0), dtype=self.dtype, device=self.device)

        def jac(inputs):
            def forward(weights):
                outputs = tuple(_evaluate(net, x, p) for net, x, p in zip(self.ansatz, inputs, weights, strict=True))
                psi = self.output_fn(outputs)
                return psi / psi.detach() if logarithmic else psi

            psi, pullback = vjp(forward, params)
            real = pullback(torch.ones_like(psi))[0]
            real = [grad.flatten() for tree in real for grad in tree.values()]
            if typ == 0:
                jacobian = torch.cat(real)
                return jacobian.conj() if complex_params else jacobian
            imag = pullback(torch.full_like(psi, 1j))[0]
            imag = [grad.flatten() for tree in imag for grad in tree.values()]
            if typ == 1:
                dx = [torch.complex(r.real, i.real) for r, i in zip(real, imag)]
                dy = [torch.complex(r.imag, i.imag) for r, i in zip(real, imag)]
                return torch.cat(dx + dy)
            return torch.cat([torch.complex(r, i) for r, i in zip(real, imag)])

        # Populate projection tensors before entering a transform.
        self._raw(spins[:1])
        inputs = self._inputs(spins)
        return vmap(jac, chunk_size=chunk_size)(inputs).to(self.dtype)

    def grad(self, spin):
        return self.jacobian(torch.as_tensor(spin).reshape(1, -1))[0]

    def get_params_flatten(self):
        params = [p.detach().flatten() for p in self.parameters()]
        return torch.cat(params) if params else torch.empty(0, device=self.device, dtype=self.dtype)

    def get_params_unflatten(self, flat):
        parameters = tuple(self.parameters())
        return tuple(x.reshape(p.shape) for x, p in zip(flat.split([p.numel() for p in parameters]), parameters))

    @torch.no_grad()
    def update_theta0(self, step0, rescale=True):
        if rescale:
            step0 = step0 + torch.where(self._maximum > 0, self._maximum.log(), 0.0)
        self._maximum.zero_()
        if not bool(torch.as_tensor(step0).isfinite()):
            warn("Invalid theta0 update.", stacklevel=2)
            return
        for net in self.ansatz:
            for layer in net.modules():
                if isinstance(layer, Theta0Layer):
                    step = torch.as_tensor(step0, device=self.device)
                    layer.theta0.sub_(step if layer.theta0.is_complex() else step.real)

    @torch.no_grad()
    def update(self, step, rescale=True):
        step = torch.as_tensor(step, device=self.device)
        if step.numel() != self.nparams + 1:
            raise ValueError("Invalid MinSR update size.")
        if not bool(step.isfinite().all()):
            warn("Invalid parameter update.", stacklevel=2)
            return
        self.update_theta0(step[0], rescale)
        params = tuple(self.parameters())
        if params and all(p.dtype == torch.float16 for p in params):
            if self._params_copy is None:
                self._params_copy = self.get_params_flatten().float()
            self._params_copy.sub_(step[1:].real)
            for p, value in zip(params, self.get_params_unflatten(self._params_copy)):
                p.copy_(value)
        else:
            for p, delta in zip(params, self.get_params_unflatten(step[1:])):
                p.sub_(delta if p.is_complex() else delta.real)

    def rescale(self):
        self.update_theta0(0)

    def save(self, file, *, metadata=None):
        file = Path(file)
        tmp = file.with_suffix(file.suffix + ".tmp")
        payload = {"version": 1, "state_dicts": [net.state_dict() for net in self.ansatz],
                   "metadata": metadata or {}}
        try:
            torch.save(payload, tmp)
            tmp.replace(file)
        finally:
            tmp.unlink(missing_ok=True)

    def load(self, file):
        from .checkpoint import load_checkpoint

        data = load_checkpoint(file)
        if len(data["state_dicts"]) != len(self.ansatz):
            raise ValueError("Checkpoint and state have different numbers of networks.")
        for net, weights in zip(self.ansatz, data["state_dicts"]):
            net.load_state_dict(weights, strict=True)
        self._params_copy = None
        return data["metadata"]

    def _combine(self, other, fn):
        if not isinstance(other, Variational):
            other = Functional(other, self.symm, device=self.device)
        index = len(self.ansatz)
        limits = [v for v in (self.max_parallel, other.max_parallel) if v is not None]
        return CombineVs(
            [self, other],
            lambda s: tuple(self.input_fn(s)) + tuple(other.input_fn(s)),
            lambda outputs: fn(self.output_fn(outputs[:index]), other.output_fn(outputs[index:])),
            max_parallel=min(limits, default=None),
        )

    def __add__(self, other):
        return self._combine(other, torch.add)

    __radd__ = __add__

    def __mul__(self, other):
        return self._combine(other, torch.mul)

    __rmul__ = __mul__


def Functional(fn, symm=None, *, device=None):
    if isinstance(fn, nn.Module):
        fn = copy.deepcopy(fn).requires_grad_(False)
        return Variational(fn, symm=symm, device=device)
    return Variational(Lambda(fn), symm=symm, device=device)


def CombineVs(states, input_fn, output_fn, *, max_parallel=None):
    ansatz = tuple(net for state in states for net in state.ansatz)
    if max_parallel is None:
        limits = [state.max_parallel for state in states if state.max_parallel is not None]
        max_parallel = min(limits, default=None)
    return Variational(ansatz, symm=states[0].symm, max_parallel=max_parallel,
                       input_fn=input_fn, output_fn=output_fn, device=states[0].device)
