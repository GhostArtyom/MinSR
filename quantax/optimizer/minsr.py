import logging
import re
from typing import Optional, Callable, Tuple, List, Any
from functools import partial
from operator import index
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from .tdvp import TDVP
from .solver import minsr_pinv_eig
from ..state import Variational
from ..nn import Sequential, filter_vjp
from ..sampler import Samples
from ..operator import Operator
from ..utils import tree_fully_flatten, to_array_shard, array_extend
from ..global_defs import is_params_cpl


def _memory_headroom(devices):
    try:
        stats = [device.memory_stats() or {} for device in devices]
    except RuntimeError:
        return None
    if not stats or any("bytes_limit" not in item or "bytes_in_use" not in item for item in stats):
        return None
    available = min(item["bytes_limit"] - item["bytes_in_use"] for item in stats)
    return available, min(item["bytes_limit"] for item in stats)


def _retry_batch(batch_size, message, devices):
    request = re.search(r"allocate (\d+) bytes", message)
    headroom = _memory_headroom(devices) if request else None
    if headroom is not None:
        available, _ = headroom
        requested = int(request[1])
        if 0 < available < requested:
            # Estimate how many samples fit with 20% headroom. The
            # failed allocation may also include fixed-size buffers.
            return max(1, 4 * batch_size * available // (5 * requested))
    return batch_size // 2


class MinSR(TDVP):
    """MinSR with optional per-sample batching of logarithmic derivatives.

    ``jacobian_batch`` limits the number of sampled configurations in a
    derivative batch. Centering and the dense solve still use all samples.
    On a single GPU, symmetry-projected states can start with a batch
    estimated from available memory. A memory failure can lower the batch.
    Gram construction and the solve use double precision.
    """

    def __init__(
        self,
        state: Variational,
        hamiltonian: Operator,
        solver: Optional[Callable] = None,
        *,
        jacobian_batch: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):

        if solver is None:
            solver = minsr_pinv_eig()
        super().__init__(state, hamiltonian, imag_time=True, solver=solver)

        if self.vs_type == 1:
            raise ValueError("'MinSR' optimizer doesn't support vs_type 1")

        complex_derivatives = (self.vs_type == 2 or is_params_cpl()
                               or np.issubdtype(self.state.dtype, np.complexfloating))
        self._derivative_dtype = np.dtype(np.complex128 if complex_derivatives else np.float64)
        if jacobian_batch is not None:
            jacobian_batch = index(jacobian_batch)
            if jacobian_batch < 1:
                raise ValueError("jacobian_batch must be positive.")
        self._jacobian_batch = jacobian_batch
        self._logger = logger if logger is not None else logging.getLogger(__name__)

        self._nodes = []
        params, others = state.partition(state.ansatz)
        for params_i in params:
            nparams = tree_fully_flatten(params_i).size
            if nparams == 0:
                self._nodes.append("unparametrized")
            elif not isinstance(params_i, Sequential):
                self._nodes.append("not_sequential")
            else:
                idx_parametrized = []
                for i, layer in enumerate(params_i):
                    vals = tree_fully_flatten(layer)
                    if vals.size > 0:
                        idx_parametrized.append(i)

                block_len = np.sqrt(len(idx_parametrized)).astype(int)
                new_nodes = []
                for i, l in enumerate(idx_parametrized):
                    if i % block_len == block_len - 1 or i == len(idx_parametrized) - 1:
                        new_nodes.append(l)
                self._nodes.append(new_nodes)

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, None, 0))
    def _forward(
        self, ansatz: Tuple[eqx.Module, ...], spin: jax.Array
    ) -> Tuple[List[jax.Array], jax.Array]:
        inputs = self.state.input_fn(spin)
        neurons = []
        batch = []
        outputs = []
        remaining_layers = []

        for net, x, nodes in zip(ansatz, inputs, self._nodes):
            batch.append(x.shape[:-1])
            x = x.reshape(-1, x.shape[-1])
            new_neurons = [x]

            if not isinstance(nodes, list):
                neurons.append(new_neurons)
                outputs.append(jax.vmap(net)(x))
                remaining_layers.append(lambda x: x)
            else:
                for i, layer in enumerate(net):
                    x = jax.vmap(layer)(x)
                    if i == nodes[-1]:
                        if self.vs_type == 2 and jnp.iscomplexobj(x):
                            raise NotImplementedError(
                                "'MinSR' optimizer only supports vs_type 2 with "
                                "complex neurons in the last few unparametrized layers."
                            )
                        outputs.append(x)
                        break
                    if i in nodes:
                        new_neurons.append(x)
                neurons.append(new_neurons)
                remains = jax.vmap(net[i + 1 :]) if i < len(net) - 1 else lambda x: x
                remaining_layers.append(remains)

        def output_fn(x: List[jax.Array]) -> jax.Array:
            x = [l(xi).reshape(b) for l, xi, b in zip(remaining_layers, x, batch)]
            psi = self.state.output_fn(x)
            return psi / jax.lax.stop_gradient(psi)

        if self.vs_type == 0:
            jac_out = jax.grad(output_fn, holomorphic=self.state.holomorphic)
            deltas = jac_out(outputs)
        else:
            jac_real = jax.grad(lambda x: output_fn(x).real)(outputs)
            jac_imag = jax.grad(lambda x: output_fn(x).imag)(outputs)
            deltas = [re + 1j * im for re, im in zip(jac_real, jac_imag)]
        return neurons, deltas

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, None, 0, 0))
    def _layer_backward(
        self, layers: eqx.Module, neuron: jax.Array, delta: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        forward = lambda net, x: net(x)

        @partial(jax.vmap, in_axes=(None, 0, 0))
        def backward(net, x, delta):
            f_vjp = filter_vjp(forward, net, x)[1]
            vjp_vals, delta = f_vjp(delta)
            return tree_fully_flatten(vjp_vals), delta

        if self.vs_type == 0:
            grad, delta = backward(layers, neuron, delta)
        else:
            gr, delta_real = backward(layers, neuron, delta.real)
            gi, delta_imag = backward(layers, neuron, delta.imag)
            grad = gr + 1j * gi
            delta = delta_real + 1j * delta_imag if delta_real is not None else None
        grad = jnp.sum(grad.astype(self._derivative_dtype), axis=0)
        return grad, delta

    @eqx.filter_jit
    def _reversed_scan_layers(
        self,
        ansatz: Tuple[eqx.Module, ...],
        spins: jax.Array,
        fn_on_jac: Callable,
        init_vals: Any,
    ):
        neurons, deltas = self._forward(ansatz, spins)

        for net, neurons_i, delta, nodes in zip(
            reversed(ansatz), reversed(neurons), reversed(deltas), reversed(self._nodes)
        ):
            if nodes == "unparametrized":
                continue
            elif nodes == "not_sequential":
                jac, delta = self._layer_backward(net, neurons_i[0], delta)
                if jac.size > 0:
                    init_vals = fn_on_jac(jac, init_vals)
            else:
                end = nodes[-1] + 1
                nodes = [0] + [n + 1 for n in nodes[:-1]]
                for start, neuron in zip(reversed(nodes), reversed(neurons_i)):
                    layers = net[start:end]
                    end = start
                    jac, delta = self._layer_backward(layers, neuron, delta)
                    if jac.size > 0:
                        init_vals = fn_on_jac(jac, init_vals)

        return init_vals

    @eqx.filter_jit
    def _batched_jacobians(self, ansatz, spins, batch_size):
        nsamples, nsites = spins.shape
        nbatches = (nsamples + batch_size - 1) // batch_size
        batch_size = (nsamples + nbatches - 1) // nbatches
        padding = nbatches * batch_size - nsamples
        if padding:
            # Duplicate a valid configuration and discard its derivative
            # before any sample means or weighted reductions are computed.
            spins = jnp.pad(spins, ((0, padding), (0, 0)), mode="edge")

        def collect(jac, blocks):
            return (*blocks, jac)

        def differentiate(batch):
            return self._reversed_scan_layers(ansatz, batch, collect, ())

        batches = spins.reshape(nbatches, batch_size, nsites)
        blocks = jax.lax.map(differentiate, batches)
        return tuple(block.reshape(-1, block.shape[-1])[:nsamples] for block in blocks)

    def _scan_layers(self, spins, fn_on_jac, init_vals):
        ansatz = self.state.ansatz
        batch_size = self._jacobian_batch
        if batch_size is None or batch_size >= spins.shape[0]:
            return self._reversed_scan_layers(ansatz, spins, fn_on_jac, init_vals)

        blocks = list(reversed(self._batched_jacobians(ansatz, spins, batch_size)))
        while blocks:
            init_vals = fn_on_jac(blocks.pop(), init_vals)
        return init_vals

    @partial(jax.jit, static_argnums=0, donate_argnums=1)
    def _get_Obar(self, Omat: jax.Array, reweight: jax.Array) -> jax.Array:
        # should be pmean here
        reweight = reweight.astype(Omat.real.dtype)
        Omat -= jnp.mean(Omat, axis=0, keepdims=True)
        Omat *= jnp.sqrt(reweight[:, None] / Omat.shape[0])
        Omat = array_extend(Omat, jax.local_device_count(), axis=1)
        Omat = to_array_shard(Omat, sharded_axis=1)
        if self.vs_type != 0:
            Omat = jnp.concatenate([Omat.real, Omat.imag], axis=0)
        return Omat
    
    @partial(jax.jit, static_argnums=0, donate_argnums=1)
    def _Tmat_scan_fn(
        self, jac: jax.Array, vals: Tuple[jax.Array, List, jax.Array]
    ) -> Tuple[jax.Array, List, jax.Array]:
        Tmat, Omean, reweight = vals
        Omean.append(jnp.mean(jac * reweight[:, None], axis=0))
        Obar = self._get_Obar(jac, reweight)
        Tmat += Obar @ Obar.conj().T
        return Tmat, Omean, reweight

    def _gram_shape_dtype(self, nsamples):
        if not jax.config.x64_enabled:
            raise ValueError("MinSR requires jax_enable_x64 for double-precision Gram computation.")
        if self.vs_type != 0:
            nsamples *= 2
        dtype = np.dtype(np.float64) if self.vs_type == 2 else self._derivative_dtype
        return (nsamples, nsamples), dtype

    def _initial_derivative_batch(self, nsamples, devices):
        if len(devices) != 1 or self.state.nsymm == 1 or not self.state.nparams:
            return None
        headroom = _memory_headroom(devices)
        if headroom is None:
            return None
        available, limit = headroom
        (rows, cols), dtype = self._gram_shape_dtype(nsamples)
        derivative_bytes = self.state.nparams * self._derivative_dtype.itemsize
        per_projected_sample = derivative_bytes * self.state.nsymm
        # Leave space for the dense Gram and eigensolver, retained derivatives,
        # centering temporaries, and other live GPU allocations.
        fixed = 3 * rows * cols * dtype.itemsize + 2 * nsamples * derivative_bytes
        budget = max(0, available - fixed - limit // 10)
        if nsamples * per_projected_sample <= budget:
            return None
        # Keep additional workspace for the projected reverse pass once we batch.
        return max(1, budget // (3 * per_projected_sample))

    def get_Tmat(self, samples: Samples) -> jax.Array:
        shape, dtype = self._gram_shape_dtype(samples.nsamples)
        return self._fill_Tmat(samples, jnp.zeros(shape, dtype=dtype))

    def _fill_Tmat(self, samples: Samples, Tmat: jax.Array) -> jax.Array:
        Omean = []
        init_vals = (Tmat, Omean, samples.reweight_factor)
        Tmat, Omean, reweight = self._scan_layers(samples.spins, self._Tmat_scan_fn, init_vals)
        self._Omean = jnp.concatenate(Omean[::-1])
        return Tmat
    
    @partial(jax.jit, static_argnums=0, donate_argnums=1)
    def _Ohvp_scan_fn(
        self, jac: jax.Array, vals: Tuple[List, jax.Array, jax.Array]
    ) -> Tuple[List, jax.Array, jax.Array]:
        outputs, reweight, vec = vals
        nparams = jac.shape[1]
        Obar = self._get_Obar(jac, reweight)
        Ohvp = (Obar.conj().T @ vec)[:nparams]
        outputs.append(Ohvp)
        return outputs, reweight, vec

    def Ohvp(self, samples: Samples, vec: jax.Array) -> jax.Array:
        """
        Compute O^dag @ vec. vec @ jac is used instead of vjp for better precision.
        """
        init_vals = ([], samples.reweight_factor, vec)
        outputs, reweight, vec = self._scan_layers(samples.spins, self._Ohvp_scan_fn, init_vals)
        outputs = jnp.concatenate(outputs[::-1])
        return outputs

    def get_step(self, samples: Samples) -> jax.Array:
        Ebar = self.get_Ebar(samples)
        if isinstance(samples.spins, jax.core.Tracer):
            return self._step_from_energy(samples, Ebar)
        devices = samples.spins.sharding.device_set
        if jax.process_count() != 1 or not any(d.platform in ("gpu", "cuda", "rocm") for d in devices):
            return self._step_from_energy(samples, Ebar)

        # Surface local-energy errors outside the derivative fallback. JAX
        # may report asynchronous allocation errors when the result is ready.
        jax.block_until_ready(Ebar)
        initial_batch = self._jacobian_batch
        shape, dtype = self._gram_shape_dtype(samples.nsamples)
        try:
            if initial_batch is None:
                self._jacobian_batch = self._initial_derivative_batch(samples.nsamples, devices)
                if self._jacobian_batch is not None:
                    self._logger.info(f"MinSR derivative sample batch={self._jacobian_batch:d} (GPU memory estimate)")
            while True:
                # A smaller derivative batch cannot shrink the dense Gram.
                Tmat = jnp.zeros(shape, dtype=dtype).block_until_ready()
                try:
                    Tmat = self._fill_Tmat(samples, Tmat)
                    return jax.block_until_ready(self._finish_step(samples, Tmat, Ebar))
                except (RuntimeError, ValueError) as error:
                    message = str(error).lower()
                    oom = (
                        ("resource_exhausted" in message and "out of memory" in message)
                        or "cuda_error_out_of_memory" in message
                    )
                    current = min(self._jacobian_batch or samples.nsamples, samples.nsamples)
                    if not oom or current <= 1:
                        raise
                del Tmat
                self._jacobian_batch = _retry_batch(current, message, devices)
                self._logger.warning(
                    f"MinSR GPU out of memory. Retrying with derivative sample batch={self._jacobian_batch:d} "
                    f"(was {current:d})"
                )
        except BaseException:
            self._jacobian_batch = initial_batch
            raise

    def _step_from_energy(self, samples: Samples, Ebar: jax.Array) -> jax.Array:
        return self._finish_step(samples, self.get_Tmat(samples), Ebar)

    def _finish_step(self, samples: Samples, Tmat: jax.Array, Ebar: jax.Array) -> jax.Array:
        step = self.solve(samples, Tmat, Ebar)
        step = self.build_step(step, self._Omean, self._energy)
        return step

    def solve(self, samples: Samples, Tmat: jax.Array, Ebar: jax.Array) -> jax.Array:
        _, dtype = self._gram_shape_dtype(samples.nsamples)
        if Tmat.dtype != dtype:
            raise ValueError(f"MinSR Gram must be constructed in {dtype}, got {Tmat.dtype}.")
        if self.vs_type != 0:
            Ebar = jnp.concatenate([Ebar.real, Ebar.imag])
        Tinv_E = self._solver(Tmat, Ebar.astype(dtype))
        if isinstance(Tinv_E, tuple):
            Tinv_E, self._solver_info = Tinv_E
        step = self.Ohvp(samples, Tinv_E)
        return step
