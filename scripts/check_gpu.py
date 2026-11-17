"""Check JAX kernels and two small MinSR updates.

Use --jax-only to skip QuSpin/Quantax. Set JAX_PLATFORMS=cpu for a CPU check.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def elementwise():
    x = jax.device_put(np.arange(32, dtype=np.float32))
    y = jax.jit(lambda a: a * a + 1)(x)
    np.testing.assert_allclose(np.asarray(y.block_until_ready()), np.arange(32) ** 2 + 1)


def convolution():
    x = jax.device_put(np.ones((1, 1, 8, 8), np.float32))
    w = jax.device_put(np.ones((1, 1, 3, 3), np.float32))
    y = jax.jit(
        lambda a, b: jax.lax.conv_general_dilated(a, b, (1, 1), "VALID")
    )(x, w)
    np.testing.assert_allclose(np.asarray(y.block_until_ready()), 9)


def eigensolver():
    a = jax.device_put(np.array([[2., 1.], [1., 2.]], np.float64))
    vals = jax.jit(jnp.linalg.eigh)(a)[0]
    np.testing.assert_allclose(np.asarray(vals.block_until_ready()), [1., 3.], atol=1e-10)


def minsr_updates():
    import quspin
    import quantax as qtx

    print(f"QuSpin {quspin.__version__}", flush=True)
    qtx.set_random_seed(42)
    qtx.sites.Square(4)
    hamiltonian = qtx.operator.Heisenberg(J=[1, 0.5], n_neighbor=[1, 2], msr=True)
    net = qtx.nn.ResSum(2, 2, 3, use_sinh=True)
    state = qtx.state.Variational(net, symm=qtx.symmetry.TotalSz(), max_parallel=32)
    sampler = qtx.sampler.NeighborExchange(
        state, 16, reweight=1.0, n_neighbor=[1, 2], thermal_steps=2, sweep_steps=2
    )
    optimizer = qtx.optimizer.MinSR(state, hamiltonian)
    for i in range(2):
        samples = sampler.sweep()
        step = optimizer.get_step(samples)
        assert bool(jnp.all(jnp.isfinite(step))), "Non-finite MinSR update"
        state.update(step * 1e-3)
        print(f"MinSR step {i + 1}: energy={float(optimizer.energy):.8f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jax-only", action="store_true", help="Check JAX without importing QuSpin or Quantax.")
    args = parser.parse_args()
    print("Python:", sys.version.split()[0], "JAX:", jax.__version__, flush=True)
    print("Devices:", [(d.device_kind, d.platform) for d in jax.devices()], flush=True)
    results = {}
    checks = [
        ("jit_elementwise", elementwise),
        ("convolution", convolution),
        ("float64_eigh", eigensolver),
    ]
    for name, check in checks:
        try:
            check()
            results[name] = "PASS"
            print(name, "PASS", flush=True)
        except Exception as error:
            results[name] = str(error)
            print(name, "FAIL:", type(error).__name__, str(error), flush=True)
    if not args.jax_only:
        if all(value == "PASS" for value in results.values()):
            try:
                minsr_updates()
                results["minsr_updates"] = "PASS"
            except Exception as error:
                results["minsr_updates"] = str(error)
                print("minsr_updates FAIL:", type(error).__name__, str(error), flush=True)
        else:
            results["minsr_updates"] = "SKIPPED: JAX checks failed"
    print("RESULTS", json.dumps(results), flush=True)
    return 0 if all(value == "PASS" for value in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
