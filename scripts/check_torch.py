"""Check PyTorch MinSR against a small reference or exercise the CUDA fast paths."""

import argparse
from pathlib import Path

import numpy as np


def configurations():
    rng = np.random.default_rng(42)
    positions = rng.random((8, 16)).argsort(axis=1)
    return (1 - 2 * (positions < 8)).astype(np.int8)


def export_jax(folder):
    import jax.numpy as jnp
    import quantax as qtx

    folder.mkdir(parents=True, exist_ok=True)
    qtx.set_random_seed(42)
    qtx.sites.Square(4)
    symm = qtx.symmetry.TotalSz() + qtx.symmetry.SpinInverse() + qtx.symmetry.C4v()
    net = qtx.nn.ResSum(4, 4, 3, use_sinh=True)
    state = qtx.state.Variational(net, symm=symm, max_parallel=64)
    hamiltonian = qtx.operator.Heisenberg(J=[1, 0.5], n_neighbor=[1, 2], msr=True)
    spins = jnp.asarray(configurations())
    wf = state(spins)
    samples = qtx.sampler.Samples(spins, wf, reweight=1.0)
    optimizer = qtx.optimizer.MinSR(state, hamiltonian)
    state.save(folder / "reference.eqx")
    np.savez(
        folder / "reference.npz",
        spins=spins, wave_function=wf,
        orbits=symm.get_symm_spins(spins), weights=samples.reweight_factor,
        local_energy=hamiltonian.Oloc(state, samples), jacobian=state.jacobian(spins),
        T=optimizer.get_Tmat(samples), step=optimizer.get_step(samples),
    )


def check(reference):
    import sys

    import torch
    import quantax_torch as qtx

    qtx.set_random_seed(42)
    qtx.sites.Square(4)
    symm = qtx.symmetry.TotalSz() + qtx.symmetry.SpinInverse() + qtx.symmetry.C4v()
    net = qtx.nn.ResSum(4, 4, 3, use_sinh=True)
    if reference is not None:
        qtx.state.load_res_sum_eqx(reference / "reference.eqx", net)
    state = qtx.state.Variational(net, symm=symm, max_parallel=64)
    hamiltonian = qtx.operator.Heisenberg(J=[1, 0.5], n_neighbor=[1, 2], msr=True)
    spins = torch.as_tensor(configurations())
    samples = qtx.sampler.Samples(spins, state(spins), reweight=1.0)
    optimizer = qtx.optimizer.MinSR(state, hamiltonian, jacobian_batch=2)
    generic = qtx.operator.Operator(hamiltonian.op_list)
    torch.testing.assert_close(hamiltonian.Oloc(state, samples), generic.Oloc(state, samples), rtol=1e-11, atol=1e-11)
    jacobian = state.jacobian(spins, chunk_size=2)
    blocks = sorted(optimizer.jacobian_blocks(spins))
    block_jacobian = torch.cat([jac for _, jac in blocks], dim=1)
    torch.testing.assert_close(block_jacobian, jacobian, rtol=1e-5, atol=2e-6)
    T = optimizer.get_Tmat(samples)
    step = optimizer.get_step(samples)
    assert step.isfinite().all()
    if reference is not None:
        got = {
            "orbits": symm.get_symm_spins(spins),
            "wave_function": samples.wave_function,
            "weights": samples.reweight_factor,
            "local_energy": hamiltonian.Oloc(state, samples),
            "jacobian": jacobian,
            "T": T,
            "step": step,
        }
        with np.load(reference / "reference.npz", allow_pickle=False) as expected:
            for name, value in got.items():
                actual = value.detach().numpy()
                atol = 2e-6 if name == "step" else 1e-6
                rtol = 2e-5 if name == "step" else 1e-5
                np.testing.assert_allclose(actual, expected[name], rtol=rtol, atol=atol, err_msg=name)
                print(f"{name}: max absolute difference {np.max(np.abs(actual - expected[name])):.3g}")
    state.update(1e-3 * step)
    sampler = qtx.sampler.NeighborExchange(state, 8, n_neighbor=[1, 2], thermal_steps=1, sweep_steps=2)
    sampled = sampler.sweep()
    assert (sampled.spins.sum(dim=1) == 0).all()
    assert hamiltonian.Oloc(state, sampled).isfinite().all()
    assert "quspin" not in sys.modules
    print("CPU MinSR, exchange sampling and local energies passed without QuSpin.")


def check_cuda_fast_paths():
    import torch
    import quantax_torch as qtx

    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    qtx.set_random_seed(42)
    qtx.sites.Square(4)
    # Exercise even padding, residual broadcasting and skipping frozen parameters.
    net = qtx.nn.ResSum(8, 4, 2, use_sinh=True).to(memory_format=torch.channels_last)
    net[2].conv1.weight.requires_grad_(False)
    state = qtx.state.Variational(net, symm=qtx.symmetry.TotalSz(), device="cuda", max_parallel=64)
    hamiltonian = qtx.operator.Heisenberg(J=[1, 0.5], n_neighbor=[1, 2], msr=True)
    spins = torch.as_tensor(configurations(), device=state.device)
    samples = qtx.sampler.Samples(spins, state(spins), reweight=1.0)
    generic = qtx.operator.Operator(hamiltonian.op_list)
    # Different convolution batch shapes can round the FP32 front end differently.
    torch.testing.assert_close(hamiltonian.Oloc(state, samples), generic.Oloc(state, samples), rtol=1e-7, atol=1e-7)
    opt = qtx.optimizer.MinSR(state, hamiltonian, jacobian_batch=2)
    blocks = sorted(opt.jacobian_blocks(spins))
    actual = torch.cat([jac for _, jac in blocks], dim=1)
    torch.testing.assert_close(actual, state.jacobian(spins, chunk_size=2), rtol=2e-5, atol=2e-6)
    energy = opt.get_Ebar(samples)
    gram = opt.get_Tmat(samples)
    uncached = opt.build_step(opt.solve(samples, gram, energy), opt._Omean, opt.energy)
    step = opt.get_step(samples)
    torch.testing.assert_close(step, uncached, rtol=2e-5, atol=2e-6)

    def sampler(graph):
        gen = torch.Generator(device=state.device).manual_seed(123)
        return qtx.sampler.NeighborExchange(
            state, len(spins), n_neighbor=[1, 2], thermal_steps=3, sweep_steps=4,
            initial_spins=spins, generator=gen, cuda_graph=graph,
        )

    eager, captured = sampler(False), sampler(True)
    assert captured._cuda_graph is not None
    torch.testing.assert_close(captured.current_spins, eager.current_spins, rtol=0, atol=0)
    state.update(step * 1e-3)
    for _ in range(2):
        expected, actual = eager.sweep(), captured.sweep()
        torch.testing.assert_close(actual.spins, expected.spins, rtol=0, atol=0)
        torch.testing.assert_close(actual.wave_function, expected.wave_function)
    torch.testing.assert_close(captured.generator.get_state(), eager.generator.get_state(), rtol=0, atol=0)
    print("CUDA Heisenberg, batched/cached MinSR and graph/eager exchange checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--export-jax", type=Path)
    mode.add_argument("--reference", type=Path)
    mode.add_argument("--cuda-fast-paths", action="store_true")
    args = parser.parse_args()
    if args.export_jax is not None:
        export_jax(args.export_jax)
    elif args.cuda_fast_paths:
        check_cuda_fast_paths()
    else:
        check(args.reference)


if __name__ == "__main__":
    main()
