"""Check PyTorch MinSR against a small reference or exercise the CUDA fast paths."""

import argparse
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

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


def _exchange_pair(state, spins, *, compact=False, reweight=1.0):
    import torch
    from quantax_torch.sampler import NeighborExchange

    return tuple(
        NeighborExchange(
            state, len(spins), reweight=reweight, n_neighbor=[1, 2], thermal_steps=3, sweep_steps=4,
            initial_spins=spins, generator=torch.Generator(device=state.device).manual_seed(123),
            cuda_graph=graph, skip_unchanged=compact,
        )
        for graph in (False, True)
    )


def _check_exchange_sweep(eager, captured):
    import torch

    maximum = captured.state._maximum
    maximum.zero_()
    expected = eager.sweep()
    expected_max = maximum.clone()
    maximum.zero_()
    actual = captured.sweep()
    torch.testing.assert_close(actual.spins, expected.spins, rtol=0, atol=0)
    torch.testing.assert_close(actual.wave_function, expected.wave_function, rtol=1e-10, atol=1e-13)
    torch.testing.assert_close(maximum, expected_max, rtol=1e-10, atol=1e-13)
    assert torch.equal(captured.generator.get_state(), eager.generator.get_state())
    return actual


def check_cuda_fast_paths():
    import torch
    import quantax_torch as qtx
    from examples_torch.run_pipeline import Config
    from examples_torch.workflows import _system
    from quantax_torch.nn._cuda_support import supports_tuned_kernels

    device = torch.device("cuda")
    tuned = supports_tuned_kernels(device)
    config = Config(L=4, depth=4, channels=19 if tuned else 4, kernel_size=2, max_parallel=64)
    hamiltonian, state = _system(config, False)
    # Even padding, the broadcast skip in the first block and a frozen weight.
    state.ansatz[0][1].conv1.weight.requires_grad_(False)
    spins = torch.as_tensor(configurations(), device=device)
    samples = qtx.sampler.Samples(spins, state(spins), reweight=1.0)
    generic = qtx.operator.Operator(hamiltonian.op_list)
    torch.testing.assert_close(hamiltonian.Oloc(state, samples), generic.Oloc(state, samples), rtol=1e-7, atol=1e-7)
    opt = qtx.optimizer.MinSR(state, hamiltonian, jacobian_batch=2)
    actual = torch.cat([jac for _, jac in sorted(opt.jacobian_blocks(spins))], dim=1)
    torch.testing.assert_close(actual, state.jacobian(spins, chunk_size=2), rtol=2e-5, atol=2e-6)

    if tuned:
        assert state.inference_fn is not None
        _, reference = _system(replace(config, ordinary_periodic_conv=False), False)
        torch.testing.assert_close(state(spins), reference(spins), rtol=2e-5, atol=1e-7)
        torch.testing.assert_close(hamiltonian.Oloc(state, spins), hamiltonian.Oloc(reference, spins),
                                   rtol=3e-5, atol=2e-5)
    eager, captured = _exchange_pair(state, spins)
    assert captured._cuda_graph is not None
    _check_exchange_sweep(eager, captured)
    state.update(opt.get_step(samples) * 1e-3)
    _check_exchange_sweep(eager, captured)
    print("CUDA local energy, ordinary Jacobian and exchange replay agree with their references.")

    check_derivative_cache()
    check_cuda_gram_syrk()
    if not tuned:
        print("Tuned periodic kernels skipped: require CUDA sm_120, Triton and a C compiler.")
        return
    check_periodic_forward()
    check_projected_local_energy_replay()
    check_projected_exchange_replay()
    check_projected_vjp()


def check_periodic_forward():
    import torch
    import quantax_torch as qtx
    from examples_torch.run_pipeline import Config
    from examples_torch.workflows import _system
    from quantax_torch.nn._periodic_cuda import _finish_res_sum

    # Baseline, even padding with a channel tail, wide channels, and a kernel
    # that wraps a small lattice more than once. Exercise both output activations.
    for channels, kernel, use_sinh in ((16, 3, True), (19, 2, True), (67, 1, True), (1, 11, False)):
        config = Config(L=4, depth=4, channels=channels, kernel_size=kernel, use_sinh=use_sinh,
                        projected_forward_batch=64)
        hamiltonian, state = _system(config, True)
        _, reference = _system(replace(config, projected_periodic_conv=False, projected_forward_batch=128), True)
        assert state.inference_fn is not None
        spins = qtx.utils.rand_spins(7, 0, device=state.device)  # One full batch and a tail.
        update = torch.full((state.nparams + 1,), 1e-3, device=state.device, dtype=state.dtype)
        for iteration in range(2):
            wf = state(spins)
            expected = reference(spins)
            torch.testing.assert_close(wf, expected, rtol=2e-5, atol=1e-7)
            actual_samples = qtx.sampler.Samples(spins, wf, reweight=1.0)
            ref_samples = qtx.sampler.Samples(spins, expected, reweight=1.0)
            torch.testing.assert_close(hamiltonian.Oloc(state, actual_samples), hamiltonian.Oloc(reference, ref_samples),
                                       rtol=2e-4, atol=5e-5)
            if iteration == 0:
                state.update(update, rescale=False)
                reference.update(update, rescale=False)

    # Nonuniform site weights detect an incorrect traversal of channels-last features.
    _, state = _system(Config(L=4, depth=4, channels=16), True)
    scale, activation, reduction = state.ansatz[0].layers[-3:]
    reduction.eigval.copy_(torch.linspace(-0.5, 1.0, reduction.eigval.numel(), device=state.device))
    features = torch.randn(3, 16, 4, 4, device=state.device).contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        torch.testing.assert_close(_finish_res_sum(features, (scale, activation, reduction)),
                                   reduction(activation(scale(features))), rtol=1e-12, atol=1e-14)
    print("Periodic inference agrees with cuDNN for wave functions and local energies.")


def check_projected_exchange_replay():
    import torch
    from examples_torch.run_pipeline import Config
    from examples_torch.workflows import _system

    config = Config(L=4, depth=6, channels=19, kernel_size=2, projected_forward_batch=64)
    _, state = _system(config, True)
    # Nineteen samples give a partial final chunk. Test both training and measurement weighting.
    spins = torch.as_tensor(configurations(), device=state.device).repeat(3, 1)[:19]
    for weight in (1.0, 2.0):
        eager, captured = _exchange_pair(state, spins, compact=True, reweight=weight)
        assert captured._compact_key is not None
        samples = _check_exchange_sweep(eager, captured)
        saved = samples.spins.clone(), samples.wave_function.clone()
        update = torch.linspace(-1e-4, 1e-4, state.nparams + 1, device=state.device, dtype=state.dtype)
        state.update(update, rescale=False)
        _check_exchange_sweep(eager, captured)
        torch.testing.assert_close((samples.spins, samples.wave_function), saved, rtol=0, atol=0)

    status = captured._status
    left, right = captured.neighbors.unbind(1)
    unchanged = status.spins[:, left] == status.spins[:, right]
    uniform = torch.ones_like(status.prob)
    # Zero old probabilities make the acceptance rule take the proposal even when uniform=1.
    status = replace(status, prob=torch.zeros_like(status.prob), propose_prob=torch.full_like(status.propose_prob, 0.5))
    forward, batch = captured._compact_backend()
    for mask in (unchanged, ~unchanged):
        assert mask.any(dim=1).all()
        positions = mask.long().argmax(dim=1)
        with torch.no_grad():
            state._maximum.zero_()
            expected = eager._advance(status, uniform, positions)
            expected_max = state._maximum.clone()
            state._maximum.zero_()
            actual = captured._advance_compact(status, uniform, positions, forward, batch)
        for field in ("spins", "wave_function", "prob", "propose_prob"):
            torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=1e-10, atol=1e-13)
        torch.testing.assert_close(state._maximum, expected_max, rtol=1e-10, atol=1e-13)

    conv = state.ansatz[0][1].conv1
    conv.weight = torch.nn.Parameter(conv.weight.detach() + 1e-3)
    _check_exchange_sweep(eager, captured)
    state.inference_batch = 96
    _check_exchange_sweep(eager, captured)
    state.inference_fn = None
    _check_exchange_sweep(eager, captured)
    assert captured._cuda_graph is None
    print("Compacted exchange replay preserves samples, RNG and the amplitude maximum, with eager fallback.")


def check_projected_local_energy_replay():
    import torch
    import quantax_torch as qtx
    from examples_torch.run_pipeline import Config
    from examples_torch.workflows import _system
    from quantax_torch.operator import common_operators

    # Batches of 3 connected configurations exercise replay and the short
    # eager tail, even though the test only uses a small lattice and state.
    config = Config(L=4, depth=4, channels=16, projected_forward_batch=48)
    fast, state = _system(config, True)
    reference, _ = _system(replace(config, projected_local_energy_graph=False), True)
    spins = qtx.utils.rand_spins(8, 0, device=state.device)
    for n in range(2):
        if n:
            update = torch.linspace(-0.003, 0.003, state.nparams + 1, device=state.device, dtype=state.dtype)
            state.update(update, rescale=False)
        samples = qtx.sampler.Samples(spins, state(spins), reweight=1.0)
        state._maximum.zero_()
        expected = reference.Oloc(state, samples)
        max_expected = state._maximum.clone()
        state._maximum.zero_()
        with patch.object(common_operators, "_replay_exchange", wraps=common_operators._replay_exchange) as replay:
            actual = fast.Oloc(state, samples)
            assert replay.called
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-11)
        torch.testing.assert_close(state._maximum, max_expected, rtol=0, atol=0)
    # Fully polarized spins have no exchanges and need no graph capture.
    polarized = torch.ones_like(spins[:1])
    torch.testing.assert_close(fast.Oloc(state, polarized), reference.Oloc(state, polarized), rtol=0, atol=0)
    print("Whole-batch local energy replay agrees with eager before and after a weight update.")


def check_projected_vjp():
    import torch
    import quantax_torch as qtx
    from examples_torch.run_pipeline import Config
    from examples_torch.workflows import _system
    from quantax_torch.optimizer import minsr

    batched = minsr.batched_res_sum_vjp

    def require_batched(*args):
        result = batched(*args)
        assert result is not None
        return result

    # Both cases split complete orbits, with a short tail. The wide case
    # crosses the reverse chunk limit without depending on the exact limit.
    for size, channels, kernel, n, cap in ((4, 19, 2, 8, 48), (6, 64, 3, 37, 1024)):
        config = Config(L=size, depth=4, channels=channels, kernel_size=kernel, max_parallel=cap)
        hamiltonian, state = _system(config, True)
        spins = qtx.utils.rand_spins(n, 0, device=state.device)
        opt = qtx.optimizer.MinSR(state, hamiltonian, jacobian_batch=2)
        samples = qtx.sampler.Samples(spins, state(spins), reweight=1.0)
        with patch.object(minsr, "batched_res_sum_vjp", return_value=None):
            reference = sorted(opt.jacobian_blocks(spins))
            if size == 4:
                reference_step = opt.get_step(samples)
        with patch.object(minsr, "batched_res_sum_vjp", require_batched):
            actual = sorted(opt.jacobian_blocks(spins))
            if size == 4:
                torch.testing.assert_close(opt.get_step(samples), reference_step, rtol=3e-4, atol=4e-6)
        for (_, got), (_, expected) in zip(actual, reference, strict=True):
            torch.testing.assert_close(got, expected, rtol=3e-5, atol=4e-6)
    print("Projected periodic VJP agrees with generic autodiff and the MinSR update.")


def check_derivative_cache():
    import torch
    import quantax_torch as qtx
    from examples_torch.run_pipeline import Config
    from examples_torch.workflows import _system
    from quantax_torch.optimizer import minsr

    hamiltonian, state = _system(Config(L=4, depth=6, channels=19), True)
    spins = torch.as_tensor(configurations(), device=state.device)
    weights = torch.linspace(0.1, 1.9, len(spins), device=state.device, dtype=state.dtype)
    samples = qtx.sampler.Samples(spins, state(spins), weights)
    opt = qtx.optimizer.MinSR(state, hamiltonian, jacobian_batch=2)
    blocks = []
    gram = opt.get_Tmat(samples, _blocks=blocks, _native_cache=True)
    torch.testing.assert_close(gram, opt.get_Tmat(samples), rtol=0, atol=0)
    # Make the small reference exercise mixed dense and triangular tiles.
    with patch.object(minsr, "_gram_tile", return_value=3):
        torch.testing.assert_close(opt.get_Tmat(samples), gram, rtol=1e-13, atol=1e-14)
    with patch.object(minsr, "_DERIVATIVE_CACHE_BYTES", 0):
        reference = opt.get_step(samples)
    native_bytes = sum(block.numel() * block.element_size() for block in blocks)
    for budget, plan in ((2 * native_bytes, ("normalized", None)),
                         (native_bytes, ("native", None)),
                         (blocks[-1].numel() * blocks[-1].element_size(), ("native", 1))):
        with patch.object(minsr, "_DERIVATIVE_CACHE_BYTES", budget):
            assert opt._cache_plan(samples, 8) == plan
            torch.testing.assert_close(opt.get_step(samples), reference, rtol=1e-11, atol=1e-11)
    print("Normalized, native and partial caches agree with a two-pass weighted MinSR update.")


def check_cuda_gram_syrk():
    import torch
    from quantax_torch.optimizer._cuda_gram import add_gram_lower_

    # A non-default stream detects dispatch on the wrong cuBLAS stream.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        block = torch.randn(71, 4133, device="cuda", dtype=torch.float64)
        original = torch.randn(71, 71, device="cuda", dtype=torch.float64)
        result = original.clone()
        assert not add_gram_lower_(result, block[:, ::2])
        if not add_gram_lower_(result, block):
            print("Native CUDA triangular Gram check skipped: cuBLAS path unavailable.")
            return
        expected = original + (block @ block.mT).tril()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(result, expected, rtol=2e-13, atol=1e-11)
    print("Native CUDA triangular Gram agrees with FP64 GEMM on a non-default stream.")


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
