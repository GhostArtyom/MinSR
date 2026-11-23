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
    batched = qtx.optimizer.MinSR(state, hamiltonian, jacobian_batch=3)
    batched_jacobian = torch.cat([jac for _, jac in sorted(batched.jacobian_blocks(spins))], dim=1)
    torch.testing.assert_close(batched_jacobian, block_jacobian, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(batched.get_Tmat(samples), T, rtol=1e-5, atol=1e-6)
    # The nearly singular Gram matrix amplifies small FP32 convolution differences.
    update_error = torch.linalg.vector_norm(batched.get_step(samples) - step)
    assert update_error <= 1e-5 * torch.linalg.vector_norm(step), update_error
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
    # JAX rounds the update to the parameter dtype before applying it.
    params = next(state.parameters())
    with torch.no_grad():
        params.view(-1)[0] = 1.0
    rounding_step = torch.zeros_like(step)
    rounding_step[1] = 0.50000003
    state.update(rounding_step, rescale=False)
    assert params.view(-1)[0] == 1.0 - rounding_step[1].to(params.dtype)
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
    from examples_torch.workflows import _periodic_forward_issue, _system

    device = torch.device("cuda")
    config = Config(L=4, depth=4, channels=19, kernel_size=2, max_parallel=64)
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

    if _periodic_forward_issue(config, device, False) is None:
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

    # The cache check compares independent FP32 backward passes exactly.
    deterministic = torch.backends.cudnn.deterministic
    torch.backends.cudnn.deterministic = True
    try:
        check_derivative_cache()
    finally:
        torch.backends.cudnn.deterministic = deterministic
    check_cuda_gram_syrk()
    check_cuda_gram_dispatch()
    check_cuda_eigh_cache()
    check_cuda_derivative_batches()
    issue = _periodic_forward_issue(Config(), device, True)
    if issue is not None:
        print(f"Periodic inference checks skipped: {issue}.")
        return
    check_periodic_dispatch()
    check_periodic_forward()
    check_projected_local_energy_replay()
    check_projected_exchange_replay()
    check_projected_vjp()


def check_periodic_dispatch():
    import torch
    from examples_torch.run_pipeline import Config
    from examples_torch import workflows
    from quantax_torch.nn import _cuda_support
    from quantax_torch.nn._periodic_cuda import res_sum_forward

    device = torch.device("cuda")
    config = Config(projected_forward_batch=2048, max_parallel=4096)
    for sm in ((8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)):
        with patch.object(_cuda_support, "cuda_capability", return_value=sm):
            for nsamples in (1, 1000, 10000, 20000):
                current = replace(config, nsamples=nsamples)
                assert workflows._cuda_inference(current, device, False) is res_sum_forward
                inference = workflows._cuda_inference(current, device, True)
                assert inference is res_sum_forward
                assert workflows._local_energy_graph_issue(current, inference) is None
    with patch.object(_cuda_support, "cuda_capability", return_value=(7, 5)):
        assert workflows._cuda_inference(config, device, True) is None
    with patch.object(_cuda_support, "triton_runtime_issue", return_value="Triton is unavailable"):
        assert workflows._cuda_inference(config, device, True) is None
    print("Periodic forward and local energy replay dispatch independently of sample count on supported CUDA targets.")


def check_cuda_gram_dispatch():
    import torch
    import quantax_torch as qtx
    from examples_torch.run_pipeline import Config
    from examples_torch.workflows import _system
    from quantax_torch.optimizer import minsr

    config = Config(nsamples=512)
    device = torch.device("cuda")
    hamiltonian, state = _system(config, False)
    spins = qtx.utils.rand_spins(config.nsamples, 0, device=device)
    samples = qtx.sampler.Samples(spins, state(spins), reweight=1.0)
    opt = minsr.MinSR(state, hamiltonian, jacobian_batch=config.jacobian_batch)
    with patch.object(minsr, "tune_gram", wraps=minsr.tune_gram) as tune:
        gram = opt.get_Tmat(samples)
        (key,) = opt._gram_backend_choices
        opt._gram_backend_choices[key] = "full GEMM"
        torch.testing.assert_close(opt.get_Tmat(samples), gram, rtol=1e-12, atol=1e-14)
        tune.assert_called_once()
        opt._gram_backend_choices[key] = "tiled GEMM"
        torch.testing.assert_close(opt.get_Tmat(samples), gram, rtol=1e-12, atol=1e-14)
    print("CUDA Gram dispatch tunes once and agrees with dense FP64 GEMM.")


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
    from quantax_torch.sampler import metropolis

    config = Config(L=4, depth=6, channels=19, kernel_size=2, projected_forward_batch=64)
    _, state = _system(config, True)
    # Nineteen samples give a partial final chunk. Test both training and measurement weighting.
    spins = torch.as_tensor(configurations(), device=state.device).repeat(3, 1)[:19]
    for weight, capacity in ((1.0, 32), (2.0, 2)):
        with patch.object(metropolis, "_COMPACT_GRAPH_BATCHES", capacity):
            eager, captured = _exchange_pair(state, spins, compact=True, reweight=weight)
            assert captured._compact_key is not None
            if capacity == 2:
                assert isinstance(captured._cuda_graph, metropolis._ChunkedExchangeGraph)
                assert set(captured._cuda_graph.graphs) == {8, 3}
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
    config = Config(L=4, depth=4, channels=16, nsamples=9, projected_forward_batch=48)
    fast, state = _system(config, True)
    reference, _ = _system(replace(config, projected_local_energy_graph=False), True)
    spins = qtx.utils.rand_spins(config.nsamples, 0, device=state.device)
    mask_limit = 4 * len(fast.op_list[0][1])
    prior = None
    for n in range(3):
        if n:
            spins = spins.roll(1, dims=1)
            update = torch.linspace(-0.003, 0.003, state.nparams + 1, device=state.device, dtype=state.dtype)
            state.update(update, rescale=False)
        if n == 2:
            conv = state.ansatz[0][1].conv1
            conv.weight = torch.nn.Parameter(conv.weight.detach().clone())
        samples = qtx.sampler.Samples(spins, state(spins), reweight=1.0)
        state._maximum.zero_()
        expected = reference.Oloc(state, samples)
        max_expected = state._maximum.clone()
        state._maximum.zero_()
        with (patch.object(common_operators, "_MAX_EXCHANGE_MASK_ELEMENTS", mask_limit),
              patch.object(common_operators, "_replay_exchange", wraps=common_operators._replay_exchange) as replay):
            actual = fast.Oloc(state, samples)
            assert replay.call_count == 2 and fast._local_energy_graph_used
        _, workspace = fast._local_energy_workspaces[torch.cuda.current_stream(spins.device)]
        if n == 1:
            assert workspace is prior
        if n == 2:
            assert workspace is not prior and workspace.graph.pool() == prior.graph.pool()
        prior = workspace
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-11)
        torch.testing.assert_close(state._maximum, max_expected, rtol=0, atol=0)
    # Fully polarized spins have no exchanges and need no graph capture.
    polarized = torch.ones_like(spins[:1])
    torch.testing.assert_close(fast.Oloc(state, polarized), reference.Oloc(state, polarized), rtol=0, atol=0)
    print("Local energy reuses its graph after weight updates and recaptures after replacing parameter storage.")


def check_projected_vjp():
    import torch
    from triton.errors import TritonError

    import quantax_torch as qtx
    from examples_torch.run_pipeline import Config
    from examples_torch.workflows import _system
    from quantax_torch.optimizer import minsr
    from quantax_torch.optimizer import _res_sum_cuda

    batched = minsr.batched_res_sum_vjp

    def require_batched(*args, **kwargs):
        result = batched(*args, **kwargs)
        assert result is not None and result[2]
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
        if size == 4:
            with patch.object(_res_sum_cuda, "periodic_weight_grad", side_effect=TritonError) as unsupported:
                fallback = sorted(opt.jacobian_blocks(spins))
                attempts = unsupported.call_count
                assert attempts > 0
                torch.testing.assert_close(sorted(opt.jacobian_blocks(spins)), fallback)
                assert unsupported.call_count == attempts
            torch.testing.assert_close(fallback, reference)
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
    native_bytes = sum(block.numel() * block.element_size() for block in blocks)
    # Keep the local-energy right-hand side fixed across cache strategies.
    with patch.object(opt, "get_Ebar", return_value=opt.get_Ebar(samples)):
        with patch.object(opt, "_cache_budget", return_value=0):
            reference = opt.get_step(samples)
        for budget, plan in ((2 * native_bytes, ("normalized", None)),
                             (native_bytes, ("native", None)),
                             (blocks[-1].numel() * blocks[-1].element_size(), ("native", 1))):
            with patch.object(opt, "_cache_budget", return_value=budget):
                assert opt._cache_plan(samples, 8) == plan
                torch.testing.assert_close(opt.get_step(samples), reference, rtol=1e-11, atol=1e-11)
        opt.jacobian_batch = 3
        with patch.object(opt, "_cache_budget", return_value=0):
            batched_reference = opt.get_step(samples)
        with (patch.object(opt, "_cache_budget", return_value=blocks[-1].numel() * blocks[-1].element_size()),
              patch.object(opt, "_block_vjp", wraps=opt._block_vjp) as backward):
            actual = opt.get_step(samples)
            assert opt._last_cache_plan == ("native", 1)
            nbatches = (len(spins) + opt.jacobian_batch - 1) // opt.jacobian_batch
            assert backward.call_count == nbatches * (2 * len(blocks) - 1)
            torch.testing.assert_close(actual, batched_reference, rtol=1e-11, atol=1e-11)
        get_Tmat = opt.get_Tmat

        def fail_with_expanded_cache(*args, **kwargs):
            if opt._cache_limit is None:
                assert opt._last_cache_plan == ("native", None)
                raise torch.cuda.OutOfMemoryError("Simulated allocation failure with full derivative cache")
            return get_Tmat(*args, **kwargs)

        with (patch.object(minsr, "_DERIVATIVE_CACHE_BYTES", 1),
              patch.object(opt, "_cache_budget", return_value=native_bytes),
              patch.object(opt, "get_Tmat", side_effect=fail_with_expanded_cache) as attempts):
            actual = opt.get_step(samples)
            assert attempts.call_count == 2
            assert opt.jacobian_batch == 3 and opt._last_cache_plan == (None, None)
            torch.testing.assert_close(actual, batched_reference, rtol=1e-11, atol=1e-11)
    print("Normalized, native and partial caches agree with a two-pass weighted MinSR update.")


def check_cuda_gram_syrk():
    import torch
    from quantax_torch.optimizer import _cuda_gram
    from quantax_torch.optimizer._cuda_gram import add_gram_lower_, add_gram_tiled_, mirror_gram_, tune_gram

    # A non-default stream detects dispatch on the wrong cuBLAS stream.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        block = torch.randn(71, 4133, device="cuda", dtype=torch.float64)
        original = torch.randn(71, 71, device="cuda", dtype=torch.float64)
        result = original.clone()
        dense = block @ block.mT
        tiled = torch.zeros_like(result)
        add_gram_tiled_(tiled, block, 24)
        mirror_gram_(tiled, 24, False)
        torch.testing.assert_close(tiled, dense, rtol=2e-13, atol=1e-11)
        with patch.object(_cuda_gram, "add_gram_lower_", return_value=False):
            assert tune_gram(tiled, block, 24) in ("full GEMM", "tiled GEMM")
        assert not torch.count_nonzero(tiled)
        assert not add_gram_lower_(result, block[:, ::2])
        if not add_gram_lower_(result, block):
            print("Native CUDA triangular Gram check skipped: cuBLAS path unavailable.")
            return
        expected = original + dense.tril()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(result, expected, rtol=2e-13, atol=1e-11)
    print("Native CUDA triangular Gram agrees with FP64 GEMM on a non-default stream.")


def check_cuda_eigh_cache():
    from types import SimpleNamespace

    import torch
    from quantax_torch.optimizer.solver import _release_eigh_cache

    if torch.version.cuda is None:
        return
    matrix = torch.empty((4, 4), device="cuda", dtype=torch.float64)
    size = matrix.numel() * matrix.element_size()
    with (patch.object(torch.cuda, "mem_get_info", return_value=(size, 20 * size)) as memory,
          patch.object(torch.cuda, "memory_stats", return_value={
              "reserved_bytes.all.current": 12 * size,
              "allocated_bytes.all.current": 2 * size,
          }) as stats,
          patch.object(torch.cuda, "empty_cache") as release,
          patch.object(torch.backends.cuda, "preferred_linalg_library",
                       return_value=SimpleNamespace(name="Cusolver")) as backend):
        _release_eigh_cache(matrix)
        release.assert_called_once()
        memory.return_value = (10 * size, 20 * size)
        _release_eigh_cache(matrix)
        memory.return_value = (size, 20 * size)
        stats.return_value["inactive_split_bytes.all.current"] = 10 * size
        _release_eigh_cache(matrix)
        backend.return_value = SimpleNamespace(name="Magma")
        stats.return_value["inactive_split_bytes.all.current"] = 0
        _release_eigh_cache(matrix)
        release.assert_called_once()
    print("Large FP64 solves release reusable CUDA cache only under device memory pressure.")


def check_cuda_derivative_batches():
    import torch
    import quantax_torch as qtx
    from examples_torch.run_pipeline import Config
    from examples_torch.workflows import _system
    from quantax_torch.optimizer import minsr

    hamiltonian, state = _system(Config(L=4, depth=4, channels=4), True)
    spins = torch.as_tensor(configurations(), device=state.device)
    opt = qtx.optimizer.MinSR(state, hamiltonian)
    capacity = 8 * 2**30
    with patch.object(minsr, "_cuda_memory", return_value=(capacity, capacity)):
        assert opt._derivative_batch(spins) == len(spins)
        expected = dict(opt.jacobian_blocks(spins))
    with patch.object(minsr, "_cuda_memory", return_value=(capacity // 10, capacity)):
        assert opt._derivative_batch(spins) == 1
        actual = dict(opt.jacobian_blocks(spins))
    for block in expected:
        torch.testing.assert_close(actual[block], expected[block], rtol=3e-5, atol=2e-6)

    samples = qtx.sampler.Samples(spins, state(spins), reweight=1.0)
    get_Tmat = opt.get_Tmat

    def fail_full_batch(*args, **kwargs):
        if opt._jacobian_batch_limit is None:
            opt._last_jacobian_batch = len(spins)
            raise torch.cuda.OutOfMemoryError("Simulated derivative activation allocation failure")
        return get_Tmat(*args, **kwargs)

    with (patch.object(minsr, "_cuda_memory", return_value=(capacity, capacity)),
          patch.object(opt, "_cache_budget", return_value=0),
          patch.object(opt, "get_Ebar", wraps=opt.get_Ebar) as energy,
          patch.object(opt, "get_Tmat", side_effect=fail_full_batch)):
        assert opt.get_step(samples).isfinite().all()
        energy.assert_called_once()
    assert opt.jacobian_batch is None and opt._last_jacobian_batch == len(spins) // 2
    print("CUDA Jacobian batches respond to memory headroom and shrink after an allocation failure.")


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
