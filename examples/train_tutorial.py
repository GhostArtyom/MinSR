"""修改开头的教程预设和共用参数，再从仓库根目录运行本文件。"""

from pathlib import Path

# ======================== 在这里修改参数 ========================
TUTORIAL = "j1j2"  # 选择下面的一个预设；修改该预设中的值即可调整体系和网络

# 各教程的物理模型、网络和训练阶段设置；未使用的专属参数不必填写。
# GEOMETRY: chain / square / triangular；SIZE: 链长或二维晶格边长。
# MODEL: 哈密顿量；J1/J2: 第一/第二近邻耦合；H_FIELD: Ising 横场。
# MSR: Marshall 符号变换，仅适用于双子格；TOTAL_SZ: sum(sigma_z)，为物理总 Sz 的两倍。
# NETWORK: ResSum / RBM_Dense / RBM_Conv；DEPTH: ResSum 卷积层数，须为正偶数。
# CHANNELS: 卷积通道数；KERNEL_SIZE: ResSum 卷积核边长；RBM_ALPHA: RBM_Dense 隐藏单元数与格点数之比。
# USE_SINH: 允许实数 ResSum 输出变号；DTYPE: 网络外计算类型，也决定 ResSum 是否使用复输出。
# STEPS: 初始采样训练更新次数；SYMMETRY_STEPS: 加入自旋反演/点群投影后的继续训练次数。
# EXACT_STEPS: 另起一个独立初态做 TDVP_exact 比较；STEP_SIZE: 参数更新的步长系数。
PRESETS = {
    "ising": {
        "GEOMETRY": "chain", "SIZE": 12,
        "MODEL": "ising", "J1": 1.0, "H_FIELD": 1.0,
        "NETWORK": "RBM_Dense", "RBM_ALPHA": 4,
        "STEPS": 100, "EXACT_STEPS": 100, "STEP_SIZE": 1e-2,
    },
    "heisenberg": {
        "GEOMETRY": "square", "SIZE": 4,
        "MODEL": "heisenberg", "J1": 1.0, "MSR": True, "TOTAL_SZ": 0,
        "NETWORK": "ResSum", "DEPTH": 4, "CHANNELS": 8, "KERNEL_SIZE": 4,
        "STEPS": 200, "SYMMETRY_STEPS": 200, "STEP_SIZE": 1e-3,
    },
    "j1j2": {
        "GEOMETRY": "square", "SIZE": 4,
        "MODEL": "heisenberg", "J1": 1.0, "J2": 0.5, "MSR": True, "TOTAL_SZ": 0,
        "NETWORK": "ResSum", "DEPTH": 4, "CHANNELS": 8, "KERNEL_SIZE": 4,
        "USE_SINH": True,
        "STEPS": 200, "SYMMETRY_STEPS": 200, "STEP_SIZE": 1e-3,
    },
    "triangular": {
        "GEOMETRY": "triangular", "SIZE": 6,  # 全局 ED 超过默认规模上限；小体系可改为 4
        "MODEL": "heisenberg", "J1": 1.0, "J2": 0.125, "TOTAL_SZ": 0,
        "NETWORK": "ResSum", "DEPTH": 4, "CHANNELS": 8, "KERNEL_SIZE": 3,
        "DTYPE": "complex128",
        "USE_NEEL120": True,  # 固定的 120° Néel 相位先验；周期 SIZE 须为 3 的倍数，改为 4 时关闭
        "STEPS": 200, "SYMMETRY_STEPS": 200, "STEP_SIZE": 1e-3,
    },
    "excited": {
        "GEOMETRY": "square", "SIZE": 4,
        "MODEL": "heisenberg", "J1": 1.0, "J2": 0.5, "MSR": True, "TOTAL_SZ": 0,
        "NETWORK": "ResSum", "DEPTH": 4, "CHANNELS": 8, "KERNEL_SIZE": 4,
        "USE_SINH": True,
        "MOMENTUM": "pi",  # 自动随 SIZE 选择 (π, π)；也可直接填扇区编号，如 4×4 上的 (2, 2)
        "SPIN_INVERSION": -1,  # 自旋反演奇扇区：psi(-s) = -psi(s)；+1 为偶扇区
        "STEPS": 200, "SYMMETRY_STEPS": 200, "STEP_SIZE": 1e-3,
    },
}

# 所有教程共用的运行参数。
PBC = True  # True 为周期边界；开放边界需用 RBM_Dense，并关闭 SYMMETRY_STEPS
OPTIMIZER = "TDVP"  # 采样训练优化器：TDVP / MinSR；不影响独立的 TDVP_exact 比较
NSAMPLES = 1000  # 每次采样的构型数，须能被本地设备数整除
MAX_PARALLEL = 32768  # 每台设备的最大前向批量；内存不足时调小
THERMAL_STEPS = None  # 初始化采样器时的热化步数；None 使用 20 * 格点数
SWEEP_STEPS = None  # 每次采样的 Metropolis 更新步数；None 使用 2 * 格点数
SEED = 2347  # 训练随机种子
PARAMS_DTYPE = "float32"  # 可训练网络参数的精度，与 DTYPE 分开设置
LOG_EVERY = 10  # 每隔多少次更新向 training.log 写入 E、Var、完成步数和耗时
SAVE_EVERY = 100  # 中间权重的保存间隔；0 表示只保存各阶段最终权重

ED_MAX_DIM = 2**20  # ED 基底维数上界的限制，在枚举基底之前检查
ED_TOL = 1e-10  # ED 迭代求解容差；越小要求越严格
ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache" / "tutorial_ed"  # 跨训练运行复用的 ED 能量缓存
OUTPUT_DIR = ROOT / "runs"  # 按体系、格点和运行时间分目录保存日志、配置、CSV、权重和 SVG
# ===================================================================

# 从所选预设读取参数；以下通常无需修改。
# get 的第二个参数是省略该键时的默认值；SYMMETRY_STEPS/EXACT_STEPS 为 0 表示关闭对应阶段。
_preset = PRESETS[TUTORIAL]
GEOMETRY, SIZE, MODEL = _preset["GEOMETRY"], _preset["SIZE"], _preset["MODEL"]
J1, J2, H_FIELD = _preset["J1"], _preset.get("J2", 0.0), _preset.get("H_FIELD", 1.0)
MSR = _preset.get("MSR", False)
TOTAL_SZ = _preset.get("TOTAL_SZ")
MOMENTUM = _preset.get("MOMENTUM")
if MOMENTUM == "pi":
    if SIZE % 2:
        raise ValueError('MOMENTUM="pi" requires an even SIZE.')
    MOMENTUM = (SIZE // 2, SIZE // 2)
SPIN_INVERSION = _preset.get("SPIN_INVERSION", 1)
USE_NEEL120 = _preset.get("USE_NEEL120", False)
NETWORK = _preset["NETWORK"]
DEPTH, CHANNELS, KERNEL_SIZE = _preset.get("DEPTH", 4), _preset.get("CHANNELS", 8), _preset.get("KERNEL_SIZE", 4)
RBM_ALPHA = _preset.get("RBM_ALPHA", 4)
USE_SINH = _preset.get("USE_SINH", False)
DTYPE = _preset.get("DTYPE", "float64")
STEPS = _preset["STEPS"]
SYMMETRY_STEPS, EXACT_STEPS = _preset.get("SYMMETRY_STEPS", 0), _preset.get("EXACT_STEPS", 0)
STEP_SIZE = _preset["STEP_SIZE"]

import csv
import hashlib
import io
import json
import logging
import math
import os
from contextlib import redirect_stdout
from datetime import datetime
from time import perf_counter
from zoneinfo import ZoneInfo

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import quantax as qtx

logger = logging.getLogger("minsr.tutorial")


def setup_plot_font():
    """Require real Arial and preserve editable SVG text."""
    try:
        path = font_manager.findfont("Arial", fallback_to_default=False)
    except ValueError:
        # A newly installed system font may be missing from Matplotlib's cache.
        for path in font_manager.findSystemFonts():
            try:
                family = font_manager.get_font(path).family_name
            except (OSError, RuntimeError):
                continue
            if family == "Arial":
                font_manager.fontManager.addfont(path)
        try:
            path = font_manager.findfont("Arial", fallback_to_default=False)
        except ValueError as error:
            raise RuntimeError("Install Arial before running this script.") from error
    if font_manager.get_font(path).family_name != "Arial":
        raise RuntimeError("Install Arial before running this script.")
    plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none"})


def write_json(path, data):
    """Atomically replace a small JSON file, including ED cache entries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def exact_energy(hamiltonian, lattice, symm=None):
    """Cache the full-system ground energy or the minimum in an explicit sector."""
    sector = None
    if symm is not None:
        sector = {
            "total_sz": symm.total_sz,
            "spin_inversion": symm.spin_inversion,
            "transforms": np.asarray(symm.get_symm_spins(jnp.arange(1, lattice.nsites + 1))).tolist(),
            "characters": [[float(z.real), float(z.imag)] for z in np.asarray(symm.eigval)],
        }
    spec = {
        "schema": 1, "geometry": GEOMETRY, "shape": lattice.shape,
        "pbc": lattice.pbc.tolist(), "terms": hamiltonian.op_list, "sector": sector,
    }
    encoded = json.dumps(spec, sort_keys=True, default=lambda x: np.asarray(x).tolist(), allow_nan=False)
    spec = json.loads(encoded)
    cache_file = Path(CACHE_DIR) / (hashlib.sha256(encoded.encode()).hexdigest() + ".json")
    residual_limit = max(1e-8, 10 * ED_TOL)
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            if (cached["system"] == spec and math.isfinite(cached["energy"])
                    and 0 < cached["tolerance"] <= ED_TOL
                    and 0 <= cached["residual_norm"] <= residual_limit):
                return cached
        except (ValueError, KeyError, TypeError):
            pass

    nsites = int(lattice.nsites)
    dimension_bound = 2**nsites
    if symm is not None and symm.total_sz is not None:
        dimension_bound = math.comb(nsites, (nsites + symm.total_sz) // 2)
    if dimension_bound > ED_MAX_DIM:
        raise ValueError(
            f"ED basis upper bound {dimension_bound:,} exceeds ED_MAX_DIM={ED_MAX_DIM:,}. "
            "Reduce SIZE, or increase ED_MAX_DIM if resources permit. "
            "The 6x6 triangular tutorial is too large for the default full-system ED."
        )
    start = perf_counter()
    if symm is None:
        operator = hamiltonian.get_quspin_op()
    else:
        # Historical QuSpin requires complex storage for some momentum sectors,
        # even when the wavefunction characters and training parameters are real.
        from quspin.operators import quantum_LinearOperator

        symm.basis.make()
        operator = quantum_LinearOperator(hamiltonian.op_list, basis=symm.basis, dtype=np.complex128)
    dimension = operator.shape[0]
    if dimension == 0:
        raise ValueError("The requested symmetry sector is empty.")
    if dimension <= 2:
        values, vectors = np.linalg.eigh(operator.matmat(np.eye(dimension)))
    else:
        # Reproducible starting vector, independent of the training PRNG.
        v0 = np.random.default_rng(0).normal(size=dimension)
        values, vectors = operator.eigsh(k=1, which="SA", tol=ED_TOL, v0=v0)
    energy = float(values[0].real)
    residual = float(np.linalg.norm(operator.matvec(vectors[:, 0]) - energy * vectors[:, 0]))
    if not math.isfinite(energy) or not math.isfinite(residual) or residual > residual_limit:
        raise RuntimeError(f"ED did not converge accurately enough: residual={residual}.")
    result = {
        "system": spec, "energy": energy, "energy_per_site": energy / nsites,
        "dimension": dimension, "tolerance": ED_TOL, "residual_norm": residual,
        "seconds": perf_counter() - start,
    }
    write_json(cache_file, result)
    return result


def main():
    start = perf_counter()
    shape = str(SIZE) if GEOMETRY == "chain" else f"{SIZE}x{SIZE}"
    couplings = f"J-{J1:g}_h-{H_FIELD:g}" if MODEL == "ising" else f"J1-{J1:g}_J2-{J2:g}"
    system_name = f"{TUTORIAL}_{GEOMETRY}_{shape}_{couplings}"
    local_tz = ZoneInfo("Asia/Hong_Kong")
    stamp = datetime.now(local_tz).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(OUTPUT_DIR) / system_name / stamp
    run_dir.mkdir(parents=True, exist_ok=False)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(run_dir / "training.log", mode="x", encoding="utf-8")
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    try:
        logger.info(f"Run directory: {run_dir}; timezone: {local_tz.key}")
        logger.info(f"System: {system_name}; model={MODEL}; PBC={PBC}")
        logger.info(
            f"Steps={STEPS}; symmetry_steps={SYMMETRY_STEPS}; exact_steps={EXACT_STEPS}; "
            f"samples={NSAMPLES}; max_parallel={MAX_PARALLEL}; step_size={STEP_SIZE}; optimizer={OPTIMIZER}"
        )
        if NETWORK == "ResSum":
            logger.info(f"Network: ResSum(depth={DEPTH}, channels={CHANNELS}, kernel_size={KERNEL_SIZE}, use_sinh={USE_SINH})")
        elif NETWORK == "RBM_Dense":
            logger.info(f"Network: RBM_Dense(alpha={RBM_ALPHA})")
        else:
            logger.info(f"Network: {NETWORK}(channels={CHANNELS})")
        logger.info(f"Symmetry settings: total_sz={TOTAL_SZ}; spin_inversion={SPIN_INVERSION}; momentum={MOMENTUM}")
        logger.info(f"Sampling: thermal_steps={THERMAL_STEPS}; sweep_steps={SWEEP_STEPS}; seed={SEED}")
        logger.info(f"Precision: dtype={DTYPE}; params_dtype={PARAMS_DTYPE}")
        logger.info(f"Devices: {jax.devices()}")
        settings = {key: value for key, value in globals().items() if key.isupper() and key != "PRESETS"}
        settings = json.loads(json.dumps(settings, default=str))
        write_json(run_dir / "config.json", settings)
        # QuSpin prints successful operator checks; keep routine output in the log concise.
        with io.StringIO() as output, redirect_stdout(output):
            run_training(run_dir, system_name, start)
    except (Exception, KeyboardInterrupt):
        logger.exception(f"Training stopped after {perf_counter() - start:.2f} s")
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()
        plt.close("all")
    return run_dir


def run_training(run_dir, system_name, start):
    if GEOMETRY not in ("chain", "square", "triangular") or SIZE < 2:
        raise ValueError("Choose chain/square/triangular and SIZE >= 2.")
    if MODEL not in ("ising", "heisenberg") or NETWORK not in ("ResSum", "RBM_Dense", "RBM_Conv"):
        raise ValueError("Unknown MODEL or NETWORK; see the settings block.")
    if OPTIMIZER not in ("TDVP", "MinSR"):
        raise ValueError("OPTIMIZER must be TDVP or MinSR.")
    if STEPS < 1 or min(SYMMETRY_STEPS, EXACT_STEPS, SAVE_EVERY) < 0 or LOG_EVERY < 1:
        raise ValueError("Require STEPS >= 1, LOG_EVERY >= 1 and nonnegative optional step counts.")
    if not math.isfinite(STEP_SIZE) or STEP_SIZE <= 0 or NSAMPLES < 1 or MAX_PARALLEL < 1:
        raise ValueError("STEP_SIZE, NSAMPLES and MAX_PARALLEL must be positive.")
    if ED_MAX_DIM < 1 or not math.isfinite(ED_TOL) or ED_TOL <= 0:
        raise ValueError("ED_MAX_DIM and ED_TOL must be positive.")
    if NETWORK == "ResSum" and (DEPTH < 2 or DEPTH % 2 or CHANNELS < 1 or KERNEL_SIZE < 1):
        raise ValueError("ResSum needs positive even DEPTH and positive CHANNELS/KERNEL_SIZE.")
    if NETWORK != "RBM_Dense" and not PBC:
        raise ValueError("These convolution networks use circular padding; use PBC=True or RBM_Dense.")
    if MSR and (GEOMETRY == "triangular" or (PBC and SIZE % 2)):
        raise ValueError("MSR requires a bipartite chain/square lattice; periodic SIZE must be even.")
    if MODEL == "heisenberg" and TOTAL_SZ is None:
        raise ValueError("Heisenberg exchange sampling requires TOTAL_SZ.")
    if SYMMETRY_STEPS and (TOTAL_SZ not in (None, 0) or not PBC):
        raise ValueError("The tutorial refinement requires TOTAL_SZ=0 and periodic boundaries.")
    if SPIN_INVERSION not in (-1, 1):
        raise ValueError("SPIN_INVERSION must be +1 or -1.")
    if MOMENTUM is not None and (GEOMETRY != "square" or NETWORK != "ResSum" or not PBC):
        raise ValueError("Nonzero tutorial momentum requires a periodic square ResSum network.")
    if USE_NEEL120 and (GEOMETRY != "triangular" or DTYPE != "complex128" or (PBC and SIZE % 3)):
        raise ValueError("Neel120 requires complex128 on a triangular lattice, with periodic SIZE divisible by 3.")
    if DTYPE not in ("float64", "complex128") or PARAMS_DTYPE not in ("float32", "float64"):
        raise ValueError("Use float64/complex128 computations and float32/float64 parameters.")
    if NETWORK == "ResSum" and DTYPE == "complex128" and CHANNELS % 2:
        raise ValueError("Complex-output ResSum requires an even CHANNELS count.")
    setup_plot_font()
    qtx.set_random_seed(SEED)
    qtx.set_default_dtype(getattr(jnp, DTYPE))
    qtx.set_params_dtype(getattr(jnp, PARAMS_DTYPE))
    lattice = {"chain": qtx.sites.Chain, "square": qtx.sites.Square, "triangular": qtx.sites.Triangular}[GEOMETRY](SIZE, pbc=PBC)
    nsites = int(lattice.nsites)
    if MODEL == "heisenberg" and (abs(TOTAL_SZ) > nsites or (TOTAL_SZ + nsites) % 2):
        raise ValueError("TOTAL_SZ must match the number of sites and its parity.")
    if NSAMPLES % jax.local_device_count():
        raise ValueError("NSAMPLES must be divisible by the local device count.")
    neighbors = [1, 2] if MODEL == "heisenberg" and J2 != 0 else [1]
    hamiltonian = (qtx.operator.Ising(h=H_FIELD, J=J1) if MODEL == "ising" else
                   qtx.operator.Heisenberg(J=[J1, J2] if len(neighbors) == 2 else [J1], n_neighbor=neighbors, msr=MSR))

    sym = qtx.symmetry
    translation = sym.TransND(MOMENTUM or 0) if PBC else sym.Identity()
    point_group = {"chain": sym.Flip, "square": sym.C4v, "triangular": sym.D6}[GEOMETRY]() if PBC else sym.Identity()
    initial_symm = sym.SpinInverse(SPIN_INVERSION) if MODEL == "ising" else sym.TotalSz(TOTAL_SZ)
    if MODEL == "ising" and PBC:
        initial_symm = initial_symm + translation + point_group
    refined_symm = initial_symm + sym.SpinInverse(SPIN_INVERSION) + point_group

    # The full Hilbert space gives a global ground state, independent of training symmetries.
    references = {"ground": exact_energy(hamiltonian, lattice)}
    if TUTORIAL == "excited" or MOMENTUM is not None or SPIN_INVERSION == -1:
        target_symm = refined_symm if SYMMETRY_STEPS else initial_symm
        references["sector"] = exact_energy(hamiltonian, lattice, target_symm + translation)

    def new_state():
        if NETWORK == "ResSum":
            net = qtx.nn.ResSum(DEPTH, CHANNELS, KERNEL_SIZE, use_sinh=USE_SINH, trans_symm=translation)
        elif NETWORK == "RBM_Dense":
            net = qtx.nn.RBM_Dense(features=RBM_ALPHA * nsites)
        else:
            net = qtx.nn.RBM_Conv(channels=CHANNELS)
        state = qtx.state.Variational(net, max_parallel=MAX_PARALLEL)
        if USE_NEEL120:
            state = state * qtx.nn.Neel120()
        return initial_symm(state)

    write_json(run_dir / "references.json", references)
    records = []
    fields = ["phase", "step", "energy", "energy_per_site", "variance", "error_per_site", "step_seconds", "elapsed_seconds"]
    with (run_dir / "training.csv").open("w", newline="") as data_file:
        writer = csv.DictWriter(data_file, fieldnames=fields)
        writer.writeheader()

        def train(state, phase, nsteps, offset=0, exact=False):
            logger.info(f"Phase: {phase}; steps={nsteps}")
            if exact:
                optimizer = qtx.optimizer.TDVP_exact(state, hamiltonian)
                sampler = None
            else:
                sampler_class = qtx.sampler.LocalFlip if MODEL == "ising" else qtx.sampler.NeighborExchange
                kwargs = {} if MODEL == "ising" else {"n_neighbor": neighbors}
                sampler = sampler_class(state, NSAMPLES, thermal_steps=THERMAL_STEPS, sweep_steps=SWEEP_STEPS, **kwargs)
                optimizer = getattr(qtx.optimizer, OPTIMIZER)(state, hamiltonian)
                jax.block_until_ready(sampler.current_wf)
            for i in range(nsteps + 1):
                tick = perf_counter()
                # Include a measurement after the final update, without another update.
                if i < nsteps:
                    step = optimizer.get_step() if exact else optimizer.get_step(sampler.sweep())
                    if not bool(jnp.all(jnp.isfinite(step))):
                        raise FloatingPointError(f"Nonfinite update in {phase}, step {i}.")
                elif not exact:
                    optimizer.get_Ebar(sampler.sweep())
                if exact:
                    psi = state.todense().normalize()
                    hpsi = hamiltonian @ psi
                    energy = float(np.real(psi @ hpsi))
                    variance = max(0.0, float(np.real(hpsi @ hpsi)) - energy**2)
                else:
                    energy, variance = float(optimizer.energy), float(optimizer.VarE)
                if not math.isfinite(energy) or not math.isfinite(variance):
                    raise FloatingPointError(f"Nonfinite energy/variance in {phase}, step {i}.")
                if i < nsteps:
                    state.update(step * STEP_SIZE)
                    jax.block_until_ready(state.ansatz)
                row = dict(zip(fields, [phase, offset + i, energy, energy / nsites, variance,
                           (energy - references["ground"]["energy"]) / nsites,
                           perf_counter() - tick, perf_counter() - start]))
                writer.writerow(row)
                data_file.flush()
                records.append(row)
                if i < nsteps and ((i + 1) % LOG_EVERY == 0 or i + 1 == nsteps):
                    logger.info(
                        f"E={energy:.6f}, Var={variance:.6g}, {i + 1}/{nsteps}, "
                        f"{row['step_seconds']:.2f}, {row['elapsed_seconds']:.2f}"
                    )
                if i < nsteps and SAVE_EVERY and (i + 1) % SAVE_EVERY == 0:
                    state.save(run_dir / f"params_{system_name}_{phase}_{i + 1}.eqx")
            state.save(run_dir / f"params_{system_name}_{phase}_final.eqx")

        state = None
        try:
            if EXACT_STEPS:
                state = new_state()  # Tutorial 1 compares two independently initialized states.
                train(state, "exact", EXACT_STEPS, exact=True)
            state = new_state()
            train(state, "sampled", STEPS)
            if SYMMETRY_STEPS:
                state = refined_symm(state)
                train(state, "symmetric", SYMMETRY_STEPS, offset=STEPS)
        except BaseException:
            if state is not None:
                checkpoint = run_dir / f"params_{system_name}_interrupted.eqx"
                state.save(checkpoint)
            raise

    # Plot only after all training has finished. Each phase retains its own curve.
    fig, ax = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
    for phase in dict.fromkeys(row["phase"] for row in records):
        rows = [row for row in records if row["phase"] == phase]
        ax.plot([row["step"] for row in rows], [row["energy_per_site"] for row in rows], label=phase)
    ax.axhline(references["ground"]["energy_per_site"], color="black", linestyle="--", label="ED global ground state")
    if "sector" in references:
        ax.axhline(references["sector"]["energy_per_site"], color="tab:red", linestyle=":", label="ED target-sector minimum")
    ax.set(xlabel="Parameter updates", ylabel="Energy per site", title=f"{TUTORIAL}: {GEOMETRY}, N={nsites}")
    ax.legend()
    ax.grid(alpha=0.2)
    figure_path = run_dir / f"energy_{system_name}.svg"
    fig.savefig(figure_path)
    plt.close(fig)
    write_json(run_dir / "summary.json", {"final_measurement": records[-1], "seconds": perf_counter() - start, "jax": jax.__version__})


if __name__ == "__main__":
    main()
