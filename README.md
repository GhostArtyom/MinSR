# MinSR
Repository for the paper

_Empowering deep neural quantum states through efficient optimization_

Related links:
- [Paper](https://www.nature.com/articles/s41567-024-02566-1)
- [Preprint](https://arxiv.org/abs/2302.01941) (not up-to-date)
- [GitHub](https://github.com/ChenAo-Phys/MinSR)
- [Zenodo](https://zenodo.org/doi/10.5281/zenodo.7657551)

![images](images/geometry_Fig1a.jpg)

![matrix](images/matrix_Fig1b.jpg)


## Contents

- Images
- Data
- Quantax, a package for reproducing results in the paper
- Tutorials for Quantax
- Examples to train networks and measure the variational energy
- Network parameters on the $10 \times 10$ square lattice at $J_2 / J_1 = 0.5$

## Installation

The repository provides two independent uv environments for Linux, including Fedora on WSL 3.x. Each directory contains its own `pyproject.toml`, `uv.lock`, and `.python-version`:

| Project directory | Python | JAX | cuDNN |
| --- | --- | --- | --- |
| `environments/py39-jax0425/` | 3.9 | 0.4.25 | 8 |
| `environments/py311-jax062/` | 3.11 | 0.6.2 | 9 |

Both use official PyPI (`https://pypi.org/simple`); PyPI artifacts download from `https://files.pythonhosted.org`. The historical GPU JAX wheel comes from Google's official archive, and QuSpin is built locally.

From the repository root, install the environment selected for the detected GPU:

```bash
uv run --no-project scripts/uv_machine.py sync --locked
```

The helper selects a project under `environments/` and installs its dependencies into the root `.venv/`. Each project keeps its own configuration and lockfile. For notebooks, select `.venv/bin/python`.

To choose an environment explicitly:

```bash
uv run --no-project scripts/uv_machine.py --environment py39-jax0425 sync --locked
uv run --no-project scripts/uv_machine.py --environment py311-jax062 sync --locked
```

Run only the command for the environment you intend to use.

### System requirements

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), Git, and the build dependencies for your distribution:

| Distribution family | Build packages |
| --- | --- |
| Debian / Ubuntu | `build-essential libboost-dev libgmp-dev libmpfr-dev libmpc-dev` |
| Fedora / RHEL | `gcc gcc-c++ make boost-devel gmp-devel mpfr-devel libmpc-devel` |
| Arch | `base-devel boost gmp mpfr libmpc` |
| Other distributions | Equivalent C/C++ compiler, OpenMP runtime, Boost headers, and GMP/MPFR/MPC development libraries |

Package availability depends on the release and enabled repositories. GCC is the simplest build option; Clang also needs a working OpenMP runtime, usually `libomp`. On clusters without administrator access, use the equivalent compiler/library modules or an administrator-provided container.

### Build QuSpin

Build the historical QuSpin commit with the Python ABI of the selected environment. For the Python 3.11 / JAX 0.6.2 environment:

```bash
git clone --filter=blob:none https://github.com/QuSpin/QuSpin.git ../QuSpin
git -C ../QuSpin checkout fc03e08
uv venv --python 3.11 ../QuSpin/.venv-py311
uv pip install --python ../QuSpin/.venv-py311/bin/python \
    "numpy==1.26.3" "cython==0.29.36" \
    "setuptools==68.2.2" "wheel==0.41.2"
```

Skip cloning if `../QuSpin` already exists. This commit reports version 0.3.7. With recent Boost headers, change the Linux `-std=c++11` option in `../QuSpin/quspin/basis/basis_general/_basis_general_core/setup.py` to `-std=c++14`. The explicit option overrides `CXXFLAGS`.

```bash
(
    cd ../QuSpin
    CC=gcc CXX=g++ CFLAGS="-O2 -g0" CXXFLAGS="-O2 -g0 -std=c++14" \
        .venv-py311/bin/python setup.py build --build-base build-py311 \
        --parallel 4 --omp bdist_wheel
)
```

Reduce parallelism if memory is limited. Clang also requires a working OpenMP runtime. The resulting wheel must remain at `../QuSpin/dist/quspin-0.3.7-cp311-cp311-linux_x86_64.whl`. Rebuild on each target machine when its Python ABI, CPU instructions, glibc/libstdc++, or OpenMP runtime differs.

### Python 3.11 / JAX 0.6.2

A rebuilt wheel has a new checksum. Register it before syncing:

```bash
uv run --no-project scripts/uv_machine.py --environment py311-jax062 remove --no-sync quspin
uv run --no-project scripts/uv_machine.py --environment py311-jax062 add --no-sync ../QuSpin/dist/quspin-0.3.7-cp311-cp311-linux_x86_64.whl
uv run --no-project scripts/uv_machine.py --environment py311-jax062 sync --locked
```

After the local wheel has been registered, routine use from the repository root is:

```bash
source .venv/bin/activate
uv run --active --locked --project environments/py311-jax062 python scripts/check_gpu.py
uv run --active --locked --project environments/py311-jax062 python examples/train.py
```

The `--active` option uses the activated root `.venv/`. In a notebook editor, select `.venv/bin/python`; `ipykernel` is included. Adjust `max_parallel` and sample counts for the available GPU memory before running the full training example.

Add `--jax-only` to check JAX kernels without importing QuSpin or Quantax. Set `JAX_PLATFORMS=cpu` for a CPU check.

The GPU check covers a JIT elementwise operation, convolution, float64 eigensolver, and two MinSR updates on a $4 \times 4$ lattice with 16 samples. Shared Quantax sharding helpers use `NamedSharding` in place of the removed `PositionalSharding`; both JAX 0.4.25 and 0.4.30 passed CPU checks of these helpers.

### Python 3.9 / JAX 0.4.25

For this environment, repeat the QuSpin build with Python 3.9, a build environment such as `../QuSpin/.venv-py39`, and `--build-base build-py39`. The wheel must be `../QuSpin/dist/quspin-0.3.7-cp39-cp39-linux_x86_64.whl`. Register the target machine's wheel and use the independent project:

```bash
uv run --no-project scripts/uv_machine.py --environment py39-jax0425 remove --no-sync quspin
uv run --no-project scripts/uv_machine.py --environment py39-jax0425 add --no-sync ../QuSpin/dist/quspin-0.3.7-cp39-cp39-linux_x86_64.whl
uv run --no-project scripts/uv_machine.py --environment py39-jax0425 sync --locked
source .venv/bin/activate
uv run --active --locked --project environments/py39-jax0425 python scripts/check_gpu.py
uv run --active --locked --project environments/py39-jax0425 python examples/train.py
```

Run these commands from the repository root. The interpreter is `.venv/bin/python` after selecting and syncing this profile. This project installs the shared Quantax source in editable mode.

Keep each project's configuration and lockfile together. Use `uv run --no-project scripts/uv_machine.py --environment py311-jax062 lock` or `uv run --no-project scripts/uv_machine.py --environment py39-jax0425 lock` to update only the intended lockfile. These projects are not workspace members; they maintain independent lockfiles.

### CUDA compatibility and troubleshooting

Install the GPU driver on the host. Under WSL this means the Windows driver; do not install a Linux display driver inside WSL. The CUDA version printed by `nvidia-smi` describes driver capability, not the user-space CUDA version selected by uv. See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) and [NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

Use `JAX_PLATFORMS=cpu` to select CPU execution. For GPU validation, the check scripts set `JAX_PLATFORMS=cuda` so initialization errors cannot silently fall back to CPU. Check whether `LD_LIBRARY_PATH` overrides the packaged CUDA libraries if library loading fails, while preserving required system or cluster paths.

For missing headers, Boost standard errors, or `GLIBC_*` / `GLIBCXX_*` failures, install compatible development packages and rebuild QuSpin with the matching compiler. A local wheel is not automatically portable between machines. Download failures require access to official PyPI and, for the legacy JAX wheel, the Google archive.

For `cannot enable executable stack` with the historical JAX wheel, install a `patchelf` version supporting `--clear-execstack` and patch local copies:

```bash
for library in .venv/lib/python3.9/site-packages/jaxlib/xla_extension.so .venv/lib/python3.9/site-packages/jaxlib/cuda/*.so; do
    cp "$library" "$library.local"
    mv "$library.local" "$library"
    patchelf --clear-execstack "$library"
done
```

Repeat after reinstalling that JAX wheel. Copying first keeps the patch separate from uv's shared cache.

### Parallelism
- Quantax can work when all devices are on the same node
- Multiple-node compatibility is to be implemented

## Questions?
For any questions, bugs, or suggestions, please raise an issue in GitHub.
