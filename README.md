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

The repository provides three independent uv environments for Linux x86-64. PyTorch is the default project at the repository root. Both JAX environments require QuSpin. For the PyTorch training and measurement examples, the lockfile covers the required Python packages.

| Project directory | Python | Backend | Environment directory |
| --- | --- | --- | --- |
| [`./`](#python-311--pytorch-2100) | 3.11 | PyTorch 2.10.0 / CUDA 12.8 | `.venv/` |
| [`environments/py39-jax0425/`](#python-39--jax-0425) | 3.9 | JAX 0.4.25 / cuDNN 8 | `environments/py39-jax0425/.venv/` |
| [`environments/py311-jax062/`](#python-311--jax-062) | 3.11 | JAX 0.6.2 / cuDNN 9 | `environments/py311-jax062/.venv/` |

### System requirements

Install uv and Git. The training examples require Arial for plots. The build packages below are for compiling QuSpin. CUDA optimizations that use Triton also require a C compiler. [uv installation](https://docs.astral.sh/uv/getting-started/installation/)

| Distribution family | Build packages |
| --- | --- |
| Debian / Ubuntu | `build-essential libboost-dev libgmp-dev libmpfr-dev libmpc-dev` |
| Fedora / RHEL | `gcc gcc-c++ make boost-devel gmp-devel mpfr-devel libmpc-devel` |
| Arch | `base-devel boost gmp mpfr libmpc` |
| Other distributions | Equivalent C/C++ compiler, OpenMP runtime, Boost headers, and GMP/MPFR/MPC development libraries |

Package availability depends on the release and enabled repositories. GCC is the simplest build option; Clang also needs a working OpenMP runtime, usually `libomp`. On clusters without administrator access, use the equivalent compiler/library modules or an administrator-provided container.

### Build QuSpin

Run the build commands from the repository root. Skip the clone command if `../QuSpin` already exists. The pinned QuSpin source reports version 0.3.7. The JAX and PyTorch projects on Python 3.11 share the same QuSpin wheel. For Python 3.9, follow the substitutions in the [JAX 0.4.25 section](#python-39--jax-0425).

```bash
git clone --filter=blob:none https://github.com/QuSpin/QuSpin.git ../QuSpin
git -C ../QuSpin checkout fc03e08
uv venv --python 3.11 ../QuSpin/.venv-py311
uv pip install --python ../QuSpin/.venv-py311/bin/python \
    "numpy==1.26.3" "cython==0.29.36" \
    "setuptools==68.2.2" "wheel==0.41.2"
```

With recent Boost headers, change the Linux `-std=c++11` option to `-std=c++14`. The explicit option overrides `CXXFLAGS`. [QuSpin `setup.py` on GitHub](https://github.com/QuSpin/QuSpin/blob/fc03e08/quspin/basis/basis_general/_basis_general_core/setup.py)

```bash
(
    cd ../QuSpin
    CC=gcc CXX=g++ CFLAGS="-O2 -g0" CXXFLAGS="-O2 -g0 -std=c++14" \
        .venv-py311/bin/python setup.py build --build-base build-py311 \
        --parallel 4 --omp bdist_wheel
)
```

Reduce parallelism if memory is limited. Clang also requires a working OpenMP runtime. The Python 3.11 JAX project registers the wheel as a local uv source. PyTorch users install it separately when they need QuSpin-backed features. Rebuild on each target machine when its Python ABI, CPU instructions, glibc/libstdc++, or OpenMP runtime differs. [Python 3.11 JAX project](environments/py311-jax062/pyproject.toml)

### Python 3.11 / JAX 0.6.2

A rebuilt wheel has a new checksum. Register it before syncing:

```bash
uv run --no-project scripts/uv_machine.py --environment py311-jax062 remove --no-sync quspin
uv run --no-project scripts/uv_machine.py --environment py311-jax062 add --no-sync ../QuSpin/dist/quspin-0.3.7-cp311-cp311-linux_x86_64.whl
uv run --no-project scripts/uv_machine.py --environment py311-jax062 sync --locked
```

After the local wheel has been registered, routine use from the repository root is:

```bash
environments/py311-jax062/.venv/bin/python -m scripts.check_jax
environments/py311-jax062/.venv/bin/python -m examples.train
```

In a notebook editor, select `environments/py311-jax062/.venv/bin/python`. The environment includes `ipykernel`. Adjust `max_parallel` and sample counts for the available GPU memory before running the full training example.

Add `--jax-only` to check JAX kernels without importing QuSpin or Quantax. Set `JAX_PLATFORMS=cpu` for a CPU check.

The full check covers JIT elementwise operations, convolution, the FP64 eigensolver, and two MinSR updates on a $4 \times 4$ lattice with 16 samples. It also checks batched derivatives against the full batch and device-side Heisenberg local energies against the QuSpin path. [JAX check script](scripts/check_jax.py)

### Python 3.9 / JAX 0.4.25

Build QuSpin with Python 3.9, a build environment such as `../QuSpin/.venv-py39`, and `--build-base build-py39`. Sync the locked dependencies first, then install the target machine's wheel separately. Use `--inexact` on later syncs to keep the installed QuSpin package.

```bash
uv run --no-project scripts/uv_machine.py --environment py39-jax0425 sync --locked --inexact
uv pip install --no-deps --python environments/py39-jax0425/.venv/bin/python \
    ../QuSpin/dist/quspin-0.3.7-cp39-cp39-linux_x86_64.whl
environments/py39-jax0425/.venv/bin/python -m scripts.check_jax
environments/py39-jax0425/.venv/bin/python -m examples.train
```

Run these commands from the repository root. The JAX projects install the shared Quantax source in editable mode.

After setting up QuSpin for the selected profile, use the first command to select the JAX profile for the detected GPU and sync its own `.venv/`. The second command updates the selected lockfile. Substitute `py39-jax0425` to update the Python 3.9 profile. [JAX environment selector](scripts/uv_machine.py)

```bash
uv run --no-project scripts/uv_machine.py sync --locked --inexact
uv run --no-project scripts/uv_machine.py --environment py311-jax062 lock
```

### Python 3.11 / PyTorch 2.10.0

From the repository root, sync the PyTorch project. It uses the official PyTorch CUDA 12.8 index for Torch. Keep `--inexact` when syncing an environment with QuSpin installed.

```bash
uv sync --locked --inexact
```

For exact diagonalization and other QuSpin-backed features, build the Python 3.11 wheel on the target machine and install it separately.

```bash
uv pip install --no-deps --python .venv/bin/python \
    ../QuSpin/dist/quspin-0.3.7-cp311-cp311-linux_x86_64.whl
```

Use the project interpreter to check CUDA availability and launch the training, symmetry training, and measurement pipeline.

```bash
.venv/bin/python -c 'import torch; assert torch.cuda.is_available()'
.venv/bin/python -m examples_torch.run_pipeline --device cuda
```

Pass `--device cpu` to run on CPU. Pipeline configuration and standalone entry points are documented in the [PyTorch guide](docs/pytorch-migration-plan.md#安装和运行).

### CUDA compatibility and troubleshooting

Install an NVIDIA driver that supports the CUDA runtime selected by the project. `nvidia-smi` reports the CUDA version supported by the driver. [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)

The JAX check script defaults to `JAX_PLATFORMS=cuda`. Set `JAX_PLATFORMS=cpu` to check CPU execution. If CUDA libraries fail to load, check whether `LD_LIBRARY_PATH` overrides the packaged libraries while preserving required system or cluster paths.

For missing headers, Boost standard errors, or `GLIBC_*` / `GLIBCXX_*` failures, install compatible development packages and rebuild QuSpin with the matching compiler. For download failures, check access to the package indexes recorded in the relevant lockfile, the Google archive for legacy JAX, or the PyTorch CUDA index for Torch.

For `cannot enable executable stack` with the historical JAX wheel, install a `patchelf` version supporting `--clear-execstack` and patch local copies:

```bash
for library in environments/py39-jax0425/.venv/lib/python3.9/site-packages/jaxlib/xla_extension.so \
    environments/py39-jax0425/.venv/lib/python3.9/site-packages/jaxlib/cuda/*.so; do
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
