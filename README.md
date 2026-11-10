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

(Many thanks to [Marin Bukov](https://github.com/mgbukov) for the help with QuSpin)

The repository includes the historical Quantax source used by MinSR. These instructions cover Linux x86_64, including WSL2, with Python 3.9 and JAX 0.4.25. The supplied lockfile uses CUDA 12 and cuDNN 8. Compatibility depends on the system libraries, compiler, and accelerator; a single lockfile cannot support every Linux installation.

### System requirements

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), Git, and the build dependencies for your distribution:

| Distribution family | Build packages |
| --- | --- |
| Debian / Ubuntu | `build-essential libboost-dev libgmp-dev libmpfr-dev libmpc-dev` |
| Fedora / RHEL | `gcc gcc-c++ make boost-devel gmp-devel mpfr-devel libmpc-devel` |
| Arch | `base-devel boost gmp mpfr libmpc` |
| Other distributions | Equivalent C/C++ compiler, OpenMP runtime, Boost headers, and GMP/MPFR/MPC development libraries |

Package availability depends on the release and enabled repositories. GCC is the simplest build option; Clang also needs a working OpenMP runtime, usually `libomp`. On clusters without administrator access, use the equivalent compiler/library modules or an administrator-provided container.

The prebuilt Python/JAX wheels require compatible glibc and CPU instructions. Older glibc systems and musl distributions may need a glibc-based container or a separate source build and lockfile. Containers still depend on the host CPU and GPU driver; they do not add missing hardware support.

### Build QuSpin

From a checkout containing `pyproject.toml` and `uv.lock`, build the required QuSpin version alongside the project:

```bash
git clone --filter=blob:none https://github.com/QuSpin/QuSpin.git ../QuSpin
git -C ../QuSpin checkout fc03e08
uv venv --python 3.9 ../QuSpin/.venv
uv pip install --python ../QuSpin/.venv/bin/python \
    "numpy==1.26.3" "cython==0.29.36" \
    "setuptools==68.2.2" "wheel==0.41.2"
```

Skip cloning if `../QuSpin` already exists. Commit `fc03e08` contains the required interfaces from the former `dev_0.3.8` branch and reports version 0.3.7.

For GCC/Clang, use C++14 with this historical source and recent Boost headers: change the Linux `-std=c++11` option to `-std=c++14` in `../QuSpin/quspin/basis/basis_general/_basis_general_core/setup.py`. The explicit option overrides `CXXFLAGS`, so setting the environment variable alone is insufficient. C++11 is an alternative only when the installed Boost headers support it; the required standard depends on those headers, not the distribution name.

```bash
(
    cd ../QuSpin
    CC=gcc CXX=g++ CFLAGS="-O2 -g0" CXXFLAGS="-O2 -g0 -std=c++14" \
        .venv/bin/python setup.py build --parallel 4 --omp bdist_wheel
)
```

For Clang, replace `CC=gcc CXX=g++` with `CC=clang CXX=clang++`. Reduce `--parallel 4` if memory is limited. Avoid `-march=native` when the wheel will be used on other CPUs.

### Install and run

A rebuilt QuSpin wheel has a new checksum. Register it before syncing; the first two commands update only its dependency entry without installing packages:

```bash
uv remove --no-sync quspin
uv add --no-sync ../QuSpin/dist/quspin-0.3.7-cp39-cp39-linux_x86_64.whl
uv sync --locked
uv run python -c 'import jax, quspin, quantax; print(jax.devices())'
```

Keep the wheel in `../QuSpin/dist/`. Its `linux_x86_64` tag does not guarantee portability across distributions: reuse it only with compatible Python ABI, CPU instructions, glibc/libstdc++, and OpenMP runtime. Rebuild on the target machine otherwise. Once registered, later installs use `uv sync --locked` directly.

### CPU or GPU execution

| Execution target | Requirements |
| --- | --- |
| CPU | Use `JAX_PLATFORMS=cpu`. The supplied lockfile still installs CUDA packages; a smaller CPU environment needs its own dependency configuration. |
| NVIDIA GPU | The GPU, driver, and every locked CUDA/cuDNN library must be mutually compatible. Install the driver separately; CUDA user-space libraries come from uv packages. |
| GPU requiring a newer CUDA/cuDNN stack | Use CPU execution or migrate Python, JAX, Equinox, and the affected Quantax APIs together. The existing lockfile does not cover that migration. |
| Other accelerators | Use CPU execution or configure and validate a separate JAX backend. |

Check the NVIDIA [cuDNN 8 support matrix](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-897/support-matrix/index.html) and [driver compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/why-cuda-compatibility.html). A newer driver can run older CUDA libraries; the CUDA version shown by `nvidia-smi` does not select the JAX wheel. For WSL2, install the GPU driver on Windows and follow NVIDIA's [WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

GPU discovery alone is insufficient. Check a compiled operation before training; `JAX_PLATFORMS=cuda` makes GPU initialization failures explicit:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python - <<'PY'
import jax
import jax.numpy as jnp

print(jax.devices())
x = jnp.ones((1, 1, 8, 8))
y = jax.lax.conv_general_dilated(x, jnp.ones((1, 1, 3, 3)), (1, 1), "VALID")
print(y.block_until_ready())
PY
```

Then run `uv run python examples/train.py`, or `JAX_PLATFORMS=cpu uv run python examples/train.py` for CPU execution. Adjust `max_parallel` and sample counts to available memory. In a notebook editor, select `.venv/bin/python` as the kernel; `ipykernel` is included.

Changing the JAX extra to `cuda13` is insufficient with JAX 0.4.25. Newer backends require a matching dependency stack and migration of APIs such as `PositionalSharding`; see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html). This repository has not validated that migration.

### Troubleshooting

| Failure | Action |
| --- | --- |
| Missing headers, `libstdc++`, or OpenMP libraries | Install/load the matching development packages and runtime; rebuild QuSpin with the selected toolchain. |
| Boost errors involving C++ standard features | Check the explicit `-std=` option in QuSpin as well as the compiler and Boost versions. Use compatible versions together. |
| `GLIBC_*` / `GLIBCXX_*` version errors or incompatible wheels | Rebuild local extensions or use a compatible glibc-based container. Rebuilding QuSpin alone does not fix incompatible third-party wheels. |
| `Illegal instruction` | Check the CPU requirements of Python/JAX and any locally compiled extensions; a source build with suitable CPU targets may be needed. |
| CUDA/cuDNN loading or version errors | Check GPU/driver/library compatibility and whether `LD_LIBRARY_PATH` overrides the packaged CUDA libraries. Preserve required cluster runtime paths. |
| Downloads fail | Configure uv's index and the site's proxy settings. The historical JAX wheel archive must also be reachable. |

For `cannot enable executable stack` with the historical JAX wheel, install a `patchelf` version supporting `--clear-execstack`, then run:

```bash
for library in .venv/lib/python3.9/site-packages/jaxlib/xla_extension.so .venv/lib/python3.9/site-packages/jaxlib/cuda/*.so; do
    cp "$library" "$library.local"
    mv "$library.local" "$library"
    patchelf --clear-execstack "$library"
done
```

Repeat this step if JAX is reinstalled. The copy keeps the patch separate from uv's shared cache.

### Parallelism
- Quantax can work when all devices are on the same node
- Multiple-node compatibility is to be implemented

## Questions?
For any questions, bugs, or suggestions, please raise an issue in GitHub.
