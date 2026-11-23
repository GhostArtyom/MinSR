# MinSR

MinSR is the reproduction repository for *Empowering deep neural quantum states through efficient optimization*. The importable packages are `quantax` for JAX and `quantax_torch` for PyTorch. `MinSR` is the optimizer. Use the APIs in this checkout.

## Global Preferences

### Editing

- Search with `rg`/`rg --files`; make manual edits with `apply_patch`, not Python replacement scripts.
- Keep changes scoped, preserve unrelated user edits, run appropriate checks, and review the final diff.
- Use f-strings for Python log messages that interpolate values. Use plain strings for literal messages.

### Writing and Mathematics

- Keep prose paragraphs on one physical line. Leave one space after a bold label ending in punctuation.
- In documentation, round durations under one hour to whole minutes. For durations of at least one hour, round hours to one decimal place. Above 24 hours, also show days rounded to one decimal place in ASCII parentheses. Omit the trailing `.0` from rounded hours and days. Use seconds for per-step or per-round rates.
- Use ASCII `/` (U+002F) for all forward slashes you write, including in Chinese prose, headings, tables, chat replies, paths, links, URLs, commands, and code. Do not use the fullwidth slash (U+FF0F). Preserve exact quotations and verbatim source text.
- Add one space between Chinese characters and adjacent English letters or numbers, never between Chinese characters, beside Chinese punctuation, or at line boundaries. Check both boundaries of edited Markdown links using the adjacent visible characters, ignoring link markup.
- Use `$...$` and `$$...$$` for math, not `\(...\)` or `\[...\]`. Leave standalone numbers in prose, tables, and time ranges as plain text; use math delimiters for variables and operations.
- Use `\ket{...}` and `\bra{...}` for bra-ket notation; include `\usepackage{braket}` in LaTeX preambles.
- In Markdown math, use `\mathrm{...}` for upright Roman text. It controls font style, not operator semantics; do not mechanically replace LaTeX `\operatorname{...}` with it.

## Repository guide

- The JAX implementation is in [`quantax/`](quantax/) and the PyTorch port is in [`quantax_torch/`](quantax_torch/). Start with the [JAX MinSR code](quantax/optimizer/minsr.py), the [JAX TDVP code](quantax/optimizer/tdvp.py), or the [PyTorch MinSR code](quantax_torch/optimizer/minsr.py).
- Both pipelines run ordinary training, symmetry training, and fixed-weight measurement. Edit the configuration block in [`examples_torch/run_pipeline.py`](examples_torch/run_pipeline.py) and run `.venv/bin/python -m examples_torch.run_pipeline` for PyTorch. For JAX, edit [`examples/run_pipeline.py`](examples/run_pipeline.py) and run `environments/py311-jax062/.venv/bin/python -m examples.run_pipeline`. When running symmetry training separately, pass `--param-file` and match the network configuration to the checkpoint.
- [`params/`](params/), [`data/`](data/), and [`images/`](images/) contain bundled checkpoints, data, and figures. Keep generated logs, plots, and checkpoints under the ignored `runs/` directory rather than overwriting these assets.
- See the [PyTorch migration guide](docs/pytorch-migration-plan.md) and [JAX project overview](docs/project-overview.md) for the workflows. Start with the [tutorial guide](docs/tutorial-guide.md) for smaller systems. Update the relevant guide when commands, inputs, or outputs change. Link file names in documentation to the actual file, an archived example, or the code that generates it.

## Run directories and Git

- JAX writes to `runs/<system>/<timestamp>/` by default, and PyTorch writes to `runs/torch/<system>/<timestamp>/`. Use `runs/<system>/<timestamp>/` for shared archives. Put any pipeline console log and PID file in the first stage's run directory. Use separate temporary directories only for debugging or smoke tests.
- When starting a run that writes logs, tell the user the log location as soon as it is available. For pipelines, share the console log and report each stage's log when it starts.
- Keep `runs/` ignored by default. When asked to commit results from completed runs, force-add the final checkpoint from each selected training stage in its native format (`.eqx` for JAX, `.pt` for PyTorch), the stage log, and the PDF plots. For completed measurements, include `measurement.log`, `measurement.csv`, `summary.json`, and any PDFs. The input checkpoint belongs in its training archive.
- Leave launcher logs, PID files, intermediate checkpoints, other CSV/JSON outputs, and smoke or benchmark runs untracked. Add files by explicit path, not the entire run directory.

## Environments and checks

- Run commands from the repository root. Follow the [README installation instructions](README.md#installation) to set up PyTorch in the root `.venv/`. The [Python 3.9 / JAX 0.4.25](environments/py39-jax0425/) and [Python 3.11 / JAX 0.6.2](environments/py311-jax062/) projects each have a local `.venv/` and an independent lockfile. Update only the lockfile affected by a dependency change.
- For JAX numerical changes, run `JAX_PLATFORMS=cpu environments/py311-jax062/.venv/bin/python -m scripts.check_jax`. Use `JAX_PLATFORMS=cuda` for CUDA checks. The script checks JAX kernels and performs two small MinSR updates. [`scripts/check_jax.py`](scripts/check_jax.py)
- For PyTorch numerical changes, use the fixed-input JAX comparison or CUDA checks as appropriate. Use a small targeted run for example changes. [CPU comparison commands](docs/pytorch-migration-plan.md#数值验收) · [CUDA check command](docs/implementation-improvements.md#验证命令) · [`scripts/check_torch.py`](scripts/check_torch.py)

## Numerical conventions

- For reproduction and implementation optimizations, preserve the reference MinSR centering, pseudoinverse, fixed step size, and sampling rules. Compare changes on fixed weights and spin configurations when checking numerical agreement.
- `Heisenberg` uses Pauli matrices. At identical couplings, its energy scale is four times that of the corresponding spin-operator convention. Check this normalization and distinguish total energy from energy per site before comparing results. `msr=True` flips the transverse exchange on first-neighbor bonds only.
- `optimizer.VarE` is a centered local-energy variance estimate. The second value from `H.expectation(..., return_var=True)` is the uncentered second moment, not the variance. Neither is the standard error of the energy estimate.
- The active lattice and numerical defaults are process-wide. Use a fresh process or restart the notebook kernel when switching systems or numerical configurations. `max_parallel` limits forward batch size, not all memory used for derivatives and MinSR solves.

## Paper references

Paper materials are in [`docs/Chen-2024-MinSR/`](docs/Chen-2024-MinSR/). Use the published PDF to verify equations, figures, and reproduction details. The Markdown is useful for searching, and the Chinese analysis is a reading guide.

- [Published paper (PDF)](docs/Chen-2024-MinSR/paper.pdf)
- [Searchable full text (Markdown)](docs/Chen-2024-MinSR/paper.md)
- [Chinese paper analysis](docs/Chen-2024-MinSR/analysis.md)

When citing a paper figure in repository documentation, use the existing assets in [`docs/Chen-2024-MinSR/images/`](docs/Chen-2024-MinSR/images/) when available. Embed or link the relevant image with a relative path and identify the original figure or panel in the caption.

The [README](README.md) also links to the publisher's version.
