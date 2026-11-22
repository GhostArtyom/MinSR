# MinSR

MinSR is the reproduction repository for *Empowering deep neural quantum states through efficient optimization*. The importable Python package is `quantax`; `MinSR` is the optimizer. Work against the API in this checkout rather than assuming compatibility with newer Quantax releases.

## Global Preferences

### Editing

- Search with `rg`/`rg --files`; make manual edits with `apply_patch`, not Python replacement scripts.
- Keep changes scoped, preserve unrelated user edits, run appropriate checks, and review the final diff.

### Writing and Mathematics

- Keep prose paragraphs on one physical line. Leave one space after a bold label ending in punctuation.
- Use ASCII `/` (U+002F) for all forward slashes you write, including in Chinese prose, headings, tables, chat replies, paths, links, URLs, commands, and code. Do not use the fullwidth slash (U+FF0F). Preserve exact quotations and verbatim source text.
- Add one space between Chinese characters and adjacent English letters or numbers, never between Chinese characters, beside Chinese punctuation, or at line boundaries. Check both boundaries of edited Markdown links using the adjacent visible characters, ignoring link markup.
- Use `$...$` and `$$...$$` for math, not `\(...\)` or `\[...\]`. Leave standalone numbers in prose, tables, and time ranges as plain text; use math delimiters for variables and operations.
- Use `\ket{...}` and `\bra{...}` for bra-ket notation; include `\usepackage{braket}` in LaTeX preambles.
- In Markdown math, use `\mathrm{...}` for upright Roman text. It controls font style, not operator semantics; do not mechanically replace LaTeX `\operatorname{...}` with it.

## Repository guide

- [`quantax/`](quantax/) contains the implementation. Start with [`quantax/optimizer/minsr.py`](quantax/optimizer/minsr.py) for MinSR and [`quantax/optimizer/tdvp.py`](quantax/optimizer/tdvp.py) for the shared optimization logic.
- [`examples/`](examples/) contains training, symmetry-constrained continuation, and energy measurement scripts. Edit the configuration block in [`examples/run_pipeline.py`](examples/run_pipeline.py) and run `python -m examples.run_pipeline` without arguments to run all three stages. For standalone `train_symm.py`, pass `--param-file` and match the network configuration to the checkpoint. Start with [`examples/train_tutorial.py`](examples/train_tutorial.py) or the [tutorial notebooks](tutorials/) for smaller systems.
- [`params/`](params/), [`data/`](data/), and [`images/`](images/) contain bundled checkpoints, data, and figures. Keep generated logs, plots, and checkpoints under the ignored `runs/` directory rather than overwriting these assets.
- See the [project overview](docs/project-overview.md) and [tutorial guide](docs/tutorial-guide.md) for the workflows. Update the relevant guide if a change affects documented commands, inputs, or outputs.

## Run directories and Git

- Keep production runs under `runs/<system>/<timestamp>/`. Put any pipeline console log and PID file in the first stage's run directory rather than a separate launcher directory. Use separate temporary directories only for debugging or smoke tests.
- Keep `runs/` ignored by default. When asked to commit results from completed runs, force-add only the final `.eqx` checkpoint from each selected training stage and the corresponding stage log and PDF plots. For completed measurements, include `measurement.log`, `measurement.csv`, `summary.json`, and any PDFs, not a copy of the input checkpoint.
- Leave launcher logs, PID files, intermediate checkpoints, other CSV/JSON outputs, and smoke or benchmark runs untracked. Add files by explicit path, not the entire run directory.

## Environments and checks

- Run commands from the repository root. Follow the [README installation instructions](README.md#installation) to set up the root `.venv/`. The [Python 3.9 / JAX 0.4.25](environments/py39-jax0425/) and [Python 3.11 / JAX 0.6.2](environments/py311-jax062/) environments have independent lockfiles; update only the one affected by a dependency change.
- For changes to numerical code, run `JAX_PLATFORMS=cpu .venv/bin/python -m scripts.check_gpu`. It checks JAX kernels and performs two small MinSR updates. For a CUDA-specific check, run the same command with `JAX_PLATFORMS=cuda`. Use a small targeted run for example changes rather than launching a full production training run.

## Numerical conventions

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
