# PCI Paper — Computational Notebooks

Companion code and notebooks for the paper *Probabilistic Causal Inference with OBCB and PCI*.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and [Git LFS](https://git-lfs.com/).

```bash
git lfs install        # once per machine — pulls the cached experiment data
git clone <repo-url>
cd explainable_paper
uv sync --group docs   # creates .venv with runtime + documentation deps
```

> **Git LFS is required.** Large experiment caches (`*.pkl`, `*.pt`, `*.npz`)
> are stored via Git LFS. Without `git lfs install` they arrive as small text
> pointers and the notebooks will fail to load their cached results.

## Running notebooks

```bash
source .venv/bin/activate
jupyter notebook
```

Notebooks live in `docs/source/`:

- `obcb_computations.ipynb` — OBCB computations and tables
- `signal_mediation_computations.ipynb` — signal mediation examples
- `actual_causality_benchmark.ipynb`, `sir_benchmark.ipynb`,
  `gradient_based_attribution.ipynb`, `desert_traveler.ipynb`,
  `responsibility_archetypes.ipynb` — the remaining experiments

## Development tasks

All targets run through `uv run`, so no manual `.venv` activation is needed:

| Command | What it does |
|---|---|
| `make lint` | `ruff` + `mypy` on `pci/`, `tests/`, and notebooks (via `nbqa`) — non-mutating |
| `make format` | auto-fix and reformat the same sources in place |
| `make remove-imports` | strip unused imports (`make format` leaves them) |
| `make test` | run `pytest` with coverage (terminal + HTML report) |
| `make coverage` | open the HTML coverage report |
| `make html` | build the Sphinx docs site (`docs/build/html/`) |
| `make notebooks-smoke` | execute every notebook under `CI=1` (fast smoke budgets) |
| `make serve-docs` | serve the built docs at `localhost:8000` |
| `make main` / `make main-clean` | build / clean the LaTeX paper |

## Documentation website

The HTML docs are built from the notebooks' stored outputs
(`nbsphinx_execute = "never"`) and deployed to GitHub Pages by CI on every push
to `main`. To build locally:

```bash
make html        # output in docs/build/html/, open index.html
make clean html  # rebuild from scratch
```

## Paper sources & Overleaf

The LaTeX paper (`main.tex`, `sections/`, `figures/`, `references.bib`) is
edited collaboratively on Overleaf, which syncs with the lean **`overleaf`**
branch of this repo. Prose edits flow back into `main`'s `sections/`; see
`scripts/sync-overleaf.sh`. Day-to-day code and notebook work happens on `main`.
