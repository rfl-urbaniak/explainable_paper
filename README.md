# PCI Paper — Computational Notebooks

Companion notebooks for the paper *Probabilistic Causal Inference with OBCB and PCI*.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group docs
```

This creates a `.venv` with all runtime and documentation dependencies.

## Running notebooks

Activate the environment and launch Jupyter:

```bash
source .venv/bin/activate
jupyter notebook
```

Notebooks live in `docs/source/`:

- `obcb_computations.ipynb` — OBCB computations and tables
- `signal_mediation_computations.ipynb` — Signal mediation examples

## Building HTML docs

```bash
cd docs
make html
```

Output lands in `docs/build/html/`. Open `docs/build/html/index.html` in a browser.

To rebuild from scratch:

```bash
cd docs
make clean html
```
