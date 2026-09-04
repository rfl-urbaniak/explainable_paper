# PCI Paper

This repo contains the sources for *A Computationally Feasible Framework for
Causal Probabilistic Explanation*, the `pci` Python package its experiments
run on, and the companion notebooks that reproduce its results. Every push to
`main` rebuilds the documentation website from the notebooks; 
see [Workflow](#workflow) if you plan to contribute.

The documentation website is at <https://rfl-urbaniak.github.io/explainable_paper/>.
The paper is on arXiv at <http://arxiv.org/abs/2609.04177>.


## Overall structure

| Object | Location |
|---|---|
| Code + notebooks + paper sources | GitHub: `rfl-urbaniak/explainable_paper`, branch `main` |
| Paper (arXiv) | <http://arxiv.org/abs/2609.04177> |
| Documentation website | <https://rfl-urbaniak.github.io/explainable_paper/> |
| CI (tests, notebook smoke, docs deploy) | GitHub → Actions tab |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and [Git LFS](https://git-lfs.com/).

```bash
git lfs install        # once per machine; pulls the cached experiment data
git clone https://github.com/rfl-urbaniak/explainable_paper.git
cd explainable_paper
uv sync --group docs   # creates .venv with runtime + documentation deps
```

> Run `git lfs install` before cloning. The large experiment caches (`*.pkl`,
> `*.pt`, `*.npz`) come through Git LFS, and without it they arrive as small
> text pointers, so the notebooks fail to load their cached results.

## Repository layout

```
main.tex, sections/, references.bib, neurips_2024.sty   the paper
figures/                                                paper figures
pci/                                                    the Python package
tests/                                                  pytest suite
docs/source/*.ipynb                                     companion notebooks
docs/  (Makefile, conf.py)                              Sphinx site config
scripts/                                                lint/format/test/sync/arxiv helpers
LICENSE.md                                              Apache-2.0, Basis Research Institute
arxiv/, arxiv.tar.gz                                    generated submission tree (gitignored)
.github/workflows/                                      CI (ci.yml, docs.yml)
```

## Development tasks

All targets run through `uv run`, so you never activate `.venv` manually:

| Command | What it does |
|---|---|
| `make lint` | `ruff` + `mypy` on `pci/`, `tests/`, and notebooks (via `nbqa`); does not mutate |
| `make format` | auto-fix and reformat the same sources in place |
| `make remove-imports` | strip unused imports (`make format` leaves them) |
| `make test` | run `pytest` with coverage (terminal + HTML report) |
| `make coverage` | open the HTML coverage report |
| `make html` | build the Sphinx docs site into `docs/build/html/` |
| `make serve-docs` | serve the built docs at `localhost:8000` |
| `make notebooks-smoke` | execute every notebook under `CI=1` (fast smoke budgets) |
| `make deploy` | publish the site from whatever branch you have checked out (must already be pushed), bypassing `main`; the next merge into `main` overwrites it |
| `make main` / `make main-clean` | build / clean the LaTeX paper |
| `make arxiv` | build and verify `arxiv/` + `arxiv.tar.gz` for submission |

---

## Workflow

```
   VS Code edits on a branch            PR merged into main
  (code, notebooks, paper)  ───────────────────────────────────►  main
                                                                     │
                                                                     ▼
                                     GitHub Actions (.github/workflows/docs.yml)
                                                                     │
                                                                     ▼
                            Website  https://rfl-urbaniak.github.io/explainable_paper/
```

A ruleset on `main` blocks direct pushes for everyone, including admins: every
change lands through a pull request, squash-merged, with one required
approving review.

## License

Apache License 2.0, Copyright 2026 Basis Research Institute; see
[LICENSE.md](LICENSE.md).
