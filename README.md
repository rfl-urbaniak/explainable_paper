# PCI Paper

Paper, code, and companion notebooks for *Probabilistic Causal Inference with
OBCB and PCI*.

This repo is the single source of truth for everything. The LaTeX paper is
edited collaboratively on **Overleaf**; the code and notebooks are developed in
**VS Code** on this repo; and a **documentation website** is built from the
notebooks and auto-deployed. The three stay in sync as described under
[Workflow](#workflow).

## Where everything lives

| Thing | Location |
|---|---|
| Code + notebooks + paper sources (canonical) | GitHub: `rfl-urbaniak/explainable_paper` (private), branch `main` |
| Lean paper mirror (Overleaf ↔ GitHub) | GitHub branch `overleaf` |
| Live paper editor | Overleaf project: <https://www.overleaf.com/project/69ef4f6d60dfbab2d44fe0d4> |
| Documentation website | <https://rfl-urbaniak.github.io/explainable_paper/> |
| CI (tests, notebook smoke, docs deploy) | GitHub → Actions tab |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and [Git LFS](https://git-lfs.com/).

```bash
git lfs install        # once per machine — pulls the cached experiment data
git clone https://github.com/rfl-urbaniak/explainable_paper.git
cd explainable_paper
uv sync --group docs   # creates .venv with runtime + documentation deps
```

> **Git LFS is required.** Large experiment caches (`*.pkl`, `*.pt`, `*.npz`)
> are stored via Git LFS. Without `git lfs install` they arrive as small text
> pointers and the notebooks fail to load their cached results.

## Repository layout

```
main.tex, sections/, references.bib, neurips_2024.sty   the paper
figures/                                                paper figures
pci/                                                    the Python package
tests/                                                  pytest suite
docs/source/*.ipynb                                     companion notebooks
docs/  (Makefile, conf.py)                              Sphinx site config
scripts/                                                lint/format/test/sync helpers
.github/workflows/                                      CI (ci.yml, docs.yml)
```

## Development tasks

All targets run through `uv run`, so no manual `.venv` activation is needed:

| Command | What it does |
|---|---|
| `make lint` | `ruff` + `mypy` on `pci/`, `tests/`, and notebooks (via `nbqa`) — non-mutating |
| `make format` | auto-fix and reformat the same sources in place |
| `make remove-imports` | strip unused imports (`make format` leaves them) |
| `make test` | run `pytest` with coverage (terminal + HTML report) |
| `make coverage` | open the HTML coverage report |
| `make html` | build the Sphinx docs site into `docs/build/html/` |
| `make notebooks-smoke` | execute every notebook under `CI=1` (fast smoke budgets) |
| `make serve-docs` | serve the built docs at `localhost:8000` |
| `make main` / `make main-clean` | build / clean the LaTeX paper |

---

## Workflow

```
                 scripts/sync-overleaf.sh pull
   Overleaf  ───────────────────────────────────►  main (sections/, main.tex)
  (prose edits)                                        │
       ▲                                               │ VS Code edits
       │  scripts/sync-overleaf.sh push                │ (code, notebooks, paper)
       └───────────────────────────────────────────── ┤
                                                        │
                                            git push origin main
                                                        │
                                                        ▼
                                   GitHub Actions (.github/workflows/docs.yml)
                                                        │
                                                        ▼
                          Website  https://rfl-urbaniak.github.io/explainable_paper/
```

Overleaf is reached through its **Git bridge** (remote `overleaf-bridge`,
branch `master`); GitHub keeps a lean mirror on the `overleaf` branch. Both
hold only the files needed to compile the paper, at paths identical to `main`,
so prose merges back into `main`'s `sections/` are conflict-free.

### 1. Overleaf edits → repo

After collaborators edit prose in Overleaf, bring it back to `main`:

```bash
git checkout main
scripts/sync-overleaf.sh pull     # fetches Overleaf, stages main.tex + sections/
git diff --cached                 # review the prose changes
git commit -m "Overleaf prose edits"
git push origin main
```

`pull` authenticates to Overleaf with your **Git token** (username `git`; create
one in Overleaf → Account Settings → Git integration).

### 2. VS Code edits on the repo → Overleaf

After editing the paper (or regenerating figures) locally on `main`:

```bash
git commit -am "..."              # commit your paper/figure changes on main
scripts/sync-overleaf.sh push     # sends the paper to Overleaf + mirrors to GitHub
```

`push` layers the current paper onto Overleaf's head (Overleaf forbids
force-push) and is a no-op if Overleaf already matches `main`.

> **Discipline:** always `pull` (and commit) **before** you `push`. A push
> sends `main`'s versions to Overleaf, so unmerged Overleaf edits would be
> overwritten. When in doubt, `pull` first.

### 3. VS Code edits → website

The website rebuilds itself — you don't deploy it by hand. On every
`git push origin main`, GitHub Actions runs `.github/workflows/docs.yml`, which:

1. installs the `docs` dependencies with `uv`,
2. runs `make -C docs html` to render the Sphinx site **from the notebooks'
   stored outputs** (`nbsphinx_execute = "never"`, so notebooks are *not*
   re-executed — commit the notebook with the outputs you want published), and
3. deploys `docs/build/html/` to GitHub Pages.

So to update the site: edit/run a notebook locally, commit it **with its
outputs**, and `git push origin main`. The site updates a couple of minutes
later. Watch progress under the **Actions** tab → *docs*; you can also trigger
it manually there (*Run workflow*).

**Access the site at:** <https://rfl-urbaniak.github.io/explainable_paper/>
(public site, served from this private repo via GitHub Pro).

---

## First-time collaborator checklist

1. Install [uv](https://docs.astral.sh/uv/) and [Git LFS](https://git-lfs.com/);
   run `git lfs install`.
2. Get added on GitHub (Settings → Collaborators) and invited on Overleaf (Share).
3. `git clone` → `uv sync --group docs`.
4. Prose lives on Overleaf; code and notebooks on `main`. Use
   `scripts/sync-overleaf.sh` to move paper edits between them.
