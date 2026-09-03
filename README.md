# PCI Paper

Paper, code, and companion notebooks for *Probabilistic Causal Inference with
OBCB and PCI*.

This repo holds the paper sources, the `pci` Python package, and the
companion notebooks the paper references. Every push to `main` rebuilds the
documentation website from the notebooks; [Workflow](#workflow) covers how
the notebooks, the site, and the arXiv build fit together.

## Where everything lives

| Thing | Location |
|---|---|
| Code + notebooks + paper sources | GitHub: `rfl-urbaniak/explainable_paper`, branch `main` |
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

All targets run through `uv run`, so no manual `.venv` activation is needed:

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
| `make deploy` | manually trigger the GitHub Pages deploy (normally automatic on push) |
| `make main` / `make main-clean` | build / clean the LaTeX paper |
| `make arxiv` | build and verify `arxiv/` + `arxiv.tar.gz` for submission |

---

## Workflow

```
   VS Code edits                       git push origin main
  (code, notebooks, paper)  ─────────────────────────────────►  main
                                                                   │
                                                                   ▼
                                   GitHub Actions (.github/workflows/docs.yml)
                                                                   │
                                                                   ▼
                          Website  https://rfl-urbaniak.github.io/explainable_paper/
```

### 1. Edits → website

CI owns deployment. On every `git push origin main`, GitHub Actions runs
`.github/workflows/docs.yml`, which:

1. installs the `docs` dependencies with `uv` and the `pandoc` binary (needed by
   nbsphinx),
2. runs `make -C docs html` to render the Sphinx site from the notebooks' stored
   outputs (`nbsphinx_execute = "never"`, so CI never re-executes a notebook;
   commit the notebook carrying the outputs you want published), uploading the
   result as a Pages artifact, and
3. publishes the artifact to GitHub Pages (`actions/deploy-pages`).

To update the site, then: edit and run a notebook locally, commit it with its
outputs, and `git push origin main`. The site follows a couple of minutes later.

Publishing to Pages needs GitHub's own deployment credentials (the
`github-pages` environment plus an OIDC token), which exist only inside an
Actions run, so the repo has no `make`-based local deploy. Locally `make html`
builds the site and `make serve-docs` previews it, while CI both builds and
deploys. To re-run a deploy without a code change, use `make deploy`, which
triggers the same workflow through `gh workflow run docs.yml`, or go to the
Actions tab → docs → Run workflow.

The site is at <https://rfl-urbaniak.github.io/explainable_paper/>.

### 2. Paper → arXiv

`make arxiv` assembles everything the paper needs into `arxiv/` at
repo-identical paths, so `main.tex` compiles there unedited, and tars the result
as `arxiv.tar.gz`.

arXiv runs pdflatex and never bibtex, so `main.bbl` ships and `references.bib`
stays behind; the build aborts
when `main.bbl` is older than `references.bib`, since a stale one would silently
publish the wrong references. arXiv also republishes the source tarball, so the
copied `.tex` files lose their whole-line comments and the superseded
definitions and preamble notes stay in the repo, while a trailing `%` survives
because LaTeX reads it as a line continuation. Finally, `\pdfoutput=1` on line 1
of `main.tex` tells arXiv's AutoTeX to run pdflatex, which the PDF figures
require.

The script then compiles the tree in a scratch directory that cannot see the
repo, so a figure nobody copied fails there instead of at arXiv. It also
checks the page count and the undefined-reference count before writing the
tarball.

Upload `arxiv.tar.gz` and take arXiv's default non-exclusive license. You keep
copyright under every option, but a Creative Commons license is irrevocable per
version and some publishers object to one on a preprint, whereas the default
leaves your options open and a later version can still move to CC BY.

## License

Apache License 2.0, Copyright 2026 Basis Research Institute; see
[LICENSE.md](LICENSE.md).

The paper falls under whichever license is chosen on the arXiv submission form,
and the LaTeX style files (`neurips_2024.sty`, `jmlr.cls`, `jmlrutils.sty`) keep
their publishers' terms.
