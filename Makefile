# Thin entry points; the real work lives in scripts/ and docs/Makefile.
# Everything runs through `uv run`, so no manual `.venv` activation is needed.

# ---- Python: lint / format / test (see scripts/) ----------------------------
lint: FORCE
	./scripts/lint.sh

format: FORCE
	./scripts/format.sh

remove-imports: FORCE
	./scripts/remove_imports.sh

test: FORCE
	./scripts/test.sh

coverage: FORCE
	xdg-open tests/coverage/index.html

# ---- Notebooks & docs website (delegate to docs/Makefile) -------------------
html: FORCE
	$(MAKE) -C docs html

notebooks-smoke: FORCE
	$(MAKE) -C docs notebooks-smoke

serve-docs: FORCE
	$(MAKE) -C docs serve

# Trigger the automated GitHub Pages deploy (.github/workflows/docs.yml),
# building and publishing from whichever branch you're currently on. This
# bypasses the main-only auto-deploy gate: whatever's on the current branch
# goes live immediately, merged or not, so use with care. The branch must
# already be pushed to origin (Actions builds from the remote, not local
# state). (Pages can only be published from CI, so there is no local deploy.)
deploy: FORCE
	gh workflow run docs.yml --ref $$(git branch --show-current)

# ---- Overleaf sync (see scripts/sync-overleaf.sh) ---------------------------
pull-from-overleaf: FORCE
	./scripts/sync-overleaf.sh pull

push-to-overleaf: FORCE
	./scripts/sync-overleaf.sh push

# ---- Paper (LaTeX) ----------------------------------------------------------
main: FORCE
	latexmk -pdf -synctex=1 -interaction=nonstopmode -halt-on-error main.tex

main-clean: FORCE
	latexmk -C main.tex
	rm -f sections/*.aux

# ---- arXiv submission (see scripts/make-arxiv.sh) ---------------------------
arxiv: FORCE
	./scripts/make-arxiv.sh

FORCE:
