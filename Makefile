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

# Trigger the automated GitHub Pages deploy (.github/workflows/docs.yml).
# Deployment normally happens automatically on `git push origin main`; this is
# just a manual kick that runs the same workflow. (Pages can only be published
# from CI, so there is no local deploy.)
deploy: FORCE
	gh workflow run docs.yml

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

FORCE:
