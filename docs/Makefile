SPHINXBUILD   = uv run --group docs sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
NBCONVERT     = uv run --group docs jupyter nbconvert
NB_TIMEOUT   ?= 900

NOTEBOOKS = $(wildcard $(SOURCEDIR)/*.ipynb)

.PHONY: help html clean serve notebooks-smoke

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)

serve:
	-fuser -k 8000/tcp
	uv run --group docs python -m http.server 8000 --directory $(BUILDDIR)/html

# Execute every notebook under docs/source with CI=1, which switches each
# notebook into its smoke-test branch (tiny sample sizes / runtime budgets).
# Outputs go to a temp directory so the committed notebooks keep their full
# publication-quality outputs.
notebooks-smoke:
	@set -e; \
	mkdir -p $(BUILDDIR)/smoke; \
	for nb in $(NOTEBOOKS); do \
	  echo "=== smoke: $$nb ==="; \
	  CI=1 $(NBCONVERT) --to notebook --execute "$$nb" \
	    --output-dir $(BUILDDIR)/smoke \
	    --ExecutePreprocessor.timeout=$(NB_TIMEOUT); \
	done
