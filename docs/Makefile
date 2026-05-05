SPHINXBUILD   = uv run --group docs sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help html clean serve notebooks build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)

serve:
	-fuser -k 8000/tcp
	uv run --group docs python -m http.server 8000 --directory $(BUILDDIR)/html
