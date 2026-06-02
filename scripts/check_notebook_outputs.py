#!/usr/bin/env python3
"""Fail if a docs/source notebook looks stripped of its outputs.

The documentation site renders the notebooks' STORED outputs
(``nbsphinx_execute = "never"``), so a notebook committed without outputs would
publish blank cells while CI stays green. We flag a notebook only when it has
real code cells but *no* stored outputs at all (the signature of an accidental
``Clear All Outputs`` / nbstripout), which avoids false positives on the many
individual code cells that legitimately produce nothing.
"""

from __future__ import annotations

import json
import pathlib
import sys

SOURCE = pathlib.Path("docs/source")

bad: list[str] = []
for nb_path in sorted(SOURCE.glob("*.ipynb")):
    nb = json.loads(nb_path.read_text())
    code_cells = [
        c
        for c in nb.get("cells", [])
        if c.get("cell_type") == "code" and "".join(c.get("source", [])).strip()
    ]
    if not code_cells:
        continue
    total_outputs = sum(len(c.get("outputs", [])) for c in code_cells)
    if total_outputs == 0:
        bad.append(
            f"{nb_path}: {len(code_cells)} code cells, 0 stored outputs "
            "(looks stripped — the site would publish blank cells)"
        )

if bad:
    print("Notebooks missing stored outputs:")
    print("\n".join("  " + b for b in bad))
    sys.exit(1)

print("OK: all docs/source notebooks have stored outputs.")
