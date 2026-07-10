#!/usr/bin/env python3
"""Sync the running-example notebooks to the canonical paper DAGs.

Each of the three notebooks gets a self-contained cell that imports the SAME
fig_* helper used to generate the paper PDFs (scripts/make_example_dags.py), so
the notebooks and the paper can never drift. Because the docs are built with
nbsphinx_execute="never", we render the cell headlessly here and store the PNG
as the cell output -- that stored output is what the HTML docs display.

Run from the repo root after editing make_example_dags.py:
    python scripts/sync_notebook_dags.py
"""
import base64
import io
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat

REPO = pathlib.Path(__file__).resolve().parent.parent
NBDIR = REPO / "docs" / "source"

CELL_TMPL = '''\
# Causal DAG for the {desc}.
#
# Rendered with the SAME helper that generates the paper figure
# (scripts/make_example_dags.py): single source of truth, so this notebook and
# the paper never drift. The rendering is colour-blind-safe -- each role is
# encoded by SHAPE (box / hexagon / double box / dashed circle) as well as
# colour, and dashed U-circles mark the exogenous noise on each stochastic
# mechanism.
import sys, pathlib
import matplotlib.pyplot as plt

_p = pathlib.Path.cwd()
for _ in range(8):
    if (_p / "scripts" / "make_example_dags.py").exists():
        sys.path.insert(0, str(_p)); break
    _p = _p.parent
from scripts.make_example_dags import {fn}  # noqa: E402

fig, ax = plt.subplots(figsize={figsize})
ax.axis("off"); ax.set_aspect("equal")
{fn}(ax)
plt.show()
'''

SPECS = {
    "obcb_computations.ipynb": dict(
        desc="OBCB loan model", fn="fig_obcb", figsize="(5.6, 4.8)",
        mode="replace", index=2),
    "signal_mediation_computations.ipynb": dict(
        desc="signal-with-mediation model", fn="fig_signal", figsize="(6.6, 4.2)",
        mode="replace", index=2),
    "desert_traveler.ipynb": dict(
        desc="desert-traveller model", fn="fig_desert", figsize="(5.8, 4.6)",
        mode="insert_after", index=3),
}


def render_png(source):
    ns = {}
    exec(source, ns)
    fig = ns["fig"]
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    for name, spec in SPECS.items():
        path = NBDIR / name
        nb = nbformat.read(str(path), as_version=4)
        src = CELL_TMPL.format(**spec)
        cell = nbformat.v4.new_code_cell(source=src)
        cell["execution_count"] = 1
        cell["outputs"] = [nbformat.v4.new_output(
            "display_data",
            data={"image/png": render_png(src), "text/plain": ["<Figure>"]},
            metadata={})]
        if spec["mode"] == "replace":
            assert nb.cells[spec["index"]].cell_type == "code"
            nb.cells[spec["index"]] = cell
        else:
            # idempotent: skip if the DAG cell is already present just after index
            nxt = nb.cells[spec["index"] + 1] if spec["index"] + 1 < len(nb.cells) else None
            if nxt and "make_example_dags" in nxt.get("source", ""):
                nb.cells[spec["index"] + 1] = cell
            else:
                nb.cells.insert(spec["index"] + 1, cell)
        nbformat.write(nb, str(path))
        print(f"updated {name}")


if __name__ == "__main__":
    main()
