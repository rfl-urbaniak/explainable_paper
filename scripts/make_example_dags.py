#!/usr/bin/env python3
r"""Paper-ready, colour-blind-safe causal DAGs for the three running examples
(OBCB loan model, signal-with-mediation, Pearl's desert traveller).

Design notes
------------
* Palette is shared with scripts/make_structure_map.py (teal / gold / ink),
  so the example figures read as a family with the "you are here" map.
* Colour is NEVER the sole channel. Each node role is encoded redundantly by
  (1) fill colour, (2) border colour + weight, (3) SHAPE, and (4) a role word.
  Roles therefore survive greyscale / CB conversion -- verified by --grayscale.
    - suspect / candidate cause -> teal  rounded box
    - witness / mediator        -> gold  hexagon      (the node PCI holds fixed)
    - outcome                   -> ink   double box
    - intermediate              -> grey  rounded box
    - exogenous noise           -> grey  dashed circle
* Exogenous noise is drawn explicitly: every STOCHASTIC mechanism gets a small
  dashed noise circle U feeding it; deterministic mechanisms stay bare, so the
  figure shows at a glance which equations carry their own randomness.
* Node boxes size themselves to their label (words always fit); descriptive
  glosses sit OUTSIDE the node as small captions so they never overflow.
* Each example is emitted as a standalone PDF (\includegraphics) and a PNG
  (preview). A combined preview PNG shows all three side by side.
"""

import argparse
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# NB: we do NOT force the Agg backend at import time, so that notebooks can
# `from scripts.make_example_dags import fig_obcb` and render inline. The CLI
# entry point below switches to Agg explicitly for headless PDF generation.

# ---------- palette (shared with make_structure_map.py) ----------
TEAL = "#1F9AA6"
GOLD = "#F08A00"
INK = "#1d2730"
SUS_FILL = "#e1f3f4"; SUS_EDGE = TEAL
WIT_FILL = "#fff2e0"; WIT_EDGE = GOLD
OUT_FILL = "#edeef7"; OUT_EDGE = "#5f5880"
INT_FILL = "#f3f5f6"; INT_EDGE = "#9aa6ae"
NOI_FILL = "#f7f9fa"; NOI_EDGE = "#9aa6ae"
EDGE_COL = "#5a6873"
CAP_COL = "#6b7780"

FONT = "DejaVu Sans"
plt.rcParams.update({
    "font.family": FONT,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,   # embed TrueType (vector, selectable) -- not Type 3
    "ps.fonttype": 42,
    "pdf.compression": 9,
})

ROLE_STYLE = {
    "suspect":      dict(fc=SUS_FILL, ec=SUS_EDGE, lw=2.0, shape="box"),
    "witness":      dict(fc=WIT_FILL, ec=WIT_EDGE, lw=2.2, shape="hex"),
    "outcome":      dict(fc=OUT_FILL, ec=OUT_EDGE, lw=2.4, shape="double"),
    "intermediate": dict(fc=INT_FILL, ec=INT_EDGE, lw=1.5, shape="box"),
    "noise":        dict(fc=NOI_FILL, ec=NOI_EDGE, lw=1.4, shape="circle"),
}


def _size_for(label):
    lines = label.split("\n")
    maxchars = max(len(s) for s in lines)
    hw = max(0.40, 0.072 * maxchars + 0.16)
    hh = 0.26 + 0.17 * (len(lines) - 1)
    return hw, hh


def draw_node(ax, x, y, label, role, gloss=None, gloss_pos="below"):
    """Draw a node, return its patch (for exact arrow clipping)."""
    st = ROLE_STYLE[role]
    shape = st["shape"]
    hw, hh = _size_for(label)
    ehw, ehh = hw, hh  # effective half-extents for gloss placement
    patch = None
    if shape in ("box", "double"):
        patch = mpatches.FancyBboxPatch(
            (x - hw, y - hh), 2 * hw, 2 * hh,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=st["lw"], edgecolor=st["ec"], facecolor=st["fc"], zorder=4)
        ax.add_patch(patch)
        if shape == "double":
            ax.add_patch(mpatches.FancyBboxPatch(
                (x - hw - 0.055, y - hh - 0.055), 2 * (hw + 0.055), 2 * (hh + 0.055),
                boxstyle="round,pad=0.02,rounding_size=0.10",
                linewidth=st["lw"], edgecolor=st["ec"], facecolor="none", zorder=4))
    elif shape == "hex":
        r = max(hw, hh) * 1.35
        ehw = ehh = r * 0.92  # flat-to-flat is a touch under the circumradius
        patch = mpatches.RegularPolygon(
            (x, y), numVertices=6, radius=r, orientation=0.0,
            linewidth=st["lw"], edgecolor=st["ec"], facecolor=st["fc"], zorder=4)
        ax.add_patch(patch)
    elif shape == "circle":
        rc = max(hh, 0.17)
        ehw = ehh = rc
        patch = mpatches.Circle(
            (x, y), radius=rc, linewidth=st["lw"], linestyle=(0, (3, 2)),
            edgecolor=st["ec"], facecolor=st["fc"], zorder=4)
        ax.add_patch(patch)
    fs = 11.0 if "\n" not in label else 10.0
    ax.text(x, y, label, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=INK, zorder=6, linespacing=0.95)
    if gloss:
        dx, dy, ha = {
            "below": (0, -ehh - 0.20, "center"),
            "above": (0, ehh + 0.20, "center"),
            "left":  (-ehw - 0.18, 0, "right"),
            "right": (ehw + 0.18, 0, "left"),
        }[gloss_pos]
        ax.text(x + dx, y + dy, gloss, ha=ha, va="center", fontsize=7.4,
                color=CAP_COL, zorder=6, linespacing=0.95, style="italic")
    return patch, (x, y)


def draw_noise(ax, target, x, y, sub):
    """Small dashed exogenous-noise circle feeding target = (patch, center)."""
    tp, tc = target
    p, c = draw_node(ax, x, y, f"$U_{{{sub}}}$", "noise")
    ax.add_patch(FancyArrowPatch(
        posA=c, posB=tc, patchA=p, patchB=tp,
        connectionstyle="arc3,rad=0.0", arrowstyle="-|>", mutation_scale=10,
        lw=1.2, color=NOI_EDGE, linestyle=(0, (2, 2)), shrinkA=2, shrinkB=2, zorder=2))


def draw_edge(ax, a, b, rad=0.0, style="-", color=EDGE_COL, lw=1.6):
    pa, ca = a
    pb, cb = b
    ax.add_patch(FancyArrowPatch(
        posA=ca, posB=cb, patchA=pa, patchB=pb,
        connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=13,
        lw=lw, color=color, linestyle=style, shrinkA=3, shrinkB=4, zorder=2))


def legend(ax, x, y):
    items = [("suspect", "candidate cause"), ("witness", "witness / mediator"),
             ("outcome", "outcome"), ("intermediate", "intermediate"),
             ("noise", "exogenous noise")]
    for i, (role, lab) in enumerate(items):
        yy = y - i * 0.46
        st = ROLE_STYLE[role]
        if st["shape"] == "hex":
            ax.add_patch(mpatches.RegularPolygon((x, yy), 6, radius=0.15,
                edgecolor=st["ec"], facecolor=st["fc"], lw=1.6, zorder=4))
        elif st["shape"] == "circle":
            ax.add_patch(mpatches.Circle((x, yy), 0.12, edgecolor=st["ec"],
                facecolor=st["fc"], lw=1.4, linestyle=(0, (2, 2)), zorder=4))
        else:
            ax.add_patch(mpatches.FancyBboxPatch((x - 0.15, yy - 0.10), 0.30, 0.20,
                boxstyle="round,pad=0.01,rounding_size=0.05",
                edgecolor=st["ec"], facecolor=st["fc"], lw=st["lw"], zorder=4))
        ax.text(x + 0.30, yy, lab, ha="left", va="center", fontsize=8.0, color=CAP_COL)


# ---------------- OBCB ----------------
def fig_obcb(ax):
    P = {}
    P["gender"] = draw_node(ax, 0.0, 4.2, "gender", "suspect")
    P["credit"] = draw_node(ax, 3.4, 4.2, "credit", "suspect")
    P["check"] = draw_node(ax, 0.0, 2.7, "check", "intermediate")
    P["cf"] = draw_node(ax, 2.0, 1.45, "check\nfailed", "witness", "bottleneck", "right")
    P["lic"] = draw_node(ax, -0.6, 0.1, "loan if\nchecked", "intermediate")
    P["loan"] = draw_node(ax, 2.7, -1.2, "loan", "outcome", "denied", "right")
    draw_noise(ax, P["gender"], -1.35, 4.2, "g")
    draw_noise(ax, P["credit"], 4.75, 4.2, "c")
    draw_noise(ax, P["check"], -1.55, 2.7, r"\mathrm{ch}")
    draw_noise(ax, P["lic"], -2.15, 0.1, "\\ell")
    E = [("gender", "check", 0.0), ("gender", "lic", -0.30),
         ("credit", "cf", 0.16), ("check", "cf", 0.0),
         ("check", "loan", -0.28), ("cf", "lic", 0.0), ("lic", "loan", 0.0)]
    for a, b, r in E:
        draw_edge(ax, P[a], P[b], rad=r)
    ax.set_xlim(-2.6, 6.4); ax.set_ylim(-2.1, 4.9)
    legend(ax, 4.3, 1.2)


# ---------------- signal ----------------
def fig_signal(ax):
    P = {}
    P["X"] = draw_node(ax, 0.0, 0.0, "X", "suspect", "signal", "below")
    P["M"] = draw_node(ax, 2.7, 0.0, "M", "witness", "mediator", "below")
    P["Y"] = draw_node(ax, 5.4, 0.0, "Y", "outcome", "prediction", "right")
    draw_noise(ax, P["X"], 0.0, 1.25, "X")
    draw_noise(ax, P["M"], 2.7, 1.25, "M")
    draw_noise(ax, P["Y"], 5.4, 1.25, "Y")
    draw_edge(ax, P["X"], P["M"], rad=0.0)
    draw_edge(ax, P["M"], P["Y"], rad=0.0)
    # ABSENT direct path X->Y: faded dashed arc below, marked with an x
    ax.add_patch(FancyArrowPatch(
        posA=P["X"][1], posB=P["Y"][1], patchA=P["X"][0], patchB=P["Y"][0],
        connectionstyle="arc3,rad=0.42", arrowstyle="-|>", mutation_scale=12,
        lw=1.4, color="#c2ccd2", linestyle=(0, (4, 3)), shrinkA=4, shrinkB=4, zorder=1))
    mx, my = 2.7, -1.30
    ax.plot([mx - 0.15, mx + 0.15], [my - 0.12, my + 0.12], color="#b0bcc4", lw=2.0, zorder=3)
    ax.plot([mx - 0.15, mx + 0.15], [my + 0.12, my - 0.12], color="#b0bcc4", lw=2.0, zorder=3)
    ax.text(2.7, -1.92, "no direct path  (X reaches Y only through M)",
            ha="center", va="center", fontsize=8.2, style="italic", color="#9aa6ae")
    ax.set_xlim(-1.2, 9.8); ax.set_ylim(-2.4, 1.9)
    legend(ax, 7.4, 1.1)


# ---------------- desert traveller ----------------
def fig_desert(ax):
    P = {}
    P["X"] = draw_node(ax, 0.0, 3.0, "X", "suspect", "empties\ncanteen", "left")
    P["u"] = draw_node(ax, 2.4, 3.2, "u", "noise", "order coin", "above")
    P["P"] = draw_node(ax, 4.8, 3.0, "P", "suspect", "poisons\nwater", "right")
    P["c"] = draw_node(ax, 1.2, 1.2, "c", "witness", "cyanide\npath", "left")
    P["d"] = draw_node(ax, 3.6, 1.2, "d", "witness", "thirst\npath", "right")
    P["y"] = draw_node(ax, 2.4, -0.7, "y", "outcome", "death", "below")
    draw_noise(ax, P["X"], -0.8, 4.1, "X")
    draw_noise(ax, P["P"], 5.6, 4.1, "P")
    E = [("P", "c", 0.0), ("X", "c", 0.10), ("u", "c", 0.0),
         ("X", "d", 0.0), ("u", "d", 0.0), ("P", "d", -0.10),
         ("c", "y", 0.08), ("d", "y", -0.08)]
    for a, b, r in E:
        draw_edge(ax, P[a], P[b], rad=r)
    ax.set_xlim(-2.6, 7.6); ax.set_ylim(-2.2, 4.7)
    legend(ax, 6.2, 1.9)


PANELS = {"obcb": fig_obcb, "signal": fig_signal, "desert": fig_desert}
TITLES = {"obcb": "OBCB (loan)", "signal": "Signal w/ mediation", "desert": "Desert traveller"}


def emit_one(name, outdir):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.axis("off"); ax.set_aspect("equal")
    PANELS[name](ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"dag_{name}.{ext}"), dpi=200,
                    bbox_inches="tight", pad_inches=0.3, facecolor="white")
    plt.close(fig)


def emit_combined(outdir):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    for ax, name in zip(axes, ["obcb", "signal", "desert"]):
        ax.axis("off"); ax.set_aspect("equal")
        PANELS[name](ax)
        ax.set_title(TITLES[name], fontsize=12, color=INK, pad=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "dag_preview_all.png"), dpi=150,
                bbox_inches="tight", pad_inches=0.3, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    plt.switch_backend("Agg")  # headless PDF/PNG generation
    os.makedirs(args.outdir, exist_ok=True)
    for name in PANELS:
        emit_one(name, args.outdir)
    emit_combined(args.outdir)
    print("wrote dag_{obcb,signal,desert}.{pdf,png} + dag_preview_all.png to", args.outdir)
