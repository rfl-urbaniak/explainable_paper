#!/usr/bin/env python3
"""Full-width Section-3 figure illustrating the PCI mechanism (two-worlds contrast).

Panel (a) -- TUNING FORK: a single noise draw u ~ P_U at the apex forks into a
necessity world (do(C=c'), hold T) and a sufficiency world (do(C=c*), hold T);
the two outcomes Y^n, Y^s rejoin at the impact kernel ci, whose expectation over
Gamma (x) Delta (x) P_U is the PCI score. The fork *is* the product
decomposition of Def. (jointnecsuf): given u, the two worlds are conditionally
independent. A small OBCB inset grounds the schematic.

<<<<<<< HEAD
Panel (b) -- OUTCOME PLANE: the kernel ci lives on the (|Y^n - y*|, Y^s) plane.
x = necessity strength (how far the alternative pushes Y from factual),
y = the sufficiency-world outcome, with the factual y* baseline marked. The
shaded field is a continuous ci; dashed lines are the binary N/S/J indicator
boundaries, so the reader sees the continuous kernel subsumes the discrete
PN/PS/PNS indicators. Each fork draw is a point; their mean is E[ci] = score.
=======
Panel (b) -- SIGNED QUADRANT PLANE: the kernel ci lives on the FULL signed plane
(Y^n - y*, Y^s - y*), not the pre-folded |Y^n - y*| used previously. Because
ci = |Y^n-y*| - |Y^s-y*| depends only on the two absolute deviations, the field
is symmetric under an independent sign flip of either axis: four qualitatively
different stories (which direction each counterfactual world moved relative to
the factual) collapse onto the same score whenever the magnitudes match. Four
marked points at matching |x|,|y| in each quadrant make that many-to-one
collapse concrete before any absolute value is taken.
>>>>>>> main

Palette + redundant role encoding shared with make_example_dags.py so the figure
reads as one family. Colour is never the sole channel.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
<<<<<<< HEAD
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
=======
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
>>>>>>> main

# ---------- palette (shared with make_example_dags.py) ----------
TEAL = "#1F9AA6"
GOLD = "#F08A00"
INK = "#1d2730"
SUS_FILL = "#e1f3f4"; SUS_EDGE = TEAL
WIT_FILL = "#fff2e0"; WIT_EDGE = GOLD
OUT_FILL = "#edeef7"; OUT_EDGE = "#5f5880"
NOI_FILL = "#f7f9fa"; NOI_EDGE = "#9aa6ae"
EDGE_COL = "#5a6873"
CAP_COL = "#6b7780"
NEC_BAND = "#eaf4f5"   # cool backing for necessity world
SUF_BAND = "#fdf3e3"   # warm backing for sufficiency world

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "pdf.fonttype": 42, "ps.fonttype": 42})


# ---------------- small drawing helpers ----------------
def rbox(ax, x, y, hw, hh, label, fc, ec, lw=2.0, fs=11, double=False, tc=INK):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x - hw, y - hh), 2 * hw, 2 * hh,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=5))
    if double:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - hw - 0.045, y - hh - 0.045), 2 * (hw + 0.045), 2 * (hh + 0.045),
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=lw, edgecolor=ec, facecolor="none", zorder=5))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=tc, zorder=7, linespacing=0.95)
    return (x, y, hw, hh)


def hexnode(ax, x, y, r, label, fc=WIT_FILL, ec=WIT_EDGE, lw=2.2, fs=10):
    ax.add_patch(mpatches.RegularPolygon(
        (x, y), numVertices=6, radius=r, orientation=0.0,
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=5))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=INK, zorder=7)
    return (x, y, r, r)


def circ(ax, x, y, r, label, fc=NOI_FILL, ec=NOI_EDGE, lw=1.5, fs=10):
    ax.add_patch(mpatches.Circle(
        (x, y), radius=r, linewidth=lw, linestyle=(0, (3, 2)),
        edgecolor=ec, facecolor=fc, zorder=5))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=INK, zorder=7)
    return (x, y, r, r)


def arrow(ax, a, b, rad=0.0, color=EDGE_COL, lw=1.6, style="-", ms=13, z=3):
    ca, cb = (a[0], a[1]), (b[0], b[1])
    ax.add_patch(FancyArrowPatch(
        posA=ca, posB=cb, connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
        linestyle=style, shrinkA=12, shrinkB=12, zorder=z))


# ============================================================
# Panel (a): tuning fork
# ============================================================
def panel_fork(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off"); ax.set_aspect("equal")

    # --- title (raised, with clear space beneath) ---
    ax.text(0.1, 9.9, "(a)  Necessity and sufficiency on a shared draw",
            ha="left", va="center", fontsize=11.5, fontweight="bold", color=INK)

    # --- inputs (left rail), named with the paper's terminology ---
    knob_x = 0.45
    knobs = [(r"$\Gamma$", "variable-selection dist.", 8.8),
             (r"$\Delta$", "alternative-value dist.", 8.4),
             (r"$P_{\mathbf{U}}$", "exogenous-noise law", 8.0)]
    for sym, name, yy in knobs:
        ax.text(knob_x, yy, sym, ha="center", va="center", fontsize=12.0,
                fontweight="bold", color=GOLD, zorder=7)
        ax.text(knob_x + 0.55, yy, name, ha="left", va="center",
                fontsize=8.0, color=CAP_COL, zorder=7)

    # --- node-shape legend (top-right) ---
    lx0, lx1, lyt = 6.45, 9.78, 9.22
    ax.add_patch(mpatches.FancyBboxPatch(
        (lx0, 7.62), lx1 - lx0, 1.78, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0, edgecolor="#d7dde1", facecolor="#fbfcfc", zorder=3))
    ax.text(lx0 + 0.18, lyt, "node shapes", ha="left", va="center", fontsize=7.8,
            fontweight="bold", color="#9aa6ae", zorder=6)
    ix, tx = lx0 + 0.36, lx0 + 0.78
    legrows = [
        ("box", SUS_FILL, SUS_EDGE, r"candidate cause $\mathbf{C}$"),
        ("hex", WIT_FILL, WIT_EDGE, r"witness $\mathbf{T}$ (held)"),
        ("double", OUT_FILL, OUT_EDGE, r"outcome $Y$"),
        ("circ", NOI_FILL, NOI_EDGE, r"exogenous noise $\mathbf{u}$"),
    ]
    for i, (shape, fc, ec, lab) in enumerate(legrows):
        yy = 8.88 - i * 0.36
        if shape == "box":
            ax.add_patch(mpatches.FancyBboxPatch((ix - 0.17, yy - 0.10), 0.34, 0.20,
                boxstyle="round,pad=0.01,rounding_size=0.05",
                linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=6))
        elif shape == "double":
            ax.add_patch(mpatches.FancyBboxPatch((ix - 0.15, yy - 0.09), 0.30, 0.18,
                boxstyle="round,pad=0.01,rounding_size=0.05",
                linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=6))
            ax.add_patch(mpatches.FancyBboxPatch((ix - 0.19, yy - 0.13), 0.38, 0.26,
                boxstyle="round,pad=0.01,rounding_size=0.05",
                linewidth=1.4, edgecolor=ec, facecolor="none", zorder=6))
        elif shape == "hex":
            ax.add_patch(mpatches.RegularPolygon((ix, yy), 6, radius=0.15,
                edgecolor=ec, facecolor=fc, lw=1.8, zorder=6))
        else:
            ax.add_patch(mpatches.Circle((ix, yy), 0.13, edgecolor=ec,
                facecolor=fc, lw=1.4, linestyle=(0, (2, 2)), zorder=6))
        ax.text(tx, yy, lab, ha="left", va="center", fontsize=7.4,
                color=CAP_COL, zorder=6)

    # --- apex: shared noise ---
    U = circ(ax, 5.0, 8.4, 0.52, r"$\mathbf{u}$", fs=12)
    ax.text(3.55, 9.45, r"one shared draw  $\mathbf{u}\sim P_{\mathbf{U}}$",
            ha="center", va="center", fontsize=8.8, color=CAP_COL, style="italic")

    # --- the two worlds (backing bands), taller for breathing room ---
    BTOP, BBOT = 6.55, 3.5

    def world(cx, band, edge, title, sub, cause_lbl, out_lbl, out_sym, mirror=False):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx - 1.95, BBOT), 3.9, BTOP - BBOT,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.5, edgecolor=edge, facecolor=band, zorder=1))
        ax.text(cx, 6.28, title, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=INK, zorder=6)
        ax.text(cx, 5.86, sub, ha="center", va="center", fontsize=8.3,
                color=CAP_COL, zorder=6)
        ax.text(cx, 5.44, out_lbl, ha="center", va="center", fontsize=7.8,
                style="italic", color=CAP_COL, zorder=6)
        # mini-DAG cause -> witness(held) -> outcome; mirror puts the outcome
        # toward the centre so both worlds feed the kernel symmetrically
        s = -1.0 if mirror else 1.0
        c = rbox(ax, cx - 1.25 * s, 4.62, 0.62, 0.40, cause_lbl, SUS_FILL,
                 SUS_EDGE, fs=8.6)
        t = hexnode(ax, cx, 4.62, 0.46, r"$\mathbf{T}$", fs=9)
        o = rbox(ax, cx + 1.25 * s, 4.62, 0.55, 0.40, out_sym, OUT_FILL,
                 OUT_EDGE, fs=10, double=True)
        arrow(ax, c, t, lw=1.5, ms=11)
        arrow(ax, t, o, lw=1.5, ms=11)
        return o

    o_nec = world(2.6, NEC_BAND, "#bfe0e3", "necessity world",
                  r"$\mathrm{do}(\mathbf{C}=\mathbf{c}'),\ \mathrm{hold}\ \mathbf{T}$",
                  r"$\mathbf{C}{=}\mathbf{c}'$", "does removing it change $Y$?", r"$Y^n$")
    o_suf = world(7.4, SUF_BAND, "#f3d6a8", "sufficiency world",
                  r"$\mathrm{do}(\mathbf{C}=\mathbf{c}^\star),\ \mathrm{hold}\ \mathbf{T}$",
                  r"$\mathbf{C}{=}\mathbf{c}^\star$", "does restoring it sustain $Y$?", r"$Y^s$",
                  mirror=True)

    # --- fork edges: apex -> each world ---
    arrow(ax, (U[0], U[1]), (2.6, BTOP), rad=0.12, lw=1.9, ms=14)
    arrow(ax, (U[0], U[1]), (7.4, BTOP), rad=-0.12, lw=1.9, ms=14)

    # --- conditional-independence annotation (off the fork arrows, right side) ---
    ax.text(8.4, 7.05,
            "shared $\\mathbf{u}$:\n$Y^n \\perp Y^s \\mid \\mathbf{u}$",
            ha="center", va="center", fontsize=8.3, style="italic",
            color="#9aa6ae", zorder=6, linespacing=1.3)

    # --- rejoin at the kernel (arrows land on the box's top edge, heads visible) ---
    ci = rbox(ax, 5.0, 2.5, 1.55, 0.50,
              r"$ci(Y^s, Y^n, y^\star)$", "#f3f5f6", "#9aa6ae", lw=1.8, fs=10.5)
    arrow(ax, (o_nec[0], 4.22), (4.55, 3.06), rad=0.05, lw=1.9, ms=14, z=6)
    arrow(ax, (o_suf[0], 4.22), (5.45, 3.06), rad=-0.05, lw=1.9, ms=14, z=6)

    # --- score ---
    arrow(ax, (5.0, 2.0), (5.0, 1.68), lw=1.9, ms=14)
    ax.text(5.0, 1.38,
            r"$\mathbb{E}_{\Gamma\otimes\Delta\otimes P_{\mathbf{U}}}"
            r"[\,ci\,]\ =\ $ PCI score for $X_k$",
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color=INK, zorder=7)

    # --- OBCB inset (concrete grounding) ---
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.25, 0.05), 9.5, 0.92, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0, edgecolor="#d7dde1", facecolor="#fbfcfc", zorder=2))
    ax.text(0.45, 0.51, "OBCB\ninstance", ha="left", va="center", fontsize=7.6,
            fontweight="bold", color=TEAL, zorder=6, linespacing=0.95)
    ax.text(2.05, 0.51,
            r"$X_k=$gender,   $\mathbf{T}=\{$check-failed$\}$ held at factual,"
            "\n"
            r"$\mathbf{c}^\star=$female (restore),   $\mathbf{c}'=$male (alternative)",
            ha="left", va="center", fontsize=7.6, color=CAP_COL, zorder=6,
            linespacing=1.15)


# ============================================================
<<<<<<< HEAD
# Panel (b): outcome plane + ci field
# ============================================================
def panel_plane(ax):
    ystar = 0.5  # factual outcome, mid-range for a generic continuous Y
    # grid
    nx = np.linspace(0, 1, 240)            # necessity strength |Y^n - y*|
    ys = np.linspace(0, 1, 240)            # sufficiency-world outcome Y^s
    NX, YS = np.meshgrid(nx, ys)
    # impact kernel EXACTLY as defined (Absolute Difference score, sec3):
    #   ci = |y^n - y*| - |y^s - y*|   (necessity gain minus sufficiency loss).
    # signed; the L1 ridge along Y^s=y* is genuine and coincides with the
    # sufficiency-factual reference line drawn below.
    CI = NX - np.abs(YS - ystar)

    cmap = LinearSegmentedColormap.from_list(
        "ci", ["#caa46a", "#ecdcc0", "#ffffff", "#bfe3e6", "#1F9AA6"])
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=1.0)
    im = ax.imshow(CI, origin="lower", extent=[0, 1, 0, 1], aspect="auto",
                   cmap=cmap, norm=norm, zorder=1, alpha=0.95)
    cs = ax.contour(NX, YS, CI, levels=[-0.25, 0.0, 0.25, 0.5, 0.75],
                    colors="#6b7780", linewidths=0.6, alpha=0.5, zorder=2)
    ax.clabel(cs, fmt="%.2f", fontsize=6.0, inline=True)

    # slim colourbar shows the actual ci scale (negative = sufficiency dominates)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(r"$ci=|y^n{-}y^\star|-|y^s{-}y^\star|$", fontsize=7.4, color=INK)
    cb.ax.tick_params(labelsize=6.5, colors=CAP_COL)
    cb.outline.set_edgecolor("#cad2d8")

    # factual reference lines: sufficiency factual (Y^s=y*, horizontal) and
    # necessity factual (Y^n=y*, i.e. zero necessity strength, vertical)
    ax.axhline(ystar, color=OUT_EDGE, lw=1.6, zorder=4)
    ax.axvline(0.0, color=OUT_EDGE, lw=1.6, zorder=4)
    ax.text(0.018, 0.985, r"$Y^n{=}y^\star$", rotation=90, ha="left", va="top",
            fontsize=6.6, color=OUT_EDGE, zorder=6)
    ax.text(0.988, ystar - 0.03, r"$Y^s{=}y^\star$", ha="right", va="top",
            fontsize=6.6, color=OUT_EDGE, zorder=6)
    ax.plot([0.0], [ystar], marker="*", ms=16, color=OUT_EDGE,
            markeredgecolor="white", markeredgewidth=1.0, zorder=7, clip_on=False)
    ax.annotate(r"factual $y^\star$", xy=(0.0, ystar), xytext=(0.14, ystar - 0.22),
                fontsize=7.6, color=OUT_EDGE, fontweight="bold", ha="left",
                va="center", zorder=8,
=======
# Shared draws feeding panel (b), seeded for a reproducible cloud.
# ============================================================
def sample_draws(seed=7, n=2500):
    rng = np.random.default_rng(seed)
    # X = Y^n - y*: necessity deviation, usually strongly positive (removing
    # the candidate cause usually flips the outcome the same way) but with
    # enough spread that noise occasionally pushes it negative.
    X = rng.normal(0.35, 0.16, size=n)
    # Y = Y^s - y*: sufficiency deviation, centred at 0 (restoring the cause
    # usually reproduces the factual outcome) with symmetric noise, so both
    # signs are common -- this is what actually populates all four quadrants.
    Y = rng.normal(0.0, 0.13, size=n)
    CI = np.abs(X) - np.abs(Y)
    return X, Y, CI


def pick_quadrant_examples(X, Y, target=(0.32, 0.14)):
    """One real draw per quadrant, each close to the same |x|,|y| target --
    a genuine near-coincidence in the sampled data, not four hand-placed
    synthetic points."""
    x0, y0 = target
    picks = {}
    for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
        mask = (np.sign(X) == sx) & (np.sign(Y) == sy)
        if not mask.any():
            continue
        idx = np.flatnonzero(mask)
        d2 = (X[idx] - sx * x0) ** 2 + (Y[idx] - sy * y0) ** 2
        picks[(sx, sy)] = idx[d2.argmin()]
    return picks


# ============================================================
# Panel (b): signed quadrant plane -- same score, four stories
# ============================================================
def panel_quadrants(ax, X, Y, CI):
    d = 0.75  # axis half-range; wide enough to hold the sampled cloud
    gx = np.linspace(-d, d, 260)
    gy = np.linspace(-d, d, 260)
    GX, GY = np.meshgrid(gx, gy)
    # impact kernel EXACTLY as defined (Absolute Difference score, sec3):
    #   ci = |Y^n - y*| - |Y^s - y*|   (necessity gain minus sufficiency loss).
    # Depends only on the two absolute deviations, so it is invariant under an
    # independent sign flip of either axis -- the field is genuinely 4-fold
    # symmetric, which is the point of drawing it unfolded like this.
    FIELD = np.abs(GX) - np.abs(GY)

    cmap = LinearSegmentedColormap.from_list(
        "ci", ["#caa46a", "#ecdcc0", "#ffffff", "#bfe3e6", "#1F9AA6"])
    norm = TwoSlopeNorm(vmin=-d, vcenter=0.0, vmax=d)
    im = ax.imshow(FIELD, origin="lower", extent=[-d, d, -d, d], aspect="auto",
                   cmap=cmap, norm=norm, zorder=1, alpha=0.9)
    cs = ax.contour(GX, GY, FIELD, levels=[-0.4, -0.2, 0.0, 0.2, 0.4],
                    colors="#6b7780", linewidths=0.6, alpha=0.5, zorder=2)
    ax.clabel(cs, fmt="%.1f", fontsize=6.0, inline=True)

    cb = ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(r"$ci=|Y^n{-}y^\star|-|Y^s{-}y^\star|$", fontsize=7.4, color=INK)
    cb.ax.tick_params(labelsize=6.5, colors=CAP_COL)
    cb.outline.set_edgecolor("#cad2d8")

    # quadrant dividers: both worlds exactly at their factual value
    ax.axhline(0.0, color=OUT_EDGE, lw=1.4, zorder=4)
    ax.axvline(0.0, color=OUT_EDGE, lw=1.4, zorder=4)
    ax.plot([0.0], [0.0], marker="*", ms=15, color=OUT_EDGE,
            markeredgecolor="white", markeredgewidth=1.0, zorder=7, clip_on=False)
    ax.annotate(r"factual $y^\star$ (both worlds)", xy=(0.0, 0.0),
                xytext=(0.05, 0.22), fontsize=6.8, color=OUT_EDGE,
                fontweight="bold", ha="left", va="center", zorder=8,
>>>>>>> main
                arrowprops=dict(arrowstyle="-|>", color=OUT_EDGE, lw=1.0),
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.85))

<<<<<<< HEAD
    # "ideal cause" corner annotation -> far-right end of the ridge (max ci);
    # curved so the arrow arcs over the draw cloud rather than through it
    ax.annotate("ideal cause\n(changes when removed,\nsustains when restored)",
                xy=(0.985, ystar + 0.01), xytext=(0.40, 0.74),
                fontsize=7.4, color=INK, ha="center", va="center", zorder=7,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1,
                                connectionstyle="arc3,rad=-0.32", shrinkB=2),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                          alpha=0.8))

    # sampled fork draws -> a cloud (nudged above y* so the mean is off the line)
    pts_x = np.array([0.82, 0.74, 0.9, 0.68, 0.86, 0.78, 0.6, 0.71, 0.88, 0.8])
    pts_y = ystar + 0.10 + np.array([0.02, -0.04, 0.05, -0.01, -0.03, 0.04,
                                     -0.06, 0.03, -0.02, 0.01])
    ax.scatter(pts_x, pts_y, s=22, facecolor="#1d2730", edgecolor="white",
               linewidths=0.6, zorder=8)
    mx, my = pts_x.mean(), pts_y.mean()
    ax.scatter([mx], [my], marker="^", s=140, facecolor=GOLD,
               edgecolor="white", linewidths=1.2, zorder=9)
    ax.annotate(r"$\mathbb{E}[ci]$", xy=(mx, my), xytext=(mx, my - 0.15),
                fontsize=8.4, fontweight="bold", color=GOLD, ha="center",
                va="center", zorder=9,
                arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.1, shrinkB=10),
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.85))

    # manual legend in the empty lower-left (low-ci) quadrant
    handles = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor=OUT_EDGE,
               markeredgecolor="white", markersize=13,
               label=r"factual outcome $y^\star$"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=INK,
               markeredgecolor="white", markersize=6, label="fork draws"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=GOLD,
               markeredgecolor="white", markersize=9,
               label=r"mean $=\mathbb{E}[ci]=$ score"),
    ]
    leg = ax.legend(handles=handles, loc="lower right", fontsize=7.0,
                    framealpha=0.93, borderpad=0.6, labelspacing=0.55,
                    handlelength=1.7, handletextpad=0.6)
    leg.get_frame().set_edgecolor("#cad2d8")
    leg.set_zorder(10)

    ax.set_xlim(-0.04, 1.0); ax.set_ylim(0, 1)
    ax.set_xlabel(r"necessity strength  $|Y^n - y^\star|$", fontsize=9.0, color=INK)
    ax.set_ylabel(r"sufficiency outcome  $Y^s$", fontsize=9.0, color=INK)
    ax.tick_params(labelsize=7.5, colors=CAP_COL)
    for s in ax.spines.values():
        s.set_color("#cad2d8")
    ax.set_title(r"(b)  Impact kernel: Absolute Difference score (one choice of $ci$)",
                 loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=8)


def emit(outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.4),
                                   gridspec_kw=dict(width_ratios=[1.18, 1.0],
                                                    wspace=0.16))
    panel_fork(ax1)
    panel_plane(ax2)
=======
    # the full sampled cloud, faint
    ax.scatter(X, Y, s=5, facecolor=INK, edgecolor="none", alpha=0.16,
               zorder=5)

    # one real draw per quadrant, matched on |x|,|y| -- an actual
    # near-coincidence in the data, not a synthetic symmetric example
    picks = pick_quadrant_examples(X, Y)
    if picks:
        idx = list(picks.values())
        ax.scatter(X[idx], Y[idx], s=70, facecolor=GOLD, edgecolor="white",
                   linewidths=1.3, zorder=9)
        cis = CI[idx]
        ax.text(-d * 0.95, d * 0.90,
                r"$ci\in[$" f"{cis.min():.2f}, {cis.max():.2f}"
                r"$]$ despite four different directions",
                fontsize=6.6, color=INK, ha="left", va="top", zorder=9,
                linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=GOLD,
                          lw=1.1, alpha=0.95))

    legend_elements = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=INK,
               markeredgecolor="none", alpha=0.6, markersize=5,
               label="all sampled draws"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GOLD,
               markeredgecolor="white", markeredgewidth=1.0, markersize=8,
               label="one draw per quadrant, magnitude-matched"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=6.5,
              frameon=True, framealpha=0.9, edgecolor="#cad2d8",
              handletextpad=0.5, borderpad=0.5)

    ax.set_xlim(-d, d); ax.set_ylim(-d, d)
    ax.set_xlabel(r"necessity deviation  $Y^n - y^\star$", fontsize=9.0, color=INK)
    ax.set_ylabel(r"sufficiency deviation  $Y^s - y^\star$", fontsize=9.0, color=INK)
    ax.tick_params(labelsize=7.5, colors=CAP_COL)
    for s in ax.spines.values():
        s.set_color("#cad2d8")
    ax.set_title(r"(b)  The impact kernel on the signed deviation plane",
                 loc="left", fontsize=9.8, fontweight="bold", color=INK, pad=8)


def emit(outdir):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.5, 5.6),
        gridspec_kw=dict(width_ratios=[1.25, 1.0], wspace=0.30))
    panel_fork(ax1)
    X, Y, CI = sample_draws()
    panel_quadrants(ax2, X, Y, CI)
>>>>>>> main
    fig.tight_layout()
    # Vector PDF for the paper (text/lines/patches stay vector; only the ci
    # heatmap is raster, embedded at high dpi); PNG is a quick preview.
    fig.savefig(os.path.join(outdir, "pci_mechanism.pdf"), dpi=600,
                bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(outdir, "pci_mechanism.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote pci_mechanism.{pdf,png} to", outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    emit(args.outdir)
