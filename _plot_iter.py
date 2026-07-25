"""Quick plot iteration script: loads /tmp/archetypes_samples_20k.pkl, computes
desiderata, renders the forest plot to figures/archetypes_threshold_forest.png.
Bypasses the 5-min sampling step in the notebook."""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

CACHE = Path("/tmp/archetypes_samples_20k.pkl")
FIG_DIR = Path(__file__).resolve().parent / "figures"
N_BOOT = 4000
SEED_BOOT = 0

cache = pickle.loads(CACHE.read_bytes())
samples_c1 = cache["samples_c1"]
samples_c2 = cache["samples_c2"]
case1_index = cache["case1_index"]
case2_index = cache["case2_index"]
suspect_names = cache["suspect_names"]
num_search_samples = cache["num_search_samples"]
factual_c1 = cache["factual_c1"]
factual_c2 = cache["factual_c2"]


def bootstrap_diff(arr_v, arr_d, n_boot=N_BOOT, seed=SEED_BOOT):
    rng = np.random.default_rng(seed)
    a = np.asarray(arr_v); d = np.asarray(arr_d)
    n = len(a)
    out = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        out[k] = np.nanmean(a[idx]) - np.nanmean(d[idx])
    point = np.nanmean(a) - np.nanmean(d)
    return point, out.std(ddof=1), np.quantile(out, [0.025, 0.975])


def delta(samples, v, kind):
    return bootstrap_diff(samples[v][kind], samples["D"][kind])


def diff_of_deltas(samples, v1, k1, v2, k2):
    rng = np.random.default_rng(SEED_BOOT)
    a = np.asarray(samples[v1][k1]); b = np.asarray(samples[v2][k2])
    d1 = np.asarray(samples["D"][k1]); d2 = np.asarray(samples["D"][k2])
    n = len(a)
    out = np.empty(N_BOOT)
    for k in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        out[k] = (np.nanmean(a[idx]) - np.nanmean(d1[idx])) - (
            np.nanmean(b[idx]) - np.nanmean(d2[idx])
        )
    point = (np.nanmean(a) - np.nanmean(d1)) - (np.nanmean(b) - np.nanmean(d2))
    return point, out.std(ddof=1), np.quantile(out, [0.025, 0.975])


def cross_flip(samples1, samples2, v, kind):
    rng = np.random.default_rng(SEED_BOOT)
    a1 = np.asarray(samples1[v][kind]); d1 = np.asarray(samples1["D"][kind])
    a2 = np.asarray(samples2[v][kind]); d2 = np.asarray(samples2["D"][kind])
    n = min(len(a1), len(a2))
    out = np.empty(N_BOOT)
    for k in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        out[k] = (np.nanmean(a2[idx]) - np.nanmean(d2[idx])) - (
            np.nanmean(a1[idx]) - np.nanmean(d1[idx])
        )
    point = (np.nanmean(a2) - np.nanmean(d2)) - (np.nanmean(a1) - np.nanmean(d1))
    return point, out.std(ddof=1), np.quantile(out, [0.025, 0.975])


def winner_loser(factual):
    if 5 * factual["O1"] >= 5 * factual["O2"]:
        return "O1", "O2"
    return "O2", "O1"


O_w_c1, O_l_c1 = winner_loser(factual_c1)
O_w_c2, O_l_c2 = winner_loser(factual_c2)

ses_all = []
for samples in (samples_c1, samples_c2):
    for v in suspect_names:
        if v == "D":
            continue
        for kind in ("nec", "suff"):
            _, se, _ = delta(samples, v, kind)
            ses_all.append(se)
sigma_max = max(ses_all)
EPS = round(1.645 * sigma_max, 3)
print(f"sigma_max = {sigma_max:.4f}  EPS = {EPS:.3f}")


def claim(num, case, label, claim_type, point, se, ci_lo, ci_hi, eps, formula):
    if claim_type == "above":
        passes = bool(point > eps)
    elif claim_type == "below_abs":
        passes = bool(abs(point) < eps)
    elif claim_type == "above_zero":
        passes = bool(point > 0)
    elif claim_type == "diagnostic":
        passes = None
    return {
        "num": num, "case": case, "claim": label, "claim_type": claim_type,
        "measured": point, "se": se, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "eps": eps, "passes": passes, "formula": formula,
    }


records = []

# Case 1: preempted regime, contestable O pair
top_other_c1 = max(
    (v for v in suspect_names if v not in ("L2", "D")),
    key=lambda v: delta(samples_c1, v, "nec")[0],
)
p, se, ci = diff_of_deltas(samples_c1, "L2", "nec", top_other_c1, "nec")
records.append(claim(1, "C1",
    "[C1] Heaviest linear contributor L₂ has the largest necessity (gate-off regime)",
    "above_zero", p, se, ci[0], ci[1], 0.0,
    "ΔN(L₂) − max ΔN(other)"))
p, se, ci = delta(samples_c1, "P", "nec")
records.append(claim(8, "C1",
    "[C1] Preempted variable P sits at the irrelevance noise floor (gate off)",
    "below_abs", p, se, ci[0], ci[1], EPS,
    "ΔN(P)"))
p, se, ci = diff_of_deltas(samples_c1, "L1", "nec", "P", "nec")
records.append(claim(9, "C1",
    "[C1] Active linear contributor L₁ ranks above preempted P",
    "above", p, se, ci[0], ci[1], EPS,
    "ΔN(L₁) − ΔN(P)"))
p, se, ci = diff_of_deltas(samples_c1, O_w_c1, "suff", O_l_c1, "suff")
records.append(claim(3, "C1",
    "[C1] Overdetermined winner is more sufficient than loser (contestable pair)",
    "above", p, se, ci[0], ci[1], EPS,
    f"ΔS({O_w_c1}) − ΔS({O_l_c1})"))
p, se, ci = diff_of_deltas(samples_c1, O_w_c1, "nec", O_l_c1, "nec")
records.append(claim(5, "C1",
    "[C1] Overdetermined winner & loser are equally necessary (contestable pair)",
    "below_abs", p, se, ci[0], ci[1], EPS,
    f"ΔN({O_w_c1}) − ΔN({O_l_c1})"))

# Case 2: unpreempted regime, dominant O winner
top_other_c2 = max(
    (v for v in suspect_names if v not in ("L2", "D")),
    key=lambda v: delta(samples_c2, v, "nec")[0],
)
p, se, ci = diff_of_deltas(samples_c2, "L2", "nec", top_other_c2, "nec")
records.append(claim(2, "C2",
    "[C2] Linear contributor L₂ (weight 10) gets the largest necessity score",
    "above_zero", p, se, ci[0], ci[1], 0.0,
    "ΔN(L₂) − max ΔN(other)"))
p, se, ci = diff_of_deltas(samples_c2, O_w_c2, "suff", O_l_c2, "suff")
records.append(claim(4, "C2",
    "[C2] Overdetermined winner is more sufficient than loser (dominant winner)",
    "above", p, se, ci[0], ci[1], EPS,
    f"ΔS({O_w_c2}) − ΔS({O_l_c2})"))
p, se, ci = diff_of_deltas(samples_c2, O_w_c2, "suff", O_w_c2, "nec")
records.append(claim(6, "C2",
    "[C2] Overdetermined winner is more sufficient than necessary (S-not-N signature)",
    "above", p, se, ci[0], ci[1], EPS,
    f"ΔS({O_w_c2}) − ΔN({O_w_c2})"))
p, se, ci = diff_of_deltas(samples_c2, "L2", "nec", O_w_c2, "nec")
records.append(claim(7, "C2",
    "[C2] Linear contributor L₂ is more necessary than the overdetermined winner "
    f"({O_w_c2}) — linear has no backup path",
    "above", p, se, ci[0], ci[1], EPS,
    f"ΔN(L₂) − ΔN({O_w_c2})"))
p, se, ci = delta(samples_c2, "P", "nec")
records.append(claim(10, "C2",
    "[C2] Active P (gate on) clears the irrelevance noise floor",
    "above", p, se, ci[0], ci[1], EPS,
    "ΔN(P)"))

# (Diagnostics D1 and D2 dropped from the plot per user request — they remain in
# the notebook desiderata table for completeness but are not visualised.)


# ---- forest plot ----
def pass_region(claim_type, eps):
    if claim_type == "above":
        return (eps, 100.0)
    if claim_type == "above_zero":
        return (0.0, 100.0)
    if claim_type == "below_abs":
        return (-eps, eps)
    return None


# Sort: numbered desiderata in claim-id order (#1..#10), then diagnostics.
def _key(row):
    n = row["num"]
    if isinstance(n, str) and n.startswith("D"):
        return (1, int(n[1:]))
    return (0, int(n))


ordered = sorted(records, key=_key)

group_titles = {
    "numbered": "Numbered desiderata (claim order)",
    "diag":     "Diagnostics (not numbered desiderata)",
}

display_rows = [{"kind": "data", "row": r} for r in ordered]


def color_for(passes):
    if passes is None:
        return "#888888"
    return "#2ca02c" if passes else "#d62728"


all_lo = min(r["ci_lo"] for r in records)
all_hi = max(r["ci_hi"] for r in records)
xpad = 0.20
xmin = all_lo - xpad
# All bars must fit in [xmin, data_right]; the right column lives in
# [annotation_left, xmax] and is fully separated from the data area.
data_right = all_hi + 0.10
annotation_left = data_right + 0.30          # gap between bars and text
formula_max_chars = max(len(r["formula"]) for r in records) + len(" = +0.000")
xmax = annotation_left + 0.085 * formula_max_chars

n_rows = len(display_rows)
fig, ax = plt.subplots(figsize=(11.0, 0.55 * n_rows + 1.5))

PASS_FILL = "#d4edda"
PASS_EDGE = "#2ca02c"

for y, entry in enumerate(display_rows):
    if entry["kind"] == "header":
        continue
    r = entry["row"]
    pr = pass_region(r["claim_type"], r["eps"])
    if pr is not None:
        # Cap the pass region at data_right so it never crosses into the annotation column.
        lo, hi = max(pr[0], xmin), min(pr[1], data_right)
        rect = Rectangle((lo, y - 0.42), hi - lo, 0.84,
                         facecolor=PASS_FILL, edgecolor=PASS_EDGE,
                         linewidth=0.7, alpha=0.85, zorder=0)
        ax.add_patch(rect)
    color = color_for(r["passes"])
    ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=color, linewidth=2.4, alpha=0.92, zorder=2)
    ax.plot([r["measured"]], [y], "o", color=color, markersize=8.5,
            markeredgecolor="white", markeredgewidth=1.6, zorder=3)
    note = f"{r['formula']} = {r['measured']:+.3f}"
    ax.text(annotation_left, y, note,
            fontsize=9, color="#222", ha="left", va="center")

ax.axvline(0, color="#444", linewidth=0.6, alpha=0.5, zorder=1)

ylabels = []
for entry in display_rows:
    if entry["kind"] == "header":
        ylabels.append(group_titles[entry["case"]])
    else:
        r = entry["row"]
        ylabels.append(f"   #{r['num']}  {r['claim']}")
ax.set_yticks(range(n_rows))
ax.set_yticklabels(ylabels, fontsize=9)
for tick, entry in zip(ax.get_yticklabels(), display_rows):
    if entry["kind"] == "header":
        tick.set_fontweight("bold")
        tick.set_color("#111")
ax.invert_yaxis()

first_header = True
for y, entry in enumerate(display_rows):
    if entry["kind"] == "header":
        if first_header:
            first_header = False
        else:
            ax.axhline(y - 0.5, color="#aaa", linewidth=1.0)

legend_handles = [
    Patch(facecolor=PASS_FILL, edgecolor=PASS_EDGE,
          label="region of values that confirm the claim"),
    plt.Line2D([0], [0], marker="o", color="#2ca02c", linewidth=2.2, markersize=8,
               markeredgecolor="white", label="claim confirmed"),
    plt.Line2D([0], [0], marker="o", color="#d62728", linewidth=2.2, markersize=8,
               markeredgecolor="white", label="claim refuted"),
]
ax.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=3,
    fontsize=9,
    frameon=False,
)

ax.set_xlim(xmin, xmax)
ax.set_xlabel(
    "PCI score gap (necessity or sufficiency, excess over the irrelevance-control "
    "noise floor)\n"
    "horizontal bar = bootstrap 95% CI; right column = formula and point estimate",
    fontsize=9.5,
)
# Despine: hide top/right; soften left/bottom
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.spines["left"].set_color("#888")
ax.spines["bottom"].set_color("#888")
ax.tick_params(colors="#444")
ax.set_title(
    "PCI archetype desiderata — does the measurement confirm the structural claim?",
    fontsize=11,
)
plt.tight_layout()
out_path = FIG_DIR / "archetypes_threshold_forest.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Wrote {out_path}")
