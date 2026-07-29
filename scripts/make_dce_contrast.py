#!/usr/bin/env python3
"""PCI vs. DCE contrast on the credit-limit grid (Example~\\ref{ex:age} in
sections/sec6_dce.tex), used as Figure~\\ref{fig:eval_dce} in the evaluation
overview and referenced from the DCE appendix.

Panel (a) -- PCI: the difference between the \\ourapproach scores for age and
for time of application, on the (age, hour) grid. Positive everywhere: PCI
ranks age above time of application throughout.

Panels (b), (c) -- DCE: the corresponding difference in DCE magnitudes, on
the hours and minutes time scales respectively. DCE favours time of
application over most of the grid (the opposite of PCI's verdict), except in
a narrow band where the sigmoid's inflection point (age near 30) produces a
sharp gradient spike; that spike would saturate a shared colour scale, so the
scale is set from the region away from it and the spike itself is shown
clipped and annotated. Comparing (b) and (c) makes DCE's unit-sensitivity
explicit: switching from hours to minutes collapses most of the
time-favouring region toward zero.

Uses the cached PCI search results (docs/source/search_results.pkl) and the
closed-form internal credit-limit model from
docs/source/gradient_based_attribution.ipynb -- no expensive re-sampling.

Run: python3 scripts/make_dce_contrast.py
Output: figures/dce_contrast_overview.{pdf,png}
"""
import os
import pickle
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import transforms

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from pci.explanation.regime import condition_on_interventional_regime
from pci.explanation.scores import abs_diff_score

warnings.filterwarnings("ignore")

INK = "#1d2730"
CAP = "#6b7780"
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})


def internal_model_hours(age_norm, hour_norm):
    """Closed-form credit-limit model: sigmoidal rise in age, small sinusoidal
    modulation in time of day. Matches internal_model_hours in
    docs/source/gradient_based_attribution.ipynb exactly."""
    steepness = 2
    amplitude = 0.03
    peak_hour = 0
    age_scaled = (age_norm - 30) * steepness
    transform = transforms.ComposeTransform(
        [transforms.SigmoidTransform(), transforms.AffineTransform(300, 200)]
    )
    limit_base = transform(age_scaled)
    multiplier = 1.0 + amplitude * torch.sin(
        2 * torch.pi * (hour_norm - peak_hour) / 6.5 + torch.pi / 2
    )
    return multiplier * limit_base


def pci_scores(search_results_path):
    ages = torch.linspace(20.0, 70.0, steps=51)
    hours = torch.linspace(0.0, 8.0, steps=33)
    A, H = torch.meshgrid(ages, hours, indexing="ij")
    A_flat, H_flat = A.reshape(-1, 1, 1), H.reshape(-1, 1, 1)
    factual_limits = internal_model_hours(A_flat, H_flat)

    with open(search_results_path, "rb") as f:
        search_results = pickle.load(f)
    results = search_results["hours_age1"]

    scores = {}
    for suspect in ("age_norm", "hour_norm"):
        cond = condition_on_interventional_regime(
            results_dictionary=results,
            reference_variable_names=[suspect],
            antecedent_regimes={suspect: True},
            witness_regimes={suspect: False},
        )
        sc = abs_diff_score(
            suff_outcomes=cond["regime_sufficiency"]["limit"].detach(),
            nec_outcomes=cond["regime_necessity"]["limit"],
            factual_outcomes=factual_limits,
        )
        scores[suspect] = sc["total"].nanmean(dim=0).flatten().numpy().reshape(51, 33)
    return ages.numpy(), hours.numpy(), scores["age_norm"] - scores["hour_norm"]


def dce_diffs():
    ages = torch.linspace(20.0, 70.0, steps=51, requires_grad=True)
    hours = torch.linspace(0.0, 8.0, steps=33, requires_grad=True)
    A, H = torch.meshgrid(ages, hours, indexing="ij")
    A, H = A.reshape(-1), H.reshape(-1)
    A.requires_grad_(True)
    H.requires_grad_(True)
    out = internal_model_hours(A, H)
    grad_age, grad_hour = torch.autograd.grad(outputs=out.sum(), inputs=(A, H))
    dce_age = grad_age.detach().numpy().reshape(51, 33)
    dce_hour_hours = grad_hour.detach().numpy().reshape(51, 33)
    dce_hour_minutes = dce_hour_hours / 60.0
    return (np.abs(dce_age) - np.abs(dce_hour_hours),
            np.abs(dce_age) - np.abs(dce_hour_minutes))


def panel(ax, age_edges, hour_edges, data, title, vmin, vmax, cbar_label, ylabel=None, spike_note=False):
    im = ax.pcolormesh(age_edges, hour_edges, data.T, cmap="PuOr", vmin=vmin, vmax=vmax, shading="flat")
    ax.set_xlabel("Age of applicant", fontsize=9.5, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, color=INK)
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK, loc="left")
    ax.tick_params(labelsize=8, colors=CAP)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7.5, colors=CAP)
    cb.set_label(cbar_label, fontsize=8, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    if spike_note:
        ax.annotate(
            "clipped: gradient\nspikes to $\\pm$103 here\n(sigmoid inflection)",
            xy=(30, 7.6), xytext=(38, 6.6), fontsize=7.2, color=INK, ha="left", va="top",
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cad2d8", alpha=0.9),
        )


def emit(outdir, search_results_path):
    ages, hours, pci_diff = pci_scores(search_results_path)
    dce_diff_hours, dce_diff_minutes = dce_diffs()

    age_edges = np.linspace(20, 70, 52)
    hour_edges = np.linspace(0, 8, 34)

    # Robust diverging scale for the DCE panels: exclude the narrow sigmoid-inflection
    # spike (age within 3 years of 30) so it doesn't saturate the whole colormap and
    # hide the pattern across the rest of the grid; the spike still shows up clipped.
    away = np.abs(ages - 30) > 3
    dce_vmax = max(np.abs(dce_diff_hours[away, :]).max(), np.abs(dce_diff_minutes[away, :]).max())
    pci_vmax = np.abs(pci_diff).max()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    panel(axes[0], age_edges, hour_edges, pci_diff, "(a) PCI: age minus time score",
          -pci_vmax, pci_vmax, "score difference", ylabel="Time of application (hours)")
    panel(axes[1], age_edges, hour_edges, dce_diff_hours, "(b) DCE: age minus time gradient\n(time in hours)",
          -dce_vmax, dce_vmax, "gradient difference", spike_note=True)
    panel(axes[2], age_edges, hour_edges, dce_diff_minutes, "(c) DCE: age minus time gradient\n(time in minutes)",
          -dce_vmax, dce_vmax, "gradient difference", spike_note=True)

    fig.suptitle("PCI ranks age above time everywhere; DCE's ranking flips and depends on units",
                 fontsize=12, fontweight="bold", color=INK, y=1.04)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "dce_contrast_overview.pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(outdir, "dce_contrast_overview.png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote dce_contrast_overview.{pdf,png} to", outdir)


if __name__ == "__main__":
    outdir = os.path.join(HERE, "figures")
    os.makedirs(outdir, exist_ok=True)
    emit(outdir, os.path.join(HERE, "docs", "source", "search_results.pkl"))
