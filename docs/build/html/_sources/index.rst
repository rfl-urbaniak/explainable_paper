PCI Paper — Computational Notebooks
=====================================

Companion notebooks for the paper *Probabilistic Causal Impact (PCI)*.
Each notebook reproduces a specific chunk of the paper's quantitative content
— verifying the numbers, generating the figures, and stress-testing PCI
against alternatives (PN/PS/PNS, marginal SHAP, causal SHAP, gradient-based
attribution, classical actual-cause probabilities).

Setup
-----

Requires `uv <https://docs.astral.sh/uv/>`_.

.. code-block:: bash

   uv sync --group docs
   make -C docs html       # build this HTML site
   make -C docs serve      # browse at http://localhost:8000

The notebooks are intended to be **read here**, on the rendered HTML site —
each one ships with its full publication-quality outputs (figures, tables,
print statements). Raw ``.ipynb`` files live under ``docs/source/`` if you
want to inspect or modify the source.

The notebooks
-------------

The set divides cleanly into two halves: closed-form verification of the
worked examples (Sections 2 and 4 of the paper) and empirical benchmarks
that compare PCI to existing attribution and actual-cause machinery.

**Closed-form verification.** Small, hand-checkable models where every number
quoted in the paper can be recomputed analytically.

:doc:`obcb_computations` — *Old Boys' Club Bank: PN, PS, PNS, PCI, SHAP.*
    Recomputes every number in Section 2 of the paper for the stochastic OBCB
    example. Population-level and individual (Alice, Bob) versions of Pearl's
    PN, PS, PNS; PCI with and without witnesses; plain SHAP and causal SHAP
    on both the 2-feature and 3-feature games. Closes with a four-method
    comparison table that drives the paper's "what each method picks up"
    discussion. Closed-form throughout — no Monte Carlo, no fitting.

:doc:`signal_mediation_computations` — *Signal with mediation: chain
X → M → Y.*
    Verifies the numerical content of Sections 4.4–4.6 on a linear Gaussian
    chain with additive noise. Computes plain SHAP, causal SHAP, and PCI
    (without and with the third variable as witness) on three factual
    instances, then assembles the full desiderata table. A Monte Carlo
    cross-check sits alongside each closed-form value.

:doc:`responsibility_archetypes` — *Three causal archetypes plus an
irrelevance control.*
    Synthetic SCM with linear-necessary-and-sufficient inputs, an
    overdetermined branch, a preempted branch, and a disconnected control.
    Runs PCI's necessity / sufficiency decomposition on two contrasting
    factual cases (preempted vs. unpreempted regime) and compares against
    marginal SHAP and causal SHAP on the same model. Demonstrates the
    methodological gap that motivates the N / S split: SHAP returns one
    number per feature; PCI returns two, and the two numbers carry distinct
    structural information.

:doc:`desert_traveler` — *Pearl's desert traveler: PCI vs Definition 10.3.5.*
    Reproduces analytically the PCI necessity expectations on Pearl's original
    desert-traveler example (under HP-style mediator witnesses), then
    introduces a weak-poison variant in which Pearl's ``PS`` reading separates
    the two enemies' degrees of responsibility while his
    probability-of-actual-causation posterior cannot. Includes a cross-check
    against the framework's ``ThinSearchSampler`` and a flag on the
    witness-mechanism gap between the paper's specification and the current
    implementation.

**Empirical benchmarks.** Larger experiments that compare PCI to existing
machinery on synthetic SCMs and a dynamical SIR model.

:doc:`actual_causality_benchmark` — *PCI vs. classical actual causality.*
    Compares PCI against Halpern–Pearl-style actual-cause probabilities on
    the generalised throwing problem. Exact actual-cause computations (one
    pair, two pairs, scaling sweep) followed by approximate estimation via
    ``SearchForExplanation``. The runtime / search-space / accuracy plots in
    the paper come from this notebook.

:doc:`sir_benchmark` — *PCI on a dynamical SIR model with policies.*
    Applies PCI to a Bayesian SIR epidemiological model with two interacting
    non-pharmaceutical policies (lockdown, mask-wearing). Mirrors the chirho
    tutorial on explainable reasoning in dynamical systems but swaps the
    explanatory machinery for the PCI thin-search sampler. Recovers chirho's
    qualitative finding — lockdown is the dominant cause of excessive
    overshoot — while exposing the necessity / sufficiency decomposition
    that classical actual causality collapses.

:doc:`gradient_based_attribution` — *PCI vs. gradient-based attribution.*
    Compares PCI's responsibility scores against gradient / sensitivity-based
    attribution methods on a controlled synthetic model. Probes invariance
    to feature scale, sensitivity to priors, and the differential causal
    effect. Targets the paper's discussion of why gradient methods are not
    causally faithful even when their numbers look plausible.

.. toctree::
   :maxdepth: 2
   :caption: Notebooks
   :hidden:

   obcb_computations
   signal_mediation_computations
   responsibility_archetypes
   desert_traveler
   actual_causality_benchmark
   sir_benchmark
   gradient_based_attribution
