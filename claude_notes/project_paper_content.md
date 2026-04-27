---
name: PCI paper content overview
description: What the paper currently contains, section by section, and what is missing or incomplete
type: project
originSessionId: ef5c8a65-3d0b-4a46-930f-d4a4a6819e3d
---
Paper: "A Computationally Feasible Framework for Causal Probabilistic Explanation"
Main file: /home/rafal/s76projects/explainable_paper/main.tex
Class: clear2026.cls (CLeaR 2026 conference format)
Method name: Probabilistic Causal Impact (PCI), macros \ourapproach / \Ourapproach

**Why:** Understanding the current paper structure is essential for planning revisions based on reviewer feedback.

**How to apply:** Use this to quickly locate sections needing work without rereading the whole file.

## Current structure

- **Abstract** (l.188): Covers the two-fold problem (AC not scalable, SHAP not causal), introduces PCI, summarizes benchmarks and real-world AVM result vs SHAP.

- **Section 1: Introduction** (l.200): Motivates causal attribution, critiques AC and SHAP/DCE, introduces PCI as generalizing Pearl's PN/PS/PNS with witness-based context. Has a \todo noting "revise this list once paper done."

- **Section 2: Connection with Actual Causality and PN/PNS** (l.238, label: app:conceptual): Currently acts as the conceptual appendix referenced from Section 1. Contains:
  - Bob and Alice at OBCB Bank (deterministic and stochastic examples)
  - Definitions: But-for (single), But-for (sets), Actual cause (with witnesses)
  - Definitions: PN, PS, PNS
  - Subsection "Conclusion" (l.537): summarizes what AC and PNS can/can't do; motivates PCI
  - Large commented-out blocks with detailed calculations

- **Section 3: Causal Impact: definitions** (l.568, label: sec:causal_impact): Formal development.
  - PSCM definition
  - Interventions, Interventional Law
  - Causal set, Variable Selection Distribution, Alternative Value Distribution
  - Joint Necessity and Sufficiency Measure (main definition)
  - Causal Impact Function and its Expectation
  - Examples: PNS for binary outcomes, Absolute Difference Impact Score

- **Bibliography** (l.844): uses references.bib

## What is absent from main.tex (referenced but missing)

- sec:th_ac — theoretical connection to AC (theorems 10 & 11, referenced in intro)
- sec:ac_benchmark — stone-throwing / overdetermination benchmark
- sec:con_benchmark — continuous benchmark
- sec:avm — real-world AVM application (Section 6)
- app:DCE — comparison to DCE (Differential Causal Effect)

These sections are listed in the intro's enumerate but the actual content does not appear in the file. They may be in separate files or not yet written.

## Key macros and commands
- \ourapproach → "PCI " (with trailing space), \Ourapproach → "PCI"
- \pn, \ps, \pns — math macros for PN, PS, PNS
- \todo{} — inline colored box (todonotes, inline mode set via \presetkeys)
- \CITE — green highlighted CITE placeholder
- \TODO — orange highlighted TODO placeholder
- \varname{} — math italic variable names

## Setup notes
- todonotes loaded with \let\todo\relax before \usepackage to avoid conflict with class definition
- \presetkeys{todonotes}{inline}{} makes \todo{} render inline by default
- Missing packages installed: texlive-science (algorithm2e), texlive-fonts-extra (bbm)
