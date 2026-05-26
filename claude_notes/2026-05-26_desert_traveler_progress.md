---
name: 2026-05-26-desert-traveler-progress
description: Mid-flight progress on desert traveller notebook + sec7b — formal-def spec landed, half-applied to notebook; resume here
type: project
---

# Desert traveller — progress saved 2026-05-26 (afternoon)

## What we decided

The desert traveller notebook + sec7b were using **inconsistent specifications**.
The fix: use the formal PCI definition (sec3:451–462) consistently. Concretely:

| Component | Choice |
|---|---|
| Suspects $\mathbf{S}$ | $\{X, P\}$ |
| Cause sets $\mathbf{C}$ | singletons |
| Witness pool $\mathbf{W}$ | **mediators** ($\{c, d\}$ for original DT; $\{c, d, v_C\}$ for weak poison) |
| Non-cause suspect | **marginalised over Bern(0.5)** — formal-def behaviour, NOT pinned. Earlier "1/8" used pinned non-cause, which is hard-coding observation and not in the spirit of sec3. |
| Noise $P_{\mathbf{U}}$ | **prior** (matches framework); conditioning $\xi$ on the factual outcome (what the notebook did before) would collapse the integral and erase sufficiency separation |
| $\Gamma$ over witness subsets | **cardinality-uniform** (matches framework's `sample_k_indices(weighting="cardinality")`) |

## Numbers under the new spec

**Original DT**, factual u=0 noise:

| cause | $Y^n$ fwk | $Y^n$ man | $Y^s$ fwk | $Y^s$ man |
|---|---|---|---|---|
| X | 0.2623 | 0.2500 | 0.1327 | 0.1250 |
| P | 0.3360 | 0.3333 | 0.0428 | 0.0417 |

**Original DT**, factual u=1:

| cause | $Y^n$ fwk | $Y^n$ man | $Y^s$ fwk | $Y^s$ man |
|---|---|---|---|---|
| X | 0.3327 | 0.3333 | 0.0412 | 0.0417 |
| P | 0.2539 | 0.2500 | 0.1225 | 0.1250 |

**Weak poison**:

| Scenario | cause | $Y^n$ fwk | $Y^n$ man | $Y^s$ fwk | $Y^s$ man |
|---|---|---|---|---|---|
| A (u=0,ξ=1) | X | 0.4775 | 0.4667 | 0.2427 | 0.2312 |
| A | P | 0.3198 | 0.3208 | **0.3889** | **0.3771** |
| B (u=1,ξ=0) | X | 0.4831 | 0.4875 | 0.1220 | 0.1187 |
| B | P | 0.2595 | 0.2500 | **0.3683** | **0.3562** |

Poison's sufficiency factor is consistently higher (less sufficient) than the
shooter's in both scenarios — matches Rafal's intuition that "sufficiency of
giving poison should be low as probability of it working is low".

## What was patched in the codebase

[pci/explanation/thin_search.py](../pci/explanation/thin_search.py):
the `witness_sites` else-branch no longer filters out `deterministic_sites`.
Deterministic sites can be witnesses (pinning them at factual via do-interventions
is well-defined); they should only be excluded from the **suspect** pool, not
the witness pool. After this patch, `witness_sites=["c", "V_C", "d"]` actually
makes the mediators eligible witnesses in the framework, and the framework's
output matches the hand-rolled enumeration under that spec.

## What's still half-applied

The notebook [docs/source/desert_traveler.ipynb](../docs/source/desert_traveler.ipynb)
currently has cells written for a SLIGHTLY EARLIER version of the spec
(forensic-conditioned noise + uniform Γ over all 8 subsets, mediator
witnesses, marginalised non-cause). It executes and prints sensible numbers
but does **not** match the framework under the final agreed spec (prior
noise, cardinality Γ).

**Next session — rewrite the notebook to the final spec:**

1. §2 (original DT): use prior noise (not forensic-conditioned), cardinality
   Γ, mediator witnesses, marginalised non-cause. Add framework cross-check
   under same spec — confirms manual = framework. Drop the "1/8 → 1/4 with
   forensic" narrative; new numbers are $Y^n$ = 1/3 (P at u=0), $1/4$ (X at
   u=0), $Y^s$ ≈ 1/24 / 1/8 depending on cell.

2. §3 (weak poison): same spec, computing both $Y^n$ AND $Y^s$. Sufficiency
   table shows P > X (poison less sufficient) in both scenarios A and B.

3. §4 (framework cross-check): merge into §2 and §3 inline — no separate
   section needed, since the framework now works on the SAME mediator witness
   pool as the manual.

Reference scripts with the correct computation:
- `/tmp/full_unified_v2.py` — original DT manual + framework match
- `/tmp/match_v2.py` — weak-poison manual + framework match

## Paper section (sec7b_paac.tex) — still needs revision

The current section text still has:
- Footnote about framework matching at 1/12 under root-only witness pool
  → no longer accurate; framework now works on mediator pool too (after the
  thin_search patch).
- Statement that "non-cause root is pinned at its factual value: it is not in
  the witness pool because the HP-style specification ... treats co-roots as
  part of the context"
  → contradicts the formal definition; should remove or rephrase.
- §7b.2's "1/8" and "1/4" numbers — under the new spec, become different
  values (computed above).
- §7b.4 explanation that sufficiency is degenerate on the desert traveller
  → no longer correct; sufficiency separates poison from shooting under the
  new spec. Need to rewrite to claim the separation honestly.

## Key conceptual lessons

1. **Don't condition $P_{\mathbf{U}}$ on the outcome.** Conditioning on the
   forensic evidence is fine ($u$ identified); conditioning on $Y=1$ too is
   what collapses $\xi$ to a point mass and erases the sufficiency reading.
   This was the original "overfix" bug.
2. **Don't hard-code the non-cause suspect.** Per sec3:451–462, only $\mathbf{C}$
   and $\mathbf{T}$ are intervened in the worlds. Non-cause suspects follow
   the SCM (= drawn from prior for roots). Hard-coding them was the implicit
   convention in the paper's §7b §2 — it gives nice 1/8-style numbers but
   isn't the formal definition.
3. **Mediators CAN be witnesses** (after the thin_search patch). The
   framework was over-filtering; deterministic sites are only excluded from
   the **suspect** pool, not the witness pool.
