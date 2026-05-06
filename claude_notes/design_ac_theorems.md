# Design note: Section 6 (Relation with Actual Causality) — revisions

**Date:** 2026-05-06
**File:** `sections/sec6_actual_causality.tex` (just copied verbatim from
`outdated_iteration_2/outdated_main.tex` lines 597–695, wired into `main.tex` after
`sec5_synthetic`)

The section currently uses the old explanation-extended-distribution notation
($P^E$, $P^{\mathit{set}}$, suspect bias $b_s$, witness bias $b_w$) which does not
appear anywhere else in the current paper. Sec3 was rewritten in PCI vocabulary
(PSCM, Variable Selection Distribution, Alternative Value Distribution, Causal
Impact Function). Bridging that gap is the main structural task.

---

## A. Compile-blocking / mechanical fixes

- [ ] **`corollary` env undeclared.** Section uses `\begin{corollary}` (Cor.
      `cor:marginalize`) but `main.tex` only declares `theorem`, `definition`,
      `example`, `observation`, `attempt`. Add
      `\newtheorem{corollary}[theorem]{Corollary}` to `main.tex:6-10`, or
      downgrade to `theorem`.
- [ ] **Stray "appendix" reference.** Opening paragraph: "for proofs, see the
      appendix." No appendix in `main.tex`. Restore proofs section or rephrase.
- [ ] **Inconsistent bold on cause symbols.** Mixed `\mathbf{C}` vs. bare `C` /
      `C^\dagger` in Theorems `th:ac-exp`, `th:local_max_to_ac` and Obs.
      `obs:misalignment`.
- [ ] **Eq. `eq:pect`:** `(C, T)` should be `\mathbf{C}, \mathbf{T}`.
- [ ] **Theorem `th:local_max_to_ac` typo:** `$p_s>0$` → `$b_s>0$` (`p_s`
      undefined). Same in Cor. `cor:marginalize`.
- [ ] **Subscript variable mismatch in Obs. `obs:misalignment`:** `\mathbf{C}
      \subseteq {X}` — capital `X` undefined; should be `\mathbf{S}` or
      `\{\mathbf{S}=\mathbf{s}\}`.
- [ ] **Double `\label`** on Theorem `th:exp-ac`: has both `\label{th:exp-ac}`
      and `\label{th:if_pr_then_nec}`. Pick one.
- [ ] **Casual prose pass** ("Let's start with...", "we wrap up...") — out of
      register vs. sec3/sec4. Defer to after content stabilizes.
- [ ] **Drop commented-out blocks** (observation stub, proof stub, "Questions"
      section).

## B. Notational integration with current sec3 (the big decision)

Old section uses
$P^{E}_{b_s, b_w, \varphi, \{\mathbf{S}=\mathbf{s}\}, \mathbf{V}}(\mathbf{u}, \mathbf{v}, \mathbf{C}, \mathbf{T})$,
$P^{\mathit{set}}_{b_s,b_w,\ldots}$, suspect bias $b_s$, witness bias $b_w$.
Current `sec3_definitions.tex` has none of these.

Two paths:

- [ ] **(B1) Re-derive theorems in current PCI notation.** Principled fix but a
      rewrite. Decide: what plays the role of $b_s$/$b_w$ in the current Variable
      Selection Distribution? What plays the role of $P^E$ vs. PCI?
- [ ] **(B2) Add a short bridge subsection** mapping old → new notation, keep
      theorems in old form. Lighter lift but reads as patched-on.

**Lean toward B1.** Need explicit go-ahead before drafting.

## C. Reviewer-driven content

(refs: `reviews/reviews_structured.md`, `notes/actions.md`)

- [ ] **(9) Theorem support conditions (S1iS pt 5).** Sharpest unresolved point:
      Theorems 10/11 require *full joint* support on alternative distribution,
      not marginal. Stone-throwing: deterministic intermediates (e.g.,
      `H = f(S, ...)`) make this impossible if both `S` and `H` are in suspect
      set. Theorems `th:ac-exp`, `th:exp-ac`, `th:local_max_to_ac` need:
    - explicit statement of support condition on alternative-value dist,
    - either exclude deterministic intermediates from suspect set, or clarify
      condition is on intervention candidates' marginals over admissible
      counterfactual values.
- [ ] **(10) Sufficiency dropped (S1iS pt 4).** Theorems here are
      necessity-only ($\mathbbm{1}(N_\ldots)$). Reviewer asks where full PCI
      with sufficiency connects to AC. Either: (a) extend at least one theorem
      to full PCI, or (b) state explicitly that this section establishes
      necessity-only correspondence and point forward.
- [ ] **(11) Descriptive normality (S1iS unresolved).** Rebuttal said refs
      would be dropped, but AC's modern formulation (Halpern 2016) uses
      normality. Decide:
    - (a) reinstate normality and connect it to suspect bias $b_s$, or
    - (b) explicitly use normality-free AC variant and footnote the choice.
- [ ] **(12) Minimality vs. cardinality misalignment (Obs. `obs:misalignment`).**
      Currently a one-liner; either prove the corrective procedure ("filter for
      subset-minimality") or move it explicitly to future work.
- [ ] **(13) Halpern–Pearl citation.** Definition `def:ac` cites no source. Add
      Halpern (2016) or Halpern \& Pearl (2005).

## D. Structural / placement

- [ ] **(14) Position in paper.** Currently after `sec5_synthetic` (experiments).
      Conventionally "Relation to..." sections come *before* experiments. Decide:
      insert between sec3 and sec4, or keep at end as "discussion/connections".
- [ ] **(15) Transition out.** Closing line ("...move to the discussion of a
      related, more probabilistic, and more scalable notion") pointed forward to
      PN/PS/PNS, which is now *upstream* (sec3). Rewrite transition.

---

**Suggested sequencing:** A (mechanical) → decide B (notation) → C (reviewer
content) → D (placement). C and D depend on B.
