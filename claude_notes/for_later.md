---
name: For later — deferred tasks
description: Items explicitly deferred for a future revision pass
type: project
---

## Cardinality range (J, K) validation run

The practitioner guidance in Section 3 (after the Variable Selection Distribution gloss)
advises: choose the smallest J, K that satisfy the desiderata; report stability across
cardinality ranges for decision-relevant applications (method 3: sensitivity analysis).

**Why:** A reviewer may ask for principled validation. We need an actual experiment
demonstrating cardinality sensitivity — e.g. on the OBCB or AVM dataset, run PCI with
J=K=1, J=1,K=2, and J=1,K=|S| and show that (a) rankings stabilise or (b) where they
diverge, higher K is needed to satisfy a desideratum. This would concretely back up the
claim that the desiderata serve as a selection criterion.

**How to apply:** Add a cardinality sensitivity table or figure to the experimental
section. Then update the practitioner guidance forward-reference in sec:causal_impact to
point to the specific experiment.

---

## Joint support condition — address in theorem section (sec:th_ac)

Reviewer S1iS raised a specific concern: if S contains causally related variables
(e.g. both "Sally throws" and "bottle is hit"), the joint alternative distribution
Δ_C(s*) may assign positive mass to structurally impossible combinations, violating
the full support precondition of Theorems 10–11. The individual exclusion condition
x ∉ supp(Δ(x)) in the Alternative Value Distribution definition (Section 3) is
insufficient to guarantee joint support.

**Why:** This is a theorem precondition issue, not a definitions issue. Section 3 is
correct as-is. The right place to address it is in the formal statement of Theorems
10–11 in sec:th_ac, with an explicit caveat that S should not contain deterministic
functions of other variables in S, or alternatively that Δ is required to have full
joint support on non-factual values.

**How to apply:** When writing sec:th_ac, add a remark after the theorem statement
clarifying the joint vs marginal support distinction, and note the practical implication:
suspect sets should avoid including both a variable and its deterministic descendants.

---

## Sufficiency in experiments

The current experiments (AC benchmark) drop the sufficiency component of `ci` because actual causality has no sufficiency counterpart — necessity-only allows a direct parallel comparison.

**Why:** Sufficiency will be used in new experiments to be built for this revision. Once those experiments exist, Task 9 (the remark after the PNS example in Section 3) should be updated to name the specific section where sufficiency is used, rather than the placeholder reference currently in the text.

**How to apply:** When the new sufficiency experiments are written up, revisit the note after the PNS binary example (around the Absolute Difference example in sec:causal_impact) and update the cross-reference.
