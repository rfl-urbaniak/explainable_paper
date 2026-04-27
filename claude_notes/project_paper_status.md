---
name: PCI paper status and reviewer concerns
description: Status of the CLeaR 2026 submission, key reviewer concerns, and what was promised in the rebuttal
type: project
originSessionId: ef5c8a65-3d0b-4a46-930f-d4a4a6819e3d
---
The paper "A Computationally Feasible Framework for Causal Probabilistic Explanation" (PCI, submission #36) was rejected at CLeaR 2026 (March 2026). The authors are Rafal Urbaniak, Sam Witty, Daniel Waxman, Andy Zane, John Feser, Poorva Garg, Drew Lehe, Eli Bingham.

**Why:** Three reviewers (S1iS: reject, 2MT8: marginal, g8o1: marginal). g8o1 was positive but downgraded after rebuttal. S1iS was the hardest reviewer. Key issues were missing comparison to causal SHAP literature, no quantitative real-world results, sufficiency dropped in practice, and rebuttal was seen as "hand-wavy — everything promised but nothing provided."

**How to apply:** When working on revisions, prioritize the concrete deliverables promised in the rebuttal (see below). Don't just discuss changes — implement them.

## Key concerns to address in revision

1. **Comparison to causal SHAP literature** (S1iS, strong) — Papers [1–6] in S1iS's review must be discussed. Particularly: Janzing et al., Sharma et al. (CF-Shapley), Bareinboim et al. (do-interventions). PCI must be formally distinguished.

2. **Quantitative real-world results** (all reviewers) — Section 6 (AVM model) currently has no figures, tables, or metrics. Promised: partial causal graph + both SHAP and PCI attribution values. Constrained by external contract — can only show partial structure.

3. **Sufficiency dropped in experiments** (S1iS) — The ci function used in Section 4 and experiments only uses necessity (Y^n ≠ Y*), dropping Y^s entirely. This undermines the main theoretical claim. Must validate full PCI with sufficiency.

4. **Method not a formal algorithm** (S1iS, 2MT8) — Section 4 uses bullet points. Promised: convert to formal algorithm.

5. **Theorem conditions unclear** (S1iS) — Theorems 10–11 lack formal specification in main text. Full support condition was misunderstood by reviewer; clarification needed. Also: stone-throwing benchmark may violate conditions if deterministic intermediate variables are in the suspect set.

6. **No running example** (2MT8, g8o1) — Paper is notation-heavy. Promised: expand continuous case into a tutorial running through the paper.

7. **Free parameters without guidance** (2MT8) — ε (excision), K and J (cardinality constraints), causal impact function choice. Promised: guidelines + sensitivity analysis.

8. **Descriptive normality references withdrawn** (S1iS) — Was a motivating claim in the intro; rebuttal said it would be dropped. Intro must be updated accordingly.

9. **PCI requires full SCM** (S1iS, post-rebuttal) — This is a stronger assumption than PN/PNS and causal Shapley. Must be stated explicitly as a scope limitation.

10. **Notation overload** (S1iS) — Y used for two different purposes in Definition 5. Clean up.

## What was promised in the rebuttal (but not yet delivered)

- Add discussion of causal SHAP literature and clarify conceptual/methodological divergence
- Move formal theorem specifications from appendix to main text
- Convert Section 4 method description to a formal algorithm
- Drop descriptive normality references from intro (or replace with something defensible)
- Expand continuous benchmark into a tutorial with SHAP comparison throughout
- Provide graph + PCI + SHAP values for real-world AVM application
- Extend conclusion with future work and scope limitations
- Clarify motivation for definitions in Section 2
- Extend discussion of complexity; clean up ci notation
- Explain why non-zero baseline ci arises naturally from model noise
