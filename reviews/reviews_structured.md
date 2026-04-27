# OpenReview: A Computationally Feasible Framework for Causal Probabilistic Explanation
**Venue:** CLeaR 2026 | **Submission:** #36 | **Decision:** Reject (08 Mar 2026)

---

## Decision

**Reject** — Program Chairs, 08 Mar 2026

---

## Reviewer S1iS — Rating: 3 (Clear rejection) | Confidence: 4

### Summary
- PCI provides tractable causal explanations in probabilistic ML models, inspired by Halpern–Pearl actual causality and Pearl's PN/PS/PNS.
- Empirical tests on overdetermination/undercutting benchmarks and a real-world spatio-temporal model.

### Originality
Several methods already claim to tractably provide causal probabilistic explanations [1–6], but no empirical or theoretical comparison is provided. The authors themselves characterize PCI as a "natural expansion" of existing ideas.

### Significance
Tempered by the existence of many prior works addressing the same intersection, none of which are discussed, along with issues with technical specification. No direct comparison with PN/PNS is made.

### Quality
- **Contribution 1 (PCI method):** Novel quantity, but theorems require assumptions not satisfied in experiments.
- **Contribution 2 (benchmarks):** Discrete benchmark acknowledged as "easy" with exploitable structure; no sample complexity analysis; continuous benchmark lacks motivation for stated qualitative expectations.
- **Contribution 3 (real-world):** No quantitative results provided.

### Clarity
- Definitions in Section 2 presented without motivation; notation overloaded (e.g., Y used for two different purposes in Definition 5).
- Theorems 10 and 11 lack formal specification in main text.
- Method in Section 4 described through bullet points, not a formal algorithm; several undefined terms.

### Detailed Comments
1. How does PCI relate to "descriptive normality" [7,8]? Does PCI resolve the counterexamples to standard actual causation put forth there?
2. **(a)** Why use cardinality-constrained distribution rather than uniform over all subsets? **(b)** How to justify choosing J≠1 or K≠|X|?
3. Why prefer ε-excised alternative distribution over observational distribution (as in SHAP literature)?
4. The ci function used in experiments ignores sufficiency entirely; Section 4 also explicitly drops sufficiency. Where is full PCI (with sufficiency) validated?
5. Theorems require alternative distribution "fully supported on non-factive values" — the stone-throwing benchmark appears to violate this.
6. Undefined terms: "existential necessity claim witness", "active antecedent", "reject-condition", "blame".
7. Real-world section provides no figure, table, or quantitative metrics despite claiming "performance results." DAG of 25 nodes/99 edges mentioned but not shown. Model details withheld due to "commercial work."

### References cited
1. Causal SHAP values (NeurIPS)
2. Causal Shapley values (NeurIPS)
3. Relevance quantification in XAI (AISTATS)
4. Graph-based approach to interpreting model predictions (AISTATS)
5. On measuring causal contributions via do-interventions — Bareinboim et al. (ICML 2022)
6. Counterfactual-Shapley Value (NeurIPS Workshop)

---

### Reviewer S1iS — Follow-up Comment (15 Feb 2026)

Thanks for partially addressing: causal Shapley literature (P1), argument against uniform sampling (P3a), theorem conditions (P6). None fully resolved. Score maintained.

**Additional concern:** PCI requires the complete causal model (structural equations + noise distributions) to sample counterfactuals. This is strictly stronger than PN/PS/PNS (which can be bounded from observational/interventional data) and causal Shapley methods. This unstated assumption qualifies the paper's claims of computational feasibility.

**Specific unresolved points:**
- **Shapley weighting:** At J=1, K=|X|, Example 1 is mathematically equivalent to Shapley weighting (Eq. 8, Lundberg et al. 2017). The rebuttal's claimed distinction does not hold. K remains a free parameter with no selection criterion.
- **Sufficiency dropped:** The rebuttal's distinction from causal SHAP rests on the sufficiency component Y^s, yet Section 4 drops it entirely.
- **Stone-throwing benchmark:** Witnesses W_i are determined deterministically by A_i; if both are in the suspect set, joint alternative distribution cannot have full support.
- **ε-excision:** Not formally motivated over observational distribution; no guidance for choosing ε or K.
- **Descriptive normality:** The rebuttal states these references will be dropped because "discussion is too involved." This withdraws a motivating claim without replacing it.

---

## Reviewer 2MT8 — Rating: 5 (Marginally below threshold) | Confidence: 3

### Summary
PCI is computationally suitable while producing causal explanations, based on PN/PS/PNS concepts. Computational feasibility tested at scale.

### Strengths
- Clear, consistent notation throughout main text and appendices.
- Good appendix section providing pedagogical connection to Pearl and Halpern.
- Universal method: no restrictions on variable domain, model form, distributions, or structural equations.
- Approximate PCI handles models with 110–150 variables while exact actual causality times out at 17 variables.
- Continuous variables handled well in the Overdetermination/Undercutting benchmark.

### Weaknesses
1. No guidance on choosing the Alternative Value Distribution (e.g., ε-ball size).
2. Variable Selection Distribution not discussed in depth; no analysis of sensitivity to choice.
3. Too many free components (Alternative Value Distribution, Variable Selection Distribution, Causal Impact Function) without guidance on what constitutes a good choice.
4. Hard to characterize what properties the method has beyond being causal. More theoretical guarantees needed.
5. Section 6 (real-world AVM): too few details, comparison to SHAP is not quantitative, and qualitatively insufficient to assess claimed advantages.

---

### Authors' Rebuttal to 2MT8 (07 Feb 2026)

- Will expand continuous example into a tutorial-style example with step-by-step PCI construction and parallel SHAP comparison.
- **Alternative distribution:** Bayesian motivation — upstream-informed posteriors, not downstream-constrained. Alternatives discussed: uniform (unrealistic), fully conditional (too tight), marginal (less local).
- **Benchmark complexity:** The stone-throwing structure is conceptually hard (overdetermination + undercutting generating asymmetry) yet formally tractable because valid witness sets are closed under supersets.
- **Theoretical properties already identifiable:** distinguishes necessity from sufficiency; breaks symmetries via mediation-inspired witness blocking; composable with Bayesian models; generalizes AC, PN, PS, PNS.

---

### Reviewer 2MT8 — Follow-up Comment (12 Feb 2026)

Appreciates discussion of Alternative Value Distribution choices and the outline of theoretical properties. No further concerns raised.

---

## Reviewer g8o1 — Rating: 5 (Marginally below threshold) | Confidence: 4

*Title: "Nice work though formalism heavy"*

### Summary
PCI reframes causal explanation as estimation over an expanded probabilistic causal model, enabling scalable approximation via Monte Carlo. Theoretical connections to actual causality and PN/PNS carefully discussed with formal partial-equivalence results.

### Strengths
- Well-motivated synthesis of actual causality and probabilistic causation.
- Novel reframing of explanation as an expectation over causal impact functions.
- Careful formal results showing partial equivalence under reasonable assumptions.

### Weaknesses
1. No running example throughout the paper — the work is theoretically heavy and notation-dense; an example-driven presentation would greatly improve accessibility.
2. Housing example relies primarily on agreement with domain expert intuition; more systematic evaluation needed.

*Overall: "I think the paper makes a nice contribution and should be accepted."*

---

### Authors' Rebuttal to g8o1 (07 Feb 2026)

- Will expand the continuous case into a running tutorial referenced throughout the main body.
- Will include quantitative results for the real-world application (constrained by external contract): part of the causal graph, with both SHAP and PCI attribution. SHAP assigns almost all responsibility to downstream variables of limited explanatory role; PCI assigns non-trivial role to upstream variables (living area, number of rooms, etc.).
- Will systematically apply SHAP across all examples and include comparisons.
- Will extend conclusion with explicit future work discussion and scope limitations.

---

### Reviewer g8o1 — Follow-up Comment (16 Feb 2026)

> "Although the authors have promised to take my comments into account, I feel that the answers are a bit hand-wavy. Everything is promised but nothing is provided."

Inclined to reduce rating. Paper needs another round of review with incorporated changes before consideration for acceptance.

---

## Summary of Key Concerns Across Reviewers

| Concern | S1iS | 2MT8 | g8o1 |
|---|:---:|:---:|:---:|
| Missing comparison to causal SHAP literature | ✓ | | |
| No quantitative results for real-world section | ✓ | ✓ | ✓ |
| Sufficiency dropped in practice | ✓ | | |
| Free parameters without selection guidance (ε, K, J) | ✓ | ✓ | |
| No running example / accessibility | | ✓ | ✓ |
| Method not presented as formal algorithm | ✓ | | |
| Theorem conditions unclear / not satisfied in experiments | ✓ | | |
| Requires full SCM — stronger assumption than claimed | ✓ | | |
| Descriptive normality motivation withdrawn | ✓ | | |
