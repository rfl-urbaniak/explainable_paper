# Design note: axioms for causal responsibility attribution

The desiderata in Sections 3 and 4.6 are stated relative to specific models. What general principles
about a responsibility measure R would *entail* them, given those model structures? Below I propose
six candidate axioms, each stated model-agnostically, followed by the derivation of the relevant
desiderata.

---

## Axiom A1 — Causal Activity (Non-nullity)

> *If there exists a positive-probability configuration of background variables under which
> intervening to change X alters Y, then R(X → Y) > 0.*

This is a minimal causal relevance condition: anything with non-zero counterfactual leverage
on the outcome must receive positive attribution.

**Entailments in OBCB.**
- *D-A1* (gender for Alice): intervening to set gender=male creates a 90% probability
  of credit evaluation, with non-negligible probability of approval. Gender has non-zero
  counterfactual leverage on loan.
- *D-A2* (credit for Alice): conditional on a counterfactual gender=male intervention,
  changing credit from bad to good raises P(loan=1) substantially. Credit is
  counterfactually active in the intervened world.
- *D-B1* (gender for Bob): setting gender=female drops P(check) from 0.9 to 0.2,
  materially changing the probability of the subsequent rejection chain.
- *D-B2* (credit for Bob): credit=bad directly caused check-failed=1 for Bob;
  setting credit=good would have changed the outcome.

**Entailments in Signal.**
DXY, DMY, DXM all follow: the structural equations $M=X+\varepsilon_M$,
$Y=M+\varepsilon_Y$ mean that changing X shifts M and thereby Y, and changing M
directly shifts Y. All three attributions are strictly positive.

---

## Axiom A2 — Causal Inactivity (Nullity)

> *If there is no directed causal path from X to Y in the model's DAG, then R(X → Y) = 0.*

Under the faithfulness assumption, absence of a directed path entails zero do-calculus
effect of X on Y, so no counterfactual leverage exists.

**Entailments in Signal.**
The DAG is $X \to M \to Y$. There is no path from Y to X, from Y to M, or from M to X.
Axiom A2 directly entails DYX = 0, DYM = 0, DMX = 0.

*(The OBCB desiderata impose no zero-attribution requirements, so A2 has no OBCB entailments.)*

---

## Axiom A3 — Factual Path Priority

> *If X's causal influence on Y was realized through a factually active path in the actual
> world, and Z's causal influence on Y is purely counterfactual (no path from Z to Y was
> activated in the factual world), then R(X → Y) > R(Z → Y), all else equal.*

"Factually active path" means the sequence of events mediated between X and Y actually
occurred in the factual scenario.

**Entailment: D-A-rank.**
For Alice: the causal path gender → loan was *factually realized* (she was rejected outright
because she is female; no credit evaluation took place). The path credit → check-failed → loan
was *never activated* in the factual world — her credit was never evaluated. By A3,
R(gender → loan | Alice) > R(credit → loan | Alice). ✓

**No OBCB violation for D-B-rank.**
For Bob: both paths are factually active (gender opened the check, credit caused the
failure). A3 does not apply directly; D-B-rank requires A4 below.

**Note:** A3 is what PNS violates for D-A-rank. PNS measures probability of necessity
and sufficiency over the joint distribution, giving credit and gender symmetric roles for
Alice, because PNS does not condition on whether a path was factually traversed.

---

## Axiom A4 — Proximity Attenuation

> *Among variables whose causal influence on Y was realized through factually active paths,
> the variable whose active path is shorter (closer to Y in the causal graph) receives
> strictly higher attribution, all else equal.*

"Shorter" means fewer directed edges on the path from X to Y in the DAG.

**Entailment: DMXY.**
In the signal model, both X and M have factually active paths to Y. M's path is
$M \to Y$ (length 1). X's path is $X \to M \to Y$ (length 2). By A4,
R(M → Y) > R(X → Y). ✓

**Entailment: D-B-rank.**
For Bob: gender's factually active path to loan runs through credit and check-failed
(gender → check → check-failed → loan, length ≥ 3). Credit's path to loan runs through
check-failed (credit → check-failed → loan, length 2). By A4,
R(credit → loan | Bob) > R(gender → loan | Bob). ✓

**Interaction between A3 and A4.**
A3 and A4 compose: a counterfactual-only path (A3) loses to any factually active path
regardless of length, and among factually active paths (A4) proximity determines ranking.
Together they give a two-level ordering: factual > counterfactual, and within factual:
proximate > distal.

---

## Axiom A5 — Mechanism Sensitivity

> *The attribution R(X → Y | individual i) depends on the causal mechanism operative for
> individual i, not only on the marginal distribution of X in the population.
> In particular, if X was causally sufficient to produce Y for i (no other variable's
> contribution was needed), while X was only a necessary enabling condition for a further
> proximate cause Z to produce Y for j, then R(X → Y | i) > R(X → Y | j).*

This is an individual-level analogue of the distinction between proximate and enabling causes.

**Entailment: D-comp.**
For Alice: gender=female was *causally sufficient* for loan=0 (the rejection pathway did
not require credit to be evaluated; gender alone determined the outcome in the dominant
case). For Bob: gender=male was a *necessary enabling condition* — it opened the
evaluation — but it was credit=bad that then caused the rejection. By A5,
R(gender → loan | Alice) > R(gender → loan | Bob). ✓

**Note:** This is the hardest desideratum for population-level methods. Any method that
assigns attribution by averaging over the population distribution of Alice/Bob-type
individuals conflates the two roles of gender, and will tend to understate gender's role
for Alice relative to Bob.

---

## Axiom A6 — Realized Efficiency

> *The sum of attributions across all variables equals the deviation of the actual realized
> outcome from its unconditional expectation:*
>
> $$\sum_i R(X_i \to Y) = Y_{\mathrm{factual}} - \mathbb{E}[Y].$$

This extends the standard SHAP efficiency axiom (which requires $\sum_i \phi_i = f(x) -
\mathbb{E}[f(X)]$) to include the contribution of residual noise. The prediction $f(x)$
need not equal $Y_{\mathrm{factual}}$ when the outcome is stochastic; A6 requires the
gap to be attributed.

**Entailment: D-ind.**
In the signal model with $Y=M+\varepsilon_Y$, the prediction $f_Y = M$ explains
$M - \mathbb{E}[M]$, but the realized $Y - \mathbb{E}[Y]$ also includes $\varepsilon_Y$.
Standard SHAP explains only the prediction; by A6, an attribution satisfying D-ind must
account for $\varepsilon_Y$ as well. ✓

**Note:** A6 is strictly stronger than SHAP's efficiency. A method can satisfy SHAP's
efficiency while violating A6; the difference is whether unmodelled noise is
attributed or silently dropped.

---

## Summary: axiom–desideratum correspondence

| Axiom | Desiderata entailed |
|---|---|
| A1 Causal Activity | D-A1, D-A2, D-B1, D-B2, DXY, DMY, DXM |
| A2 Causal Inactivity | DYX, DYM, DMX |
| A3 Factual Path Priority | D-A-rank |
| A4 Proximity Attenuation | DMXY, D-B-rank |
| A5 Mechanism Sensitivity | D-comp |
| A6 Realized Efficiency | D-ind |

---

## Observations

**Logical structure.** Each axiom is necessary for its target desiderata: no smaller set of
the six axioms entails all nine desiderata. A1 and A2 are the weakest — they encode that
the measure tracks causal graph structure at all. A3–A5 encode increasingly fine-grained
distinctions between types of causal roles. A6 is orthogonal to the others.

**PNS failures are precisely A3 and A5 violations.** PNS fails D-A-rank because it does not
condition on whether paths were factually traversed (A3), and fails D-B1 because its
particular form sets gender's value to zero for Bob — a degenerate collapse that A1
is too weak to prevent without A3's factual-path conditioning. D-comp is unrepresentable
in PNS since it is a population-level measure.

**SHAP failures are A3, A4, A5, A6 violations.** Plain SHAP integrates out causal
structure entirely, so factual path priority and proximity are both lost. Causal SHAP
partially restores A2 (via do-calculus conditioning) but does not enforce A3 or the
individual-level A5.

**The witness mechanism is the implementation of A3.** Conditioning on witnesses held at
their factual values is precisely the operation that makes A3 computable: it determines
whether a given path was factually traversed by checking whether the intermediate nodes
took their actual values.
