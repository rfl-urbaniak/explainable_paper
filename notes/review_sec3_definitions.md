# Section 3 Review — Causal Impact Definitions
*Session: 2026-04-27. For continuation next session.*

---

## Context

Section 3 (`sec:causal_impact`, l.535–797 of main.tex) presents the formal development of PCI:
PSCM → Interventions → Interventional Law → Causal set → Variable Selection Distribution →
Alternative Value Distribution → Joint Necessity and Sufficiency Measure → Causal Impact
Function → two examples (PNS binary, absolute difference).

Fixes already applied this session (do NOT re-apply):
- Acyclicity added to PSCM definition (l.552)
- `\dom(X_k)=\{0,1\}` brace fix in PNS example (l.789)
- Missing "and" in Absolute Difference example (l.793)
- T∩C=∅ constraint added to both sums in Joint Necessity and Sufficiency Measure (l.710, 719)
- Factual condition added to Actual Cause definition (l.329)

---

## Reviewer Concerns That Touch Section 3

### S1iS (strong reject)

**Comment 2a/b — Variable Selection Distribution (no guidance on J, K):**
The Cardinality-Constrained Uniform Selection example (l.611–616) gives no reason to
prefer it or guidance on J and K. S1iS notes that at J=1, K=|X|, Example 1 is
*mathematically equivalent to Shapley weighting* — the rebuttal didn't rebut this.
**Action needed:** Either add a clear distinguishing argument in the text (why PCI's
weighting differs from Shapley), or acknowledge the connection and explain when/why
to choose J≠1 or K<|X|. The current example just states the distribution without
any motivation for *why* you'd constrain cardinality.

**Comment 3 — ε-excised distribution (no motivation):**
The example (l.655–668) defines the distribution formally but gives no argument for
why to use it over the observational distribution (as SHAP does) or a fully conditional
posterior. The prose at l.637–638 nods at this ("semantics for 'alternative value'")
but the argument is informal. S1iS in follow-up says this was not resolved.
**Action needed:** Add 2–3 sentences after the ε-excised example explaining the
practical trade-off: observational distribution conflates signal with correlation;
fully conditional posterior is too tight (overfits to noise); ε-excised hits a
middle ground. Connect to the Bayesian motivation in the rebuttal to 2MT8.

**Comment 4 — sufficiency dropped in practice:**
The PNS binary example (l.773–790) correctly shows the ci function with both
Y^s and Y^n. But S1iS's concern is that Section 4 (the method) and all experiments
use only necessity (Y^n ≠ Y*), dropping Y^s. Section 3 doesn't pre-empt this.
**Action needed:** Add a brief remark after the PNS example noting that the full
ci function (with sufficiency) is computationally validated in the theoretical
results (sec:th_ac) and both components are used in the binary PNS case. The
absolute difference example also uses both. Alternatively, note that ci is a
user-defined function and experiments choose a ci that focuses on necessity for
specific reasons — and state those reasons here.

**S1iS follow-up — PCI requires full SCM:**
Not currently stated in Section 3. The PSCM definition implicitly requires full
structural equations, but this isn't called out as a scope limitation.
**Action needed:** Add a sentence after the PSCM definition or in the intro to
Section 3 noting that PCI, unlike PN/PNS, requires the full structural model
(equations + noise distributions) to sample counterfactuals — this is stronger
than what's needed for PN/PNS bounds from observational data. This is a scope
limitation, not a defect, but must be stated explicitly.

### 2MT8 (marginal)

**Weakness 3 — Too many free components:**
The section introduces three free components with no guidance: Alternative Value
Distribution, Variable Selection Distribution, Causal Impact Function. The
section needs at least one sentence per component explaining how a practitioner
should think about choosing it.
**Action needed:** After each definition or example, add 1–2 sentences of
practical guidance. See specific notes below.

**Weakness 4 — Properties of the method:**
Section 3 defines the machinery but doesn't state what properties the resulting
causal impact has (e.g., it generalises PN/PS/PNS, it recovers AC under certain
conditions). The PNS example gestures at this but the section needs a brief
"what PCI gives you" paragraph before or after the main definition.

### g8o1 (marginal, positive)

**Weakness 1 — No running example:**
Section 3 builds entirely in the abstract; the OBCB example from Section 2 is
never referenced again. A reader coming from Section 2 loses their grounding
the moment Section 3 starts. This was promised in the rebuttal (expand continuous
case into running tutorial) but not yet delivered.
**Action needed (larger task):** Thread the OBCB stochastic example through
Section 3. After each key definition, show what it means for the OBCB model.
E.g., after Joint Necessity and Sufficiency Measure, work through
P^s_gender(·|u) for Alice's case. This is the single most impactful change
for accessibility and directly addresses both g8o1 and 2MT8.

---

## Grad Student Accessibility Issues

### 1. Duplicate S/W/C/T introduction (l.582–601)

The suspects/witnesses/active-suspects/active-witnesses terminology is introduced
*twice* in almost identical prose: once at l.582–591 and again at l.598–601. This
is confusing — a reader wonders if there's a subtle difference. Merge into one
clean paragraph, then proceed to the definitions.

### 2. Variable Selection Distribution definition (l.604–606) — notation tangle

Current text:
> "Let Y be a random set of random variables that is distributed according to Γ(2^X)
> such that X ⊇ Y ~ Γ(2^X) and probability mass function p^Γ(Y) = P(Y = Y)."

Problems:
- `X ⊇ Y` is redundant (Y ∈ 2^X already means Y ⊆ X)
- `Y ~ Γ(2^X)` appears in the constraint clause and reads circularly
- `P(Y = Y)` has Y on both sides with different roles (random variable vs value)
- The curly script Y (Y) is introduced but then abandoned for plain Y

Suggested rewrite:
> "A variable selection distribution Γ is a probability distribution on the power
> set 2^X. We write Y ~ Γ to mean Y is a random subset of X drawn from Γ, with
> probability mass function p^Γ(Y) = P(Y = Y) for Y ⊆ X."

### 3. Joint Necessity and Sufficiency Measure — no prose scaffold inside definition

The definition jumps directly from notation setup into the formulas. A grad student
reading this for the first time has to simultaneously decode: 2^S_k notation,
restriction maps, pushforward of Δ_C, two nested sums, an integral, and the meaning
of the do() arguments. There's no signpost saying "the sufficiency measure fixes C
to its factual value; the necessity measure integrates over alternative values for C."

**Action needed:** Add 2 sentences before the align block:
> "The *sufficiency measure* P^s_k(A|u) fixes C to its factual values and T to
> factual context, then asks how likely A is. The *necessity measure* P^n_k(A|u)
> integrates over alternative values c' for C, again with T fixed at factual context."

### 4. `\linebreak` hack in Causal Impact Function definition (l.759, 761)

Two forced `\linebreak`s inside a definition box are a sign the sentence structure
is fighting the column width. Reword to avoid them.

### 5. The Interventional Law definition (l.564–579) — Dirac not named

The definition says "this will be a Dirac measure" in the prose before the definition,
but the formula just shows an indicator function. A grad student who doesn't recognise
the Dirac structure may not see the connection. Add a parenthetical: "(i.e., a Dirac
measure at Y'(u))" after the indicator formula.

### 6. No connection back to Section 2 (OBCB) — see reviewer note above

---

## Priority Order for Next Session

1. **[HIGH, S1iS + g8o1 + 2MT8]** Thread OBCB example through Section 3 — after
   Joint Necessity and Sufficiency Measure, show what P^s_gender and P^n_gender look
   like for Alice's case concretely. This is the promised running tutorial.

2. **[HIGH, S1iS]** Add scope limitation statement: PCI requires full SCM
   (structural equations + noise), stronger than PN/PNS. Place after PSCM definition
   or in the section intro.

3. **[MEDIUM, S1iS + 2MT8]** Motivate ε-excised distribution vs observational
   (2–3 sentences after the example). Draw on the Bayesian argument in rebuttal to 2MT8.

4. **[MEDIUM, S1iS follow-up]** Address the Shapley-weighting equivalence concern
   at J=1, K=|X| — either distinguish PCI's weighting formally or acknowledge
   connection and explain when to deviate.

5. **[MEDIUM, grad student]** Fix Variable Selection Distribution definition notation
   (circular, redundant X ⊇ Y, Y vs Y clash).

6. **[MEDIUM, grad student]** Merge the two S/W/C/T introduction paragraphs into one.

7. **[LOW, grad student]** Add prose scaffold inside Joint Necessity and Sufficiency
   Measure before the align block.

8. **[LOW, grad student]** Fix `\linebreak` hacks in Causal Impact Function definition.

9. **[LOW, S1iS comment 4]** Add note clarifying that Section 4's ci is a specific
   choice and that the full PCI (with sufficiency) is instantiated in the PNS example.

---

## State of main.tex at End of Session

- All five numbered fixes from earlier today are in and compiled clean.
- Current compile: no errors, only undefined reference warnings (expected —
  sec:th_ac, sec:ac_benchmark, sec:con_benchmark, sec:avm, app:DCE all missing).
- Tab:pn_ps_pns corrected: Bob PN(gender)=0, PS(credit)=0.955, PNS(credit)=0.86.
- PN narrative corrected: 0.43 and 0.53 (not "roughly the same 0.43 and 0.45").
- Alice PS(credit) phrasing corrected to P(loan=1|female,bad)=0.
- Bob PS(credit) = 0.955 now discussed in text.
- But-for single, Attempt 1, Actual Cause all have explicit factual condition.
- obcb_computations.ipynb re-executed and correct; 4 discrepancies documented.
