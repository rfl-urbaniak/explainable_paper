# Section 5 (Synthetic) — design and revision plan

Working doc for the redesign of `sections/sec5_synthetic.tex` and
`docs/source/synthetic_explanation.ipynb` in light of CLeaR 2026 reviewer concerns. Nothing in here is
committed yet — entries are proposals or open questions until marked **DECIDED**.

---

## 1. Goals driving the revision

Reviewer-derived obligations this section must discharge:

- **G1 (rebuttal commitment to all three reviewers).** Add an off-the-shelf SHAP comparison, run
  systematically through the section.
- **G2 (S1iS).** Validate the **sufficiency** component, not only the combined `total` score. The
  rebuttal argued that sufficiency is what distinguishes PCI from cSHAP; the section currently never
  shows it.
- **G3 (S1iS).** Make explicit that suspects are confined to root variables. Argue that **with
  deterministic downstream mediators no causal information is lost** by this restriction (root
  values determine everything downstream). Gesture at a theorem to be stated in another section that
  formalises this; here we just state the consequence and the assumption that licenses it.
  *(2026-05-05 user direction.)*
- **G4 (S1iS).** Define the causal-impact function precisely in this section. **Plan: revert to
  reporting necessity and sufficiency components separately rather than a combined `total`.** Be
  explicit about which $ci$ we use and why. *(2026-05-05 user direction.)*
- **G5 (g8o1).** Make this section the *running tutorial* the rebuttal promised — one observation
  walked through step by step, then the batch.
- **G6 (S1iS / minor editorial).** Credit Lundberg et al. (2017) — explicit citation when SHAP is
  introduced in §5, and a sentence acknowledging that at $J=1, K=|X|$ PCI's Example 1 is
  mathematically equivalent to Shapley weighting (Eq. 8 of Lundberg et al. 2017). Also check
  Section 4 / Example 1 and add the credit there if Shapley weighting is referenced.
  *(2026-05-05 user direction: A6.)*

Pedagogical goal added in discussion:

- **G7.** The model should have variables that differ clearly and intuitively in their necessary vs
  sufficient powers, so the N/S decomposition is visible in the figures.

---

## 2. Critical review of the current state — fixes vs improvements

(From the initial pass on the existing `.tex` and notebook.)

### Fixes — required to honour rebuttal commitments

- **F1.** No SHAP / cSHAP overlay. Promised in all three rebuttals.
- **F2.** Sufficiency component computed in code but never plotted; only `total` is shown. **Plan:
  switch to reporting necessity and sufficiency separately** as the primary outputs (per
  G4 / 2026-05-05 user direction).
- **F3.** `ci(V)` undefined in section 5. **Plan: state the chosen impact function explicitly,
  arguing for it, and report its necessity and sufficiency components separately rather than a
  combined `total`.**
- **F4.** Section does not state that suspects are root variables (so full-support holds). **Plan:
  state the restriction; argue that with deterministic downstream mediators no causal information is
  lost (root values determine all downstream values); cross-reference the formal theorem in another
  section.**
- **F5.** No formal algorithm presented; section forwards to `\ref{sub:amv}` and `\ref{sec:scaling}`.
- **F6.** Lundberg / Shapley weighting not credited. **Plan: cite Lundberg et al. (2017) explicitly
  when SHAP is introduced; add the sentence about $J=1, K=|X|$ Shapley equivalence; check Section 4
  / Example 1 for the same credit.**

### Fixes — issues internal to section / notebook

- **F7.** Broken sentence at `sec5_synthetic.tex:41`. Grammar; "potentially the" is a typo.
- **F8.** Overstated generalisation claim at `sec5_synthetic.tex:20`. "Disjunction → addition"
  captures undercutting but **not overdetermination** — neither $B$ nor $D$ alone "suffices" for $E$.
  Either temper the claim or fix the model. (G7 addresses this directly via redesign.)
- **F9.** "Bill" / "Billy" inconsistency.
- **F10.** `cs_win`, `cs_dot` typeset as multiplication in the align block. Use `\mathrm{}`.
- **F11.** Forward references on l.16 are opaque to a tutorial reader.
- **F12.** Caption of `\ref{fig:synthetic_summary}` mis-describes the third desideratum (filter is on
  scores, comparison is on factual values; current wording suggests filter is on coefficient ordering).
- **F13.** Threshold "<2" for "approximately equal" is unjustified — the same "free parameter without
  selection criterion" pattern reviewers flagged.
- **F14.** No qualitative-claim table in main text; only a barplot figure. A small table is easier to
  cite and to extend with a SHAP column.
- **F15.** Notebook expectations list mentions `ci(C3)` but the model has `C0, C1, C2`. Same bug
  inherited by `.tex` l.43.

### Improvements — discretionary, would strengthen the case

- **I1.** Search-sample-size sensitivity (e.g. 1k vs 5k vs 10k).
- **I2.** $\epsilon$ sensitivity sweep; surface the value currently buried inside `ThinSearchSampler`.
- **I3.** State the inactive-$C$ desideratum as "scores near zero" rather than "$\approx ci(A_0)$".
- **I4.** Make the section structurally a tutorial: model → desiderata → step-by-step on one event →
  SHAP comparison on same event → batch.

---

## 3. SHAP integration plan

(Proposed; pending the redesign decision in §4.)

### Step 0 — Dependency
Add `shap` to `dependencies` in `pyproject.toml`. **OPEN: confirm with user.**

### Step 1 — Model wrapping
Wrap the deterministic mapping from feature inputs to $E$ as a plain Python function `f(X) -> y`
(numpy / torch). **Recommendation: re-implement the deterministic part directly** rather than running
SHAP through Pyro `condition`, because (a) much faster and (b) the equations are five lines.
**OPEN: user to confirm approach.**

### Step 2 — Explainer
With $d \le 8$ features, use `shap.ExactExplainer` (enumerates all $2^d$ coalitions, exact, fast).
Background: the existing 500 generated units. **OPEN: confirm.**

### Step 3 — Case-study comparison
For each of the 1–3 case-study observations, compute SHAP $\phi$ vector and produce a parallel figure
mirroring the PCI histogram layout. **Recommendation: separate mirror figure** rather than overlaying
SHAP scalars on PCI histograms. **OPEN: confirm.**

### Step 4 — Batch comparison
Compute SHAP $\phi$ for all 500 units. Two parallel figures per existing PCI summary figure.
Run the qualitative-desiderata frequency check using SHAP scores in place of PCI scores.

### Step 5 — Narrative
Three short markdown cells: SHAP setup intro; case-study readout; batch readout. Drafted *after*
figures exist.

### Step 6 — cSHAP — **IN SCOPE** (2026-05-05 user direction)
Implementation plan in §4.4. Reuses existing `SearchableModel` / intervention machinery so the
implementation generalises beyond this section to AVM and other examples.

### Step 7 — `.tex` updates following the notebook
Done after notebook figures exist; concrete `.tex` diffs proposed at that point.

---

## 4. Model redesign for clear N/S contrasts (G7)

Three options on the table.

### Option A — Minimal modification
Keep $A_0, A_1, A_2$ and the `cs_win` undercutting; add an overdetermined pair $O_1, O_2$ via
$\max(5 O_1, 5 O_2)$ as an additional branch in $E$.
- 8 variables total.
- Pros: minimal disturbance to existing narrative; Sally/Billy continuity preserved.
- Cons: three branches to explain; broken symmetry of noise distributions.

### Option B — Replace undercutting with overdetermination
Drop $C$s and `cs_win`; replace with an overdetermined pair via `max`.
- 5 variables: $A_0$ (decoy), $A_1, A_2$ (linear N+S), $O_1, O_2$ (overdetermined).
- Pros: cleanest two-archetype contrast.
- Cons: loses undercutting demonstration (covered in discrete benchmark anyway).

### Option C — Full three-archetype model — **DECIDED 2026-05-05**
Three pairs covering all archetypes in a single example:
$$
E = 5 L_1 + 10 L_2 + \max(5 O_1, 5 O_2) + 5 (P_1 + P_2) \cdot \mathbb{1}\{|L_2| < \tau\}
$$
- $L_1, L_2$: linear non-redundant (N+S; different magnitudes).
- $O_1, O_2 \sim \mathcal{N}(1,1)$: overdetermined via max (S, not N).
- $P_1, P_2 \sim \mathcal{N}(0,1)$: preempted when $|L_2|$ large (neither in that regime; N+S in the
  other).
- $\tau$ chosen so preemption is active ~half the time, giving both regimes in the 500-event sweep.
- Pros: all four archetypes visible; sharpest answer to S1iS-A2; SHAP can't separate S-not-N from
  N-not-S, PCI can — strongest visual argument vs SHAP.
- Cons: most disruptive; loses literal Sally/Billy mapping (recoverable verbally).

**Constraint added in same conversation:** keep evaluation patterns *similar to current* — three
case-study observations, batch over 500 events, qualitative-desiderata frequency barplot. No new
plot families introduced (so D3's 2D N-vs-S scatter is dropped from the default plan; can revisit
later if histograms turn out not to show the contrast clearly enough).

### Open decisions
- **D1.** Which design? **DECIDED: C.**
- **D2.** Single tutorial observation vs three case-study observations. **DECIDED implicitly via
  "similar evaluation patterns": three case-study observations**, chosen to highlight the three
  archetypes (proposed selection criteria in §4.1 below).
- **D3.** Add a 2D necessity-vs-sufficiency scatter per variable in addition to histograms?
  **DECIDED: no, dropped** to keep evaluation patterns similar. Reconsider if needed.
- **D4.** Sally/Billy framing: keep + rename variables to match, or drop and use generic letters?
  **OPEN.** Mild preference for using descriptive letters ($L$, $O$, $P$) and reframing the
  Sally/Billy story as motivation in the opening paragraph rather than as a literal variable
  mapping. Generic letters make the archetype labelling easier in figure legends.

### 4.1 Detailed plan for Design C

**Variables and structural equations.**
$$
\begin{aligned}
L_1 &\sim \mathcal{N}(0, 1) \\
L_2 &\sim \mathcal{N}(0, 1) \\
O_1 &\sim \mathcal{N}(1, 1) \\
O_2 &\sim \mathcal{N}(1, 1) \\
P_1 &\sim \mathcal{N}(0, 1) \\
P_2 &\sim \mathcal{N}(0, 1) \\
\text{lin} &= 5 L_1 + 10 L_2 \\
\text{od}  &= \max(5 O_1,\, 5 O_2) \\
\text{preempt} &= \mathbb{1}\{|L_2| > \tau\} \\
\text{p\_branch} &= 5 (P_1 + P_2) \cdot (1 - \text{preempt}) \\
E &= \text{lin} + \text{od} + \text{p\_branch}
\end{aligned}
$$
$\tau$ to be calibrated empirically so preemption is active in ~50% of the 500 events. With
$L_2 \sim \mathcal{N}(0,1)$, $\Pr(|L_2| > 0.674) \approx 0.5$, so **$\tau = 0.674$** is a good
starting point.

**Suspect set:** root variables only — $\{L_1, L_2, O_1, O_2, P_1, P_2\}$. This is exactly six
suspects, which keeps SHAP `ExactExplainer` cheap ($2^6 = 64$ coalitions). Justification of the
restriction: §4.3 below.

**Per-event archetype expectations (with $L_2$ factually large, i.e. preempted regime):**

| variable | necessity | sufficiency | role |
|---|---|---|---|
| $L_1$        | moderate    | moderate    | linear N+S, smaller weight |
| $L_2$        | high        | high        | linear N+S, larger weight (also gates $P$) |
| $O_1$ (winner) | low–mid   | high        | overdetermined: pinned, dominates max |
| $O_2$ (loser)  | low       | low         | overdetermined: not pinned, partner carries |
| $P_1$        | ~0          | ~0          | preempted: gated off |
| $P_2$        | ~0          | ~0          | preempted: gated off |

**Per-event archetype expectations (with $L_2$ factually small, preemption inactive):**

| variable | necessity | sufficiency | role |
|---|---|---|---|
| $L_1$        | moderate    | moderate    | linear N+S |
| $L_2$        | moderate    | moderate    | linear N+S, smaller because $|L_2|$ small |
| $O_1$ (winner) | low–mid   | high        | overdetermined |
| $O_2$ (loser)  | low       | low         | overdetermined |
| $P_1$        | mid         | mid         | additive, both regimes-dependent |
| $P_2$        | mid         | mid         | additive |

These tables are the **scientific content** of the section. The figures should let a reader read off
roughly these patterns from the data.

**Case-study selection (mirrors current notebook's three-cases approach).**

Three observations to walk through, chosen by filters on the 500-event sample:

- **Case 1 — preempted regime, contestable max.** $|L_2| > \tau$, $|O_1 - O_2| < \delta$ (both
  contributors close, the max winner is *barely* the winner). Showcases overdetermination at full
  symmetry.
- **Case 2 — preempted regime, dominant winner in max.** $|L_2| > \tau$, one of $O_1, O_2$ much
  larger than the other. Shows that the loser of the max gets near-zero scores.
- **Case 3 — unpreempted regime.** $|L_2| < \tau$, $P_1, P_2$ active. Shows the regime-flip: $P$s
  go from "neither" to N+S; everything else roughly stable.

**Qualitative desiderata for the 500-event batch (six checks; mirrors current four-desiderata
plot but extended).** **OPEN — proposed list, please review:**

1. $ci(L_1) < ci(L_2)$ (linear ordering by coefficient).
2. $ci(O_\text{winner}) > ci(O_\text{loser})$, where winner = $\arg\max(5 O_1, 5 O_2)$.
3. **Sufficiency-specific:** sufficiency-component score of $O_\text{winner}$ exceeds sufficiency
   of $L_2$. (Captures S-not-N character of the overdetermined pair.)
4. **Necessity-specific:** necessity-component score of $L_2$ exceeds necessity of
   $O_\text{winner}$. (Captures that $L_2$ is more necessary than the overdetermined winner.)
5. In preempted regime ($|L_2| > \tau$): $ci(P_1), ci(P_2)$ are below some small threshold.
6. In unpreempted regime: $ci(P_1) \approx ci(P_2)$ (close in absolute difference, picking up the
   symmetry of the additive pair).

Compared to the current four desiderata, items 3 and 4 are the **new and important** ones — they
are the sufficiency-vs-necessity contrasts the reviewer specifically asked for.

**SHAP comparison overlays** to be added to each plot per §3. Specific expectations:

- SHAP for the preempted-regime $P$s should be near zero (correct).
- SHAP for the overdetermined $O$ pair will likely give roughly equal attribution to both — it
  cannot separate winner from loser without causal structure. This is the single sharpest
  qualitative difference from PCI.
- SHAP for $L_1$ vs $L_2$ should respect the coefficient ordering (likely matching PCI's necessity
  component closely).

### 4.2 Causal impact function — choice and presentation (G4)

**Decision (2026-05-05).** Report **necessity and sufficiency components separately** rather than
collapsing into a combined `total`. The current notebook computes `e_suff` and `e_nec` and folds
them through `abs_diff_score`; we will keep the underlying computation but expose the two
components as the primary attribution outputs.

**Why this matters.** S1iS's strongest unresolved concern is that sufficiency is "the component the
paper never validates" — the rebuttal claimed it as the differentiator from cSHAP, then dropped it
in §4. Splitting the report makes the differentiator visible and falsifiable.

**Notation to use in §5.** Define:
- $ci_N(V)$ — necessity component: roughly $\mathbb{E}\,|E_{\text{do}(V = v')} - e|$ over alternative
  values $v'$ for the suspect $V$, with appropriate witnesses held fixed.
- $ci_S(V)$ — sufficiency component: roughly $\mathbb{E}\,|E_{\text{do}(V = v,\,\text{others}=
  \text{alt})} - e|$, i.e. how much $E$ stays near factual $e$ when $V$ is pinned and others vary.

(Exact definitions to mirror what `ThinSearchSampler` + `condition_on_interventional_regime`
actually compute. To be tightened against the code when we draft the prose.)

The figures will then show *two* histograms / two bars per variable — one for $ci_N$, one for
$ci_S$ — rather than one collapsed score. This roughly doubles figure real-estate; we accept that
as the cost of validating both components.

**Combined score, if shown at all.** Optional in an appendix or a single summary panel; not the
headline number. **OPEN — confirm with user once first draft of figures exists.**

### 4.3 Suspect restriction to root variables (G3)

**Argument to include in §5.**

We restrict the suspect set to the **root (exogenous) noise variables**
$\{L_1, L_2, O_1, O_2, P_1, P_2\}$ and *not* the deterministic mediators ($\text{lin}, \text{od},
\text{p\_branch}$, etc.). Two reasons:

1. **Information completeness.** All non-root variables are deterministic functions of the roots.
   Fixing the root values fixes every downstream value, so any intervention on a downstream
   variable can be expressed as some intervention on the roots. Restricting suspects to roots
   therefore loses no causal information about how features drive $E$.

2. **Theorem precondition.** Theorems 10–11 require full support for the alternative-value
   distribution at any joint suspect assignment $\mathbf{s}'$. Including a deterministic mediator
   alongside its parents in the suspect set creates impossible joint assignments
   ($P(\mathbf{S} = \mathbf{s}') = 0$ for some $\mathbf{s}'$), violating the precondition.
   Restricting to roots eliminates this — root noise variables are mutually independent and each
   has full support. (S1iS Follow-up flagged this as the unresolved version of the rebuttal's
   "marginal support is enough" reply; the resolution is to confine suspects to roots, not to
   relax the theorem.)

**Forward reference.** We will gesture at — and in §5 cite — a more general statement: *for any
SCM with deterministic structural equations from a set of independent exogenous noise variables,
restricting the suspect set to the exogenous noise loses no expressive power for any intervention
on observable variables.* The formal statement and proof live in another section
(**OPEN: which section? §3 / §4 / appendix? need to add the placeholder theorem**). §5 says one
sentence and points to it.

**Practical consequence in §5.** The notebook code already does this — `suspects = ["A0", "A1",
"A2", "C0", "C1", "C2"]` are the roots — so the restriction is current behaviour. The change is in
the prose: we make it deliberate, justified, and named.

### 4.4 Causal SHAP implementation via existing model infrastructure (2026-05-05 user direction)

**Goal.** Implement cSHAP once, in a way that rides on PCI's existing model-representation
machinery (`SearchableModel`, `ThinSearchSampler`, `condition_on_interventional_regime`), so that
adding cSHAP to a new example doesn't require re-deriving structural equations or re-implementing
intervention logic for each new model.

**Why this is feasible.** cSHAP is built from interventional queries of the form $f(\text{do}(X_S
= x_S))$ averaged over some distribution of out-of-coalition variables. PCI's
`ThinSearchSampler` already knows how to:
- intervene on a designated set of variables (suspects under PCI naming = "in-coalition" under
  cSHAP naming),
- sample alternative values for the remaining variables from a chosen distribution,
- evaluate the model's downstream output.

The cSHAP value of feature $j$ is then a weighted sum over coalitions $S$ of differences of these
interventional expectations. The Shapley combinatorics is just bookkeeping on top.

**Plan (skeleton only — implementation details defer to actual coding pass).**

1. Identify the cSHAP variant to target. Candidates:
   - **Heskes et al. — Causal Shapley.** Uses a causal chain ordering; out-of-coalition variables
     are intervened to a baseline / sampled from a conditional.
   - **Janzing / Lundberg interventional SHAP.** Out-of-coalition variables sampled from a
     marginal/observational distribution under an intervention.
   - **Sharma et al. — CF-Shapley.** Uses a single reference point $x^{\text{ref}}$ rather than a
     distribution.
   **OPEN: which variant first?** Recommendation: **Heskes-style causal Shapley** as the most
   commonly cited and the one S1iS explicitly named. CF-Shapley as a second comparison if room.

2. Define a thin module — tentatively `pci/explanation/causal_shap.py` — exposing:
   ```
   causal_shap_values(searchable_model, factual, features, baseline_sampler, ...)
       -> dict[feature -> phi]
   ```
   Internally, for each coalition $S \subseteq \text{features}$, build an interventional regime
   that forces $X_S$ to factual and lets $X_{\bar S}$ be sampled by `baseline_sampler`. Use the
   existing intervention machinery (chirho-backed) to evaluate $\mathbb{E}[E \mid \text{do}(X_S =
   x_S),\, X_{\bar S} \sim \text{baseline}]$. Then combine the coalition values with the Shapley
   weights.

3. The `baseline_sampler` is a parameter — observational, marginal, or a single reference. This
   directly tracks the cSHAP-variant choice from item 1.

4. Reuse points (no re-implementation needed):
   - `SearchableModel` for the structural-equation wrapper.
   - The intervention infrastructure in `ThinSearchSampler` / `condition_on_interventional_regime`
     for setting `do(X_S = x_S)`.
   - The same model object that PCI uses → automatic consistency between PCI and cSHAP attributions.

5. Validation strategy: on the synthetic Design-C model, the closed-form analytical cSHAP can be
   computed and compared against the implementation, as a unit test. (Once written, the same
   module can be used on AVM or other examples without re-deriving anything.)

**Cost estimate.** First-pass implementation: probably ~half a day if `ThinSearchSampler` already
exposes the right intervention API. We should look at the API before committing — flagging as a
prerequisite read.

**Status.** **In scope; lives between SHAP integration and §5 narrative.** Removed from "deferred"
in §3 step 6; see updated §3 below.

---

## 5. Order of execution

**Scope (2026-05-05 user direction).** Address the **A-items first** (rebuttal-driven fixes
F1–F6 / G1–G7). The B-items (F7–F15, internal clarity / wording fixes) and the C-items / I-items
(improvements) are explicitly **deferred** to a second pass after the As land. The §5 below
reflects only A-scope steps.

A-scope steps, in order:

1. **(G7) Implement Design C model in the notebook.** Replace `synthetic_model` with the
   six-variable $\{L_1, L_2, O_1, O_2, P_1, P_2\}$ structure from §4.1. Render the new DAG. Generate
   factual data for 500 events. Calibrate $\tau$ empirically.
2. **(F4 / G3) Justify suspect restriction.** Add the prose argument from §4.3 to the notebook
   (and flag the placeholder forward-reference for the formal theorem in another section).
3. **(F2, F3 / G2, G4) Necessity and sufficiency reported separately.** Refactor the score-display
   code so $ci_N$ and $ci_S$ are surfaced per variable rather than collapsed into `total`. Update
   case-study histograms to show both components.
4. **(case-study cells)** Find the three indices using §4.1 selection criteria; produce N-and-S
   histograms for each, walking through the archetype expectations table.
5. **(500-event batch)** Replace the existing single-axis bar plot with one that splits N and S.
   Implement the six new qualitative desiderata from §4.1 — including the two sufficiency- and
   necessity-specific contrasts (items 3 and 4 of that list).
6. **(F1 / G1) Add SHAP comparison.** Per §3 steps 0–5: add dep, wrap model, ExactExplainer, mirror
   figure for case studies, parallel summary for batch, parallel desiderata-frequency check.
7. **(G1, cSHAP) Causal-SHAP module.** Per §4.4: implement `pci/explanation/causal_shap.py`
   reusing `SearchableModel` / intervention infrastructure; validate on the closed-form Design-C
   analytical cSHAP; add as a third bar/series on each comparison figure.
8. **(F6 / G6) Cite Lundberg et al. (2017)** in the notebook prose where SHAP is introduced; add
   the Shapley-equivalence sentence. Check `sec4_examples.tex` and add the same credit there if
   Shapley weighting is referenced.
9. **(F5 / G5)** Restructure `sec5_synthetic.tex` to mirror the notebook tutorial: model →
   archetype expectations → case-study walkthrough → batch sweep → SHAP comparison → cSHAP
   comparison. Add the necessity/sufficiency separation, the suspect-restriction sentence, the
   Lundberg credit, and the new figures.

**B-scope and C-scope** (deferred): all of F7–F15, all of I1–I4, the table version of the
desiderata frequencies, the $\epsilon$ sensitivity sweep, etc. Will revisit once steps 1–9 land.

---

## 6. Open questions / parking lot

**Resolved 2026-05-05:**

- **D4.** Variable naming: **descriptive letters $L, O, P$**.
- **Q1.** Theorem: **placeholder pointer for now, formal statement to be added later** in another
  section. §5 just cites the placeholder and states the consequence.
- **Q2.** cSHAP variant first: **Heskes-style causal Shapley**.
- **Q3.** `baseline_sampler` follows Heskes: out-of-coalition variables sampled from the
  conditional distribution implied by the causal chain ordering, in/out determined by the
  coalition. (Specifics per Heskes et al. — to be matched to the paper when implementing.)
- **Q5.** Keep the combined `total` score too — figures show $ci_N$, $ci_S$, **and** `total`.
- **Q6.** Decisions:
  - Add `shap` dep — yes.
  - **Do not reimplement structural equations in numpy.** Wrap the canonical Pyro model and
    expose a function that conditions the six root sites on a given feature vector, runs the
    model forward, and reads out $E$. This is the same pattern that the cSHAP module will use
    (§4.4), so SHAP, cSHAP, and PCI all consume the same model definition.
  - `ExactExplainer` — confirmed (six features, $2^6=64$ coalitions, exact and fast).
  - Separate mirror figure for case studies — confirmed.

**Open at design stage:**

- **Q4.** Six desiderata in §4.1: list left as a proposal. Will revisit when first figures exist
  and we can see what the data actually shows. (User: "I don't know which desiderata.")

**Open but can wait until first draft of figures exists:**

- Whether the figures separating $ci_N$, $ci_S$, total are visually clear or we need a 2D scatter
  (D3 reconsideration).
- Whether `ci(V)` definition is fully self-contained in §5 or back-references §4.

**Deferred (B-scope and C-scope):**

- All F7–F15 wording / clarity fixes.
- All I1–I4 improvements.
- $\epsilon$ sensitivity (I2): depends on whether `ThinSearch` exposes $\epsilon$.
- cSHAP variants beyond Heskes.

---

## 7. Pre-implementation: walkthrough notebook on one observation

**Purpose.** Before touching the published `synthetic_explanation.ipynb` or starting on the §5
order of execution, build a separate scratch notebook that:

1. Implements Design C as a Pyro model.
2. Generates a single factual observation in a regime that exercises every archetype
   (preempted regime, contestable max — Case 1 from §4.1).
3. Walks through `ThinSearchSampler` step by step on that observation.
4. Computes $ci_N(V)$, $ci_S(V)$, and a combined `total` for each of the six suspects.
5. Compares the numbers against the §4.1 archetype-expectation table cell by cell.
6. Records any infrastructure gaps, surprises, or mismatches.

This is the feasibility pass that de-risks the broader plan. If the tool produces something
materially different from the expectation table, the redesign needs to be revisited before any
production-figure work happens.

**Open: notebook path and name.** Two options:
- `docs/source/synthetic_design_c_walkthrough.ipynb` — sits alongside the published notebook;
  visible if it accidentally gets built into the docs.
- `claude_notes/synthetic_design_c_walkthrough.ipynb` — clearly scratch / exploratory; not picked
  up by sphinx.
**Recommendation: `claude_notes/`** during the walkthrough phase; promote / replace once the
results validate the design.

**Proposed cell structure** (no code yet — outline for approval):

1. Markdown — purpose, link back to `notes/sec5_design.md`.
2. Code — imports + smoke-test pattern (mirror `synthetic_explanation.ipynb`).
3. Code — Design C Pyro model (`synthetic_model_c`); render DAG.
4. Code — generate $n$ factuals; pick one observation in Case 1 regime
   ($|L_2| > \tau$, $|O_1 - O_2| < \delta$); display the factual values; verify $E$ by hand.
5. Markdown — archetype expectations table for this specific observation, copied from §4.1 with
   regime-specific row pruning.
6. Code — `SearchableModel` with root suspects $\{L_1, L_2, O_1, O_2, P_1, P_2\}$, run
   `ThinSearchSampler`, inspect what comes back.
7. Code — for each suspect, compute $ci_N$ and $ci_S$ separately
   (via `condition_on_interventional_regime` + the underlying suff/nec outcomes, *without*
   collapsing through `abs_diff_score`). Tabulate values.
8. Code — also compute the combined `total` for comparison.
9. Markdown — per-variable readout: predicted archetype vs measured ($ci_N$, $ci_S$); annotate
   matches / mismatches.
10. Markdown — observations / next steps; whether the design needs revisiting.

**What we are NOT doing in this notebook:**

- No 500-event batch.
- No SHAP / cSHAP comparison.
- No prose intended for the paper.
- No figures intended for the paper.

It is purely a sanity-check of (model + tool + expectations) on one observation.

**Decisions needed before creating it:**

- **D5.** Notebook path — `docs/source/` or `claude_notes/`? **Recommendation: `claude_notes/`.**
- **D6.** Calibration of $\tau$ and $\delta$ — pick concrete starting values, accept that the
  observation-search may need to retry with different seeds. **Proposal: $\tau = 0.674$,
  $\delta = 1.0$, seed = 0; refit if no observation matches in $n=500$.**

---

## 8. Decision log

- 2026-05-05: **D1 = Design C** (full three-archetype model). Reason: clearest answer to S1iS-A2,
  strongest visual contrast with SHAP, makes sufficiency validation possible.
- 2026-05-05: **Evaluation patterns kept similar to current** (case-study histograms + 500-event
  batch + qualitative-desiderata barplot). Implies D2 = three case studies, D3 = no 2D scatter.
- 2026-05-05: **G4 — report necessity and sufficiency separately**, not a combined `total`.
  Reason: directly addresses S1iS's "sufficiency never validated" complaint.
- 2026-05-05: **G3 — restrict suspects to root variables**, justified by the deterministic-mediator
  argument; gesture at a formal theorem to live in another section.
- 2026-05-05: **G6 — cite Lundberg et al. (2017)** and acknowledge the $J=1, K=|X|$ Shapley
  equivalence. Apply both in §5 and (if relevant) in §4 / Example 1.
- 2026-05-05: **cSHAP brought into scope**, implemented via the existing `SearchableModel` /
  intervention infrastructure rather than from scratch per example.
- 2026-05-05: **A-items first.** F7–F15 (B-items) and I1–I4 (C-items) deferred to a second pass.
- 2026-05-05: **D4 = descriptive letters $L, O, P$** (Sally/Billy story stays as motivating prose).
- 2026-05-05: **Q1 = placeholder cross-reference**; theorem to be stated formally elsewhere later.
- 2026-05-05: **Q2 = Heskes-style causal Shapley** as first cSHAP variant.
- 2026-05-05: **Q3 = baseline sampler follows Heskes** (conditional under causal ordering).
- 2026-05-05: **Q5 = keep `total` alongside $ci_N$ and $ci_S$**.
- 2026-05-05: **Q6 — model wrapping for SHAP via Pyro `condition`** on the canonical model rather
  than numpy reimplementation. SHAP, cSHAP, PCI all consume the same model definition. Reuses the
  intervention infrastructure we'll build for §4.4.
- 2026-05-05: **Pre-implementation step inserted: walkthrough notebook on one observation** (see
  §8 below) before any of the §5 steps are executed. User instruction:
  > "let's design one case (create a new notebook) and work through it in detail with thin sampler
  > before you even do anything else"
