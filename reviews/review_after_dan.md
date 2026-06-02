# Review after Dan — math correctness, notational uniformity, content flow

Date: 2026-06-01. Scope: full `main.tex` and all `sections/*.tex` (incl. appendices A–C).
Method: every numerical computation in the worked examples was re-derived by hand;
every theorem proof in §7 was read line by line; all `\ref`/`\label` targets were
checked. This review is read-only — nothing was edited.

**Bottom line.** The mathematics is in good shape: every numeric table I checked is
internally consistent and re-derivable, and the §7 actual-causality correspondence
proofs are sound. The main weaknesses are (a) **notation that drifts section to
section** — the outcome variable alone is written five different ways — and (b) a
handful of **concrete cross-reference / typo bugs** that will show up in the compiled
PDF. None of the math is *wrong*; the cost is to the reader, who has to re-key the
same objects under new symbols in nearly every section.

---

## Part 1 — Mathematical correctness

### What checks out (verified by hand)

- **§3 OBCB worked example.** Alice-gender PCI = 0.263 reproduced exactly
  (u-region split 0.2/0.7/0.1; necessity 0.192/0.508/0; sufficiency 2/3). Necessity
  marginal 0.394 and sufficiency marginal 0.667 both reproduce, and Total = 0.263.
  All four rows of Table `tab:pci_decomp` and all seven rows of `tab:desiderata` are
  internally consistent.
- **Appendix A (SHAP for OBCB Option A).** All four `v(S)` values for Alice and Bob,
  and all eight Shapley magnitudes (0.107 / 0.174 / 0.107 / 0.343), reproduce exactly,
  including the `g = 1 − f` sign flip. `v(∅) = 0.28125` checks.
- **Appendix B (signal-with-mediation).** All plain-SHAP and Causal-SHAP coalition
  values and Shapley values reproduce (e.g. `v({Y})=0.889`, `φ_X=0.306`, `φ_Y=0.194`
  for target M; Causal `φ_M=0.357`, `φ_Y=0`). The PCI per-pair table
  (0.320 / 0 / 0.425 → 0.186) reproduces. Covariances and the three conditional
  expectations are correct.
- **§4 signal margins.** D-MXY margins 0.034 (W=∅) and 0.133 (W=3rd), and the
  baseline-instance margins 0.035 / 0.097, all reproduce.
- **§7 proofs (Theorems `th:ac-exp`, `th:exp-ac`, `th:local_max_to_ac`,
  Corollary `cor:marginalize`, Proposition `prop:filter`, Observation
  `obs:misalignment`).** The three-factor positivity argument, the
  cardinality-decreasing minimality argument, and the filter-recovers-AC argument are
  all valid. The footnote on the closure condition for `S` (excluding deterministic
  descendants from the suspect set) correctly discharges the joint-support concern.
  The binary-PNS alignment claim ("exact when S={X_k}, W=∅, dom={0,1}") is genuinely
  exact — I verified it reduces to Pearl's `P(Y_x=1, Y_{x'}=0)`.
- **§7b desert-traveller tables.** Internally consistent. In particular the
  `S=1 ⟹ J=N` identity holds wherever sufficiency is saturated (e_A poisoner: N=J=1/3;
  e_B shooter: N=J=1/2; weak e_A poisoner: N=J≈0.208), which is a good cross-check.

### Genuine concerns (math/definition level)

1. **§7b explanatory arithmetic contradicts the stated weighting.** §7b point (2)
   (lines 161–163) explains the 1/3 cap as "1/4 × 1 contribution … average over the
   **four subsets** gives 1/3," i.e. uniform-1/4 weighting. But the spec (and
   Appendix C, `app:desert_components` line 33) says the witness selection is
   **cardinality-uniform** — weights 1/3, 1/6, 1/6, 1/3 over {∅,{c},{d},{c,d}}, not
   1/4 each. The table value 1/3 comes from the notebook and is presumably right; it
   is the *prose rationalization* that is loose/inconsistent. Either restate the
   weighting or drop the "1/4 × 1" gloss.

2. **§7b `N_C/S_C/J_C` use a different normalization than §3's `E[ci]`.** Equation
   `eq:nsj` defines `N_C = E[1{Y^n≠y*}]` etc. as plain expectations bounded by 1 (the
   tables report `S=1`). But §3's `E[ci]` (Def. `def:causalimpact`) is integrated
   against the **sub-probability** `P_k^{s,n}`, whose mass is `Pr_Γ[X_k∈C] ≤ 1`, so it
   is *not* normalized per-cause. The §7b quantities are implicitly conditioned on the
   candidate cause being the fixed singleton under test. This is fine but should be
   stated — as written, a reader who carries the §3 normalization into §7b will be
   confused why `S` can equal 1.

3. **"Total = product of the two marginals" is presentation-specific, not general.**
   In `tab:pci_decomp` the bold Total equals (necessity marginal) × (sufficiency
   marginal) in all four rows. That holds *only because* `P_k^s(·|u)` is constant in
   `u` here, so it factors out of `∫ P^s P^n dP_U`. In general the joint does **not**
   factor into the product of marginals (the whole point of the conditional-on-`u`
   construction). One sentence in the caption noting this would prevent a reader from
   over-generalizing the relationship.

4. **Minor measure-theoretic gap in Def. `def:jointnecsuf`.** `Δ` is required to
   exclude the factual point (`s* ∉ supp Δ`), but the per-`C` pushforward
   `Δ_C(s*)` (restriction map) can place mass on `s*|_C` even when the full `s*` is
   excluded. The necessity integral is still well-defined, so this is not an error,
   but the "genuine counterfactual departure" guarantee is on the *joint*, not on each
   restricted marginal. Worth a footnote if a referee is picky.

---

## Part 2 — Notational uniformity (the biggest cost to the reader)

5. **The outcome variable is written five different ways.** This is the single most
   pervasive issue.
   - §3 / §6 / §7: `Y`, factual `y*`, worlds `Y^s`, `Y^n`. (the intended convention)
   - §4 signal: outcome target is generic `B`, worlds `B^n`, `B^s`, `B*`, impact `c̄`
     (instead of `Y`, `y^n`, `y^s`, `ci`).
   - §5: outcome named `E` structurally, then switches to `Y`/`y*` in the `ci`
     definitions within the same section.
   - §8: outcome named `C` with factual `c*` — which **collides head-on with the
     core symbol `C` = active suspect set**; disambiguation rests entirely on
     bold/non-bold (`C` vs `\mathbf{C}`), e.g. line 91 `|y^n_j − c*|` sits right next
     to `\mathbf{C}_j`.
   - §9: worlds written `y_{nec}`, `y_{suff}` instead of `y^n`, `y^s`.
   - §10: outcome is lowercase `y`.
   **Recommendation:** standardize on `Y`/`y*`/`y^n`/`y^s` everywhere; where a section
   genuinely needs a local name (E, B), state the alias once at the top of the section.

6. **`Γ` is reused for a causal DAG in §10 (line 42–43).** `Γ` is a *defined core
   symbol* (the variable-selection distribution, with marginals `Γ_s`, `Γ_w`). Using
   it for "a causal DAG `Γ`" is a direct collision. Rename the DAG (`\mathcal{G}`).

7. **`Γ*` and `Δ*` denote argmax sets in §7** (`th:local_max_to_ac`,
   `cor:marginalize`). Starring the two core distribution symbols to mean "set of
   optimizers" is confusing; consider `\mathcal{O}_Γ`, `\mathcal{O}_Δ` or similar.

8. **Prime overload in §7b.** Alternative values use `c'` throughout the paper; the
   weak-poison structural equations (line 222) use `u'`, `x'`, `p'` for **Boolean
   complement**. Two unrelated meanings of the prime in the same subsection.

9. **`F` overloaded in §2.** Used both as Boolean "False" (`loan = F`) and as a
   conditioning-variable placeholder (`PNS_c(… | F=f*)`, lines ~315–318). Rename the
   conditioning placeholder.

10. **Generic `A`/`B` (cause/target) and `c̄` are introduced only in Appendix B**
    (lines 126–128) but are *used in the §4 main-text table caption*
    (`tab:signal_desid`, "neither `A` nor `B`", `c̄ = E[|B^n−B*|]−…`). The reader hits
    the symbols before the definition. Define `A`/`B`/`c̄` in §4 §`sec:signal` body.

11. **`\mathbf{t}^{\star\prime}` in the §7 `th:exp-ac` proof** (line 206) is
    non-standard — elsewhere the held-fixed witness value is `\mathbf{w}^\star|_{T'}`.
    Make it uniform.

12. **`check-failed` vs `check\text{-}failed`** are rendered two ways for the same
    variable (e.g. §2 lines 102/357 vs 170/312). Pick one macro.

13. **Appendix A line 65** mixes `\mathcal{S}` and plain `S` in one equation
    (`\sum_{x_{N\setminus S}}` with the coalition written `\mathcal{S}` two lines up).
    Should be `N\setminus\mathcal{S}`.

14. **Terminology drift "responsibility" / "attribution" / "impact" / "score."** Used
    interchangeably; §9 and §7b lean on "responsibility," §10 on "attribution mass,"
    §3 on "impact." Harmless but a quick pass to settle on one primary term per role
    would tighten the paper.

---

## Part 3 — Concrete bugs (will appear in the compiled PDF)

15. **`appC_desert.tex:46`** contains a literal authoring annotation:
    `Section~\ref{sec:causal_impact} (sec3:330--336)`. The `(sec3:330--336)` is a
    leftover line-number note and will print verbatim. Remove it.

16. **§7b line 38:** "discuss other differences between PCI and `\ourapproach`" —
    `\ourapproach` renders to **PCI**, so this reads "differences between PCI and PCI."
    Should be "between PCI and Pearl's probability of causation."

17. **§4 lines 67 and 763** attribute the seven OBCB desiderata to
    "Section~\ref{sec:motivations}" (§2). They are defined and labelled in
    **§3 (`sec:causal_impact`, `\pageref{desid}`)** — §2 line 415 already (correctly)
    points to §3 "that section." Repoint §4's two refs to `sec:causal_impact`.

18. **§10 repeatedly cites a "SHAP-vs-PCI divergence in Section~\ref{sub:synthetic}"**
    (§5) — lines 11, 21, 34, 98. But §5 contains **no SHAP comparison with results**:
    it mentions SHAP only in passing (the actual SHAP-vs-PCI numerical comparison is in
    **§4, `sec:shap_examples`**). Either §5's promised "comparison block below"
    (its line 80) is missing its SHAP numbers, or §10 should cite §4. Resolve one way.

19. **§8 algorithm line-number citations are all shifted** (~9 of them: lines 105,
    106–107, 108, 111–113, 116, 120, 121–123, 123–124, 130). The prose counts only
    `\State` lines while the `algorithmic` block auto-numbers `\If`/`\For`/`\Require`
    too. E.g. "active suspect set `C_j` (line 6)" — `C_j` is sampled on line 8;
    "`u_j` drawn on line 4" — it is drawn on line 6. Re-key all nine against the
    rendered numbering (or switch to `\State`-only numbering).

20. **§11 line 82: "lease decision"** — the running example is a **loan/credit**
    decision (OBCB). Almost certainly should read "loan decision."

21. **§6 line 27: "token-level responsibility."** "Token-level" is an NLP term and
    does not fit the credit-limit/continuous setting. Likely meant
    "individual-level" / "instance-level."

22. **§5 diagnostic D2 (lines ~221–224):** the desideratum is stated with absolute
    bars `|Δ_N(P) − Δ_N(L1)| < ε` but the reported figure is the signed `−0.267`. State
    D2 on the signed difference or report `0.267`.

23. **Typos in §7b:** "disscuss" (line 6); "Sso far" and "intuiton" (lines 200–201);
    the sentence at lines 334–338 ends mid-clause with a semicolon ("each posterior
    independently saturates at 0 or 1;") — the explanation of why (2) is captured but
    (1)/(3) are missed is left grammatically unfinished.

---

## Part 4 — Content flow

**Strengths.** The "Where we are" / "Where we go next" connective paragraphs are
genuinely good — the reader always knows the local position in the argument and what
the next section buys. Section ordering (motivations → definitions → SHAP/DCE
baselines → AC correspondence → benchmarks → deployment → discussion) is coherent. The
"verify the seven desiderata" closing of §3 is a strong anchor.

**Redundancy to trim.**
- **§2 re-motivates twice.** Lines 188–196 make the "witnesses don't extend to
  probabilistic models, so look at Pearl" setup two paragraphs in a row; and the
  "Path Forward" subsection (lines 356–363) re-derives the conditioning-vs-intervention
  failure already established at lines 314–321. This is the largest flow redundancy.
- **§2 Pearl preview (lines 383–407)** is a dense ~25-line technical excursion
  (Def. 10.3.5, `U_xy`, causal beam) at the end of a *motivations* section, then
  explicitly deferred to §7b. Trim to a sentence + forward pointer.
- **§10 states "this section is only qualitative / quantitative validation is in §5"
  four times** (lines 14–24 and 97–107). Once is enough.
- **§1** has a descriptive-normality digression (lines 57–68) that hedges against a
  connection the reader hasn't been set up to expect; consider moving to related-work.
  The "surfaces design choices as explicit knobs" selling point is repeated 3× in the
  intro.

**Possible reader-confusion spots.**
- **§9 polarity:** "mask-only is comparatively weak" while mask-only has the *highest*
  overshoot probability (0.87) — correct (high overshoot = weak policy) but easy to
  misread; add a clause. Also the single-world lockdown total (1.54) and the
  context-split values (0.81 fixed / 2.67 free) come from *different partitions* and
  needn't reconcile — say so, or a reader will try to average them.
- **§11 actionability passage (lines 75–87)** is the longest single block in the
  discussion and introduces a normative argument (gaming, gender suppression) not
  foreshadowed anywhere earlier. Either foreshadow in §1 or flag it as an explicitly
  normative aside.

---

## Suggested priority order for fixes

1. PDF-visible bugs: #15 (`sec3:330` leftover), #16 (PCI-vs-PCI), #20 ("lease"),
   #23 (typos/unfinished sentence), #19 (algorithm line numbers).
2. Wrong cross-refs: #17 (desiderata → §3), #18 (§10 → §4 for SHAP).
3. Notation unification: #5 (outcome symbol), #6 (`Γ` for DAG), then #7–#13.
4. Math clarifications: #1 (§7b weighting gloss), #2 (normalization note), #3 (joint ≠
   product in general).
5. Flow trims: §2 double-motivation and Pearl preview; §10 4× repetition.
