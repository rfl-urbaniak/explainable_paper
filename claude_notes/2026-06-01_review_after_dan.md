# Session 2026-06-01 — review_after_dan + fixes

## What happened
Did a full math-correctness + notation + flow review of the paper, wrote it to
`reviews/review_after_dan.md` (23 numbered items in 4 parts). Then applied fixes in
graded groups A → B → C with user sign-off on judgment calls. Paper compiles clean
(latexmk, exit 0, no undefined refs).

## Math verdict (verified by hand)
All worked numbers reproduce: §3 OBCB (Alice gender PCI=0.263, marginals 0.394/0.667),
appA SHAP (0.107/0.174/0.107/0.343), appB signal (all plain+causal SHAP + PCI 0.186),
§4 margins (0.034/0.133), §7b desert-traveller (S=1⟹J=N checks hold). §7 AC
correspondence proofs (Th `th:ac-exp`/`th:exp-ac`/`th:local_max_to_ac`, `prop:filter`,
`obs:misalignment`) are sound. No math is wrong — issues are notation drift + ref bugs.

## Applied — Group A (PDF-visible bugs)
appC removed leftover `(sec3:330--336)`; §7b "PCI and \ourapproach"→"PCI and Pearl's
probability of causation"; §11 "lease"→"loan"; §6 "token-level"→"individual-level";
§7b typos (disscuss/Sso/intuiton) + closed a dangling sentence; §8 all 9 algorithm
line-number cites re-keyed against the **actually-compiled** numbering (verified by
compiling a minimal snippet — `\Require`/`\Ensure` unnumbered, `\If`/`\For` numbered).

## Applied — Group B
B7: §4 desiderata refs `sec:motivations`→`sec:causal_impact` (the 7 desiderata live in
§3, not §2). B8 **resolution = fix-refs-only**: the §5 "comparison block below" SHAP
block **never existed** (git: the promise line was added in the archetype rewrite
`0189b0d`; pre-refactor §5 had zero SHAP). Removed the false promise from §5; repointed
all 4 §10 "SHAP-vs-PCI divergence" claims (lines 21,98,124 + caption) from `sub:synthetic`
(§5) to `sec:shap_examples` (§4). §5 = PCI-vs-ground-truth only; §4 = the SHAP contrast.

## Applied — Group C (notation), user chose "targeted" for the outcome symbol
- check-failed/loan-if-checked → `\text{-}` form; §10 DAG `Γ`→`\mathcal{G}`;
  §7 `t^{⋆′}`→`w^⋆|_{T'}` and argmax sets `Γ⋆`/`Δ⋆`→`\mathcal{O}^⋆`/`\mathcal{O}^Σ`;
  appA `N\S`→`N\mathcal{S}`; §2 conditioner `F`→`Z`.
- §8 outcome `C`/`c⋆` → `Y`/`y⋆` (kills clash with active-suspects `\mathbf{C}`; figures
  unaffected). §4 defined generic `A`/`B`/`B^n`/`B^s`/`B⋆`/`c̄` used in `tab:signal_desid`.
- §5 `E` and §9 `y_nec`/`y_suff`: **alias note added, symbols kept** (see discovery).

## KEY DISCOVERY — figure-coupled outcome symbols
§9's `y_nec`/`y_suff` and §5's `E` are NOT free to rename: the SIR figures
(`docs/source/sir_benchmark.ipynb`) render axis labels `$y_{\mathrm{nec}}$`/`$y_{\mathrm{suff}}$`
via set_xlabel/set_ylabel, and the §5 archetype DAG figure shows an `E` node. A text-only
rename to `y^n`/`y^s`/`Y` would desync text↔figure. So they got alias sentences instead.
**If full unification to `Y`/`y^n`/`y^s` is ever wanted: must also regenerate those
figures from the notebooks** (one-line set_xlabel edits + rerun).

## Applied — outstanding stragglers (#8, #22, #4)
#8: §7b weak-poison eqs + appC basic eqs now note "primes denote Boolean complement"
(vs alternative-value c'). #22: §5 diagnostic D2 restated on the signed gap (−0.267,
magnitude 0.267 > ε) instead of a signed value under |·| bars. #4: §3 `def:jointnecsuf`
footnote notes the restricted marginal Δ_C may put mass on s⋆|_C even though Δ excludes
the joint s⋆ (genuine-departure guarantee is on the joint).

## Applied — Group D (math clarifications, prose-only) — DONE
§7b `eq:nsj` footnote: N/S/J are per-cause (∈[0,1]) vs §3 sub-prob E[ci] (mass
Pr_Γ[X_k∈C]). §7b point (2): replaced the inconsistent "1/4×1 over four subsets" gloss
with the correct cardinality-uniform weights (1/3,1/6,1/6,1/3). §3 `tab:pci_decomp`
caption: added caveat that Total = marginal product only because P_k^s is constant in u.

## Applied — Group E (flow trims) — DONE (clear redundancy only)
§10: cut the 4th "qualitative-only/validation-in-§5" repetition. §2: merged the
double re-motivation (189–196); trimmed the 25-line Pearl preview (383–407) to a flag +
pointer to eq:pearl-paac in §7b; condensed the Path-Forward "conditioning is not a
workaround" recap. §9: added a polarity clause (higher overshoot prob = weaker policy)
and a note that the context-partition scores don't average to the single-world totals.

## Still open (author-discretion / deferred)
- §1 normality digression (57–64) and §11 actionability passage (75–87): NOT trimmed —
  these are intentional positioning/normative content, an authorial call, not redundancy.
- C14: terminology drift responsibility/attribution/impact (broad, low value).
- Optional: full Y/y^n/y^s unification would need regenerating SIR + archetype figures.

All edits compile clean (latexmk exit 0, 0 undefined refs).
