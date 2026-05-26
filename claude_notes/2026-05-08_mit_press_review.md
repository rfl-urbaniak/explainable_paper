# MIT Press Reviewer Report (simulated) — 2026-05-08

Reviewer simulation of *A Computationally Feasible Framework for Causal
Probabilistic Explanation*, performed against the current state of `main.tex`
and `sections/`. All file:line references are clickable from the IDE.

## Recommendation

**Major revisions.** The framework is novel, the worked examples are unusually
careful, and the formal correspondences with Halpern's actual causality are a
real contribution. But the manuscript is in a transitional state: the build is
broken in several places, two theorem statements have proof gaps, and one of
the headline desiderata (D-MXY) is satisfied only in a footnoted "illustrative"
sense rather than within a single game. None of these are fatal, but they need
to be fixed before publication.

---

## A. Build / structural issues (must fix before resubmission)

1. **Broken `\input` chain.** [main.tex:190](../main.tex#L190) imports
   `sections/sec6_actual_causality`, which does not exist. The actual file is
   [sections/sec7_actual_causality.tex](../sections/sec7_actual_causality.tex).
   The paper as posted will not compile against the current `sections/` tree.
2. **Section [sec6_dce.tex](../sections/sec6_dce.tex) not included.** It defines
   `\label{sec:DCE}` but is unreferenced in `main.tex`.
   [sec9_discussion.tex:52](../sections/sec9_discussion.tex#L52) cites
   `\ref{sec:dce}` (lowercase) — both the inclusion and the case mismatch must
   be fixed.
3. **Dangling theorem references.** [main.tex:21-22](../main.tex#L21-L22)
   declares `\newtheorem*` shells for `th:ac->positive` and `th:ci_halpern`;
   the actual labels in the AC section are `th:ac-exp` and `th:exp-ac`. These
   shells are also unused.
4. **Other dangling refs** flagged by the build log: `sec:TODO`
   ([sec3_definitions.tex:664](../sections/sec3_definitions.tex#L664)),
   `sec:dce` ([sec9_discussion.tex:52](../sections/sec9_discussion.tex#L52)),
   `sec:definitions` ([sec6_dce.tex:75](../sections/sec6_dce.tex#L75) — should
   be `sec:causal_impact`).
5. **Missing bibliography entries** for `titsias2009variational` and
   `hensman2013gaussian` ([sec8_avm.tex:36](../sections/sec8_avm.tex#L36)).
6. **Section numbering ambiguity.** Two files start with `sec7_*`
   (`sec7_actual_causality.tex` and `sec7_ac_benchmark.tex`) plus
   `sec7b_sir_benchmark.tex`; the AC theory section is logically Section 6 in
   the running order. Renumber the files to match the table of contents.
7. **Placeholder figures in
   [sec8_avm.tex:42-53](../sections/sec8_avm.tex#L42-L53),
   [87-100](../sections/sec8_avm.tex#L87-L100)** (anonymised DAG and
   attribution comparison are `\fbox` placeholders). MIT Press camera-ready
   cannot ship with these.

---

## B. Math / proof checks

I worked through every closed-form computation in §2–§4. The arithmetic is
correct everywhere I checked:

- Alice's PN=0.045, PS=1, PNS=0.045
  ([sec2:178](../sections/sec2_motivations.tex#L178)). ✓
- Bob's PS for credit = 0.955 (= 0.9·0.95+0.1·1). ✓
- Population-level Bob-gender PN=0.02 (I get 0.017 — rounds to 0.02). ✓
- The full Alice-gender PCI calculation through the three noise regions,
  yielding 0.263 ([sec3:587](../sections/sec3_definitions.tex#L587)). ✓
- Plain-SHAP and Causal-SHAP coalition values for the 2-feature OBCB game and
  the signal-with-mediation example (all of
  [sec4:496-582](../sections/sec4_examples.tex#L496-L582)). ✓
- The four-way (C,T) decomposition giving PCI(X→Y|W={M}) =
  ¼(0.320+0+0.425) = 0.186
  ([sec4:609-623](../sections/sec4_examples.tex#L609-L623)). ✓

Three substantive concerns:

8. **Theorem 9 (`th:local_max_to_ac`) — proof gap.**
   [sec7_actual_causality.tex:209-251](../sections/sec7_actual_causality.tex#L209-L251).
   The contradiction step compares Φ(C†,T†) with Φ(C♯,T♯) using only the
   Γ-weights, but Φ is the product of three factors: p^Γ, the sufficiency
   indicator (=1 for both), and the **necessity integral**
   ∫𝟙[·]Δ_C(dc′). The proof of Theorem 7 only establishes that the necessity
   integrals are *strictly positive*, not that they are equal or bounded
   below. A strictly cardinality-decreasing Γ_s can be overwhelmed by a much
   larger necessity integral at C† (e.g., when many c′ flip φ at the larger
   set but few do so at the subset). Either add a hypothesis that the
   necessity integrals are comparable (e.g., monotone in cardinality), or
   argue why Δ_{C♯} assigns at least as much ¬φ-mass as Δ_{C†} does on the
   projected set. As stated, the contradiction does not go through.

9. **Theorem 7 — continuity argument is sketched.**
   [sec7_actual_causality.tex:137-140](../sections/sec7_actual_causality.tex#L137-L140)
   writes "in the continuous case, on a neighbourhood by continuity of the
   structural equations under fixed u." For the indicator
   𝟙[(M,u)⊨[C←c′,…]¬φ] to be 1 on a neighbourhood of c′ you need (i) the
   structural equations downstream of C to be continuous in c′ (with u
   fixed), and (ii) ¬φ to be an open event in dom(Y), or at least to have
   measure-zero boundary under the pushforward. State (i) and (ii) explicitly
   as hypotheses. Threshold events like the SIR overshoot>24 in
   [sec7b_sir_benchmark.tex:30](../sections/sec7b_sir_benchmark.tex#L30)
   trivially satisfy this, but binary-output models with discontinuous
   structural equations may not.

10. **Theorem 8 (`th:exp-ac`) — implicit factivity.**
    [sec7_actual_causality.tex:152-171](../sections/sec7_actual_causality.tex#L152-L171).
    The conclusion is "context-sensitive necessity", but Φ>0 also forces the
    sufficiency indicator =1, which under a fixed u implies (M,u)⊨φ
    (factivity). State this explicitly — Proposition 11 already needs it and
    currently re-derives it informally.

11. **Sub-probability subtlety.** Definition 16 (Joint Necessity-Sufficiency
    Measure,
    [sec3:393-439](../sections/sec3_definitions.tex#L393-L439)) defines
    P_k^{s,n}(A×B) on rectangles and notes total mass Pr_Γ[X_k∈C]≤1. Two
    clarifications would strengthen the definition: (a) state that the
    rectangle-by-rectangle definition extends uniquely to a sub-probability
    measure on the product σ-algebra (Carathéodory); (b) clarify whether
    E[ci(Y^s,Y^n,y★)] in Definition 17 is taken against this sub-probability
    or its renormalisation. The downstream calculations integrate against the
    sub-probability, so just say so.

12. **D-MXY only met under split witness sets.**
    [sec4:691-695](../sections/sec4_examples.tex#L691-L695) acknowledges the
    comparison uses W={M} for D-XY and W={X} for D-MY — "illustrative rather
    than a single-game ranking". This is the most striking *quantitative*
    claim against SHAP and it is currently propped up by switching games
    between rows. Either (a) report a W setting under which both cells use
    the same witness configuration and D-MXY still holds, or (b) demote
    D-MXY to a desideratum about *configurations* rather than methods, with
    appropriate hedging in the abstract / introduction.

13. **Synthetic benchmark tolerance.**
    [sec5_synthetic.tex:142-149](../sections/sec5_synthetic.tex#L142-L149)
    accepts "noise floor stable across regimes" with ε_baseline=2.0 against a
    measured drift of 1.088. Since the desideratum is precisely *cross-regime
    stability of the noise floor*, half the tolerance budget is being
    consumed by the very effect being tested. Tighten the tolerance and
    report the gap honestly, or rephrase the desideratum as "drift is
    dominated by signal" with a comparative bound (e.g.,
    |ci_N(D,1)−ci_N(D,2)| < c·min_i Δ_N(L_i)).

---

## C. Conceptual / framing comments

14. **PN definition mixes potential outcomes and conditioning.** Definition 5
    ([sec2:148-150](../sections/sec2_motivations.tex#L148-L150)) writes
    P(Y_{c′}≠y★ | C=c★, Y=y★). Readers from a non-Pearl background will not
    immediately see that Y_{c′} is a random variable on the same probability
    space as the conditioning event. A one-sentence twin-network or
    potential-outcome footnote would resolve this.

15. **"Context-sensitive necessity" terminology drift.** §2 introduces the
    witness mechanism and §3 the witness set T, but §3 stops calling the
    necessity component "context-sensitive." The label is useful and would
    help readers track which condition is doing what.

16. **The relation to descriptive normality is mentioned but not exercised.**
    [sec1:44-52](../sections/sec1_intro.tex#L44-L52) and
    [sec3:307-318](../sections/sec3_definitions.tex#L307-L318) flag the link
    to Halpern (2015) and Icard et al. (2017), but no example actually uses a
    non-trivial normality ordering. Either work through one such example or
    downgrade the claim to "shares an intuition with."

17. **AC benchmark structural caveats.** The "Scope of the conclusion"
    paragraph at
    [sec7_ac_benchmark.tex:144-164](../sections/sec7_ac_benchmark.tex#L144-L164)
    is excellent and rare in this literature. Consider lifting one sentence
    into the abstract and the introduction, since the headline scaling
    result is currently presented without these caveats in §1.

18. **Discussion is too brief on negative-result territory.**
    [sec9_discussion.tex](../sections/sec9_discussion.tex) is two pages, half
    of which restate the framework. There is no clear statement of (a) when
    PCI does *not* match AC verdicts (Observation 12!), (b) the cardinality
    knob's interaction with sample efficiency, (c) the cost of the
    PSCM-availability assumption relative to PN/PS bounds. All three are
    gestured at but deserve their own paragraphs.

19. **AVM section's role.**
    [sec8_avm.tex:1-10](../sections/sec8_avm.tex#L1-L10) appropriately scopes
    the section as a feasibility witness, but the placeholder figures
    undermine even that minimal claim. Either ship the anonymised figures or
    remove the section and merge a one-paragraph "feasibility at production
    scale" pointer into §9.

---

## D. Editorial / writing suggestions

20. **Typos.** "isssue" → "issue"
    ([sec3:285](../sections/sec3_definitions.tex#L285)). "overriden" →
    "overridden" ([sec2:297](../sections/sec2_motivations.tex#L297)).

21. **Comma-series inconsistency** in the abstract: "the variable selection
    distribution, alternative value distribution, and causal impact function"
    appears twice with slightly different forms; pick one.

22. **Long footnote in
    [sec3:355-365](../sections/sec3_definitions.tex#L355-L365)** spans nearly
    half a page about ε-excised distributions, ROPE, and Cohen's effect
    sizes. This is good content; promote to its own short paragraph in the
    body.

23. **Notation churn in §4.**
    [sec4:37-42](../sections/sec4_examples.tex#L37-L42) overloads **S** as
    both PCI's suspect set and SHAP's coalition by typeface alone (bold vs.
    plain). On a printed page the distinction is fragile. Use 𝒮 or
    **C**_shap for the SHAP coalition.

24. **Definition 16's split across lines
    [sec3:410-426](../sections/sec3_definitions.tex#L410-L426)** is hard to
    parse. Consider stating P_k^s and P_k^n as named definitions, then
    defining P_k^{s,n} in a third step.

25. **Tables in §3 (`tab:pn_ps_pns`, `tab:pci_obcb`, `tab:desiderata`,
    `tab:pci_decomp`)** could be consolidated into one master table with
    method as a column. Four tables in four pages on the same eight
    Alice/Bob cells is a lot.

26. **Captions over-explain.** Several captions
    ([sec4:213-216](../sections/sec4_examples.tex#L213-L216),
    [sec4:677-695](../sections/sec4_examples.tex#L677-L695)) repeat material
    from the body. Captions should be self-contained but not redundant.

27. **§5 sidewaystable** breaks the reading flow; a regular table with the
    rows split into two halves would scan better. Alternatively, move
    detailed inequalities to an appendix and keep a compact ✓/✗ summary in
    the body.

28. **§6 (DCE)** opens without recapping the running example. A reader
    arriving from §5 needs a one-sentence handoff explaining why we are now
    in a credit-limit world.

29. **Author block is incomplete.** [main.tex:167](../main.tex#L167)
    `% TODO: add Andy, Sam, Drew (full names, emails, affiliations).`
    Resolve before resubmission.

30. **The abstract is one long paragraph of 13 sentences**
    ([main.tex:174](../main.tex#L174)). Split into a problem statement, the
    contribution, and the empirical witnesses; current density obscures the
    claims.

31. **`\Ourapproach` / `\ourapproach` macro**
    ([main.tex:134-135](../main.tex#L134-L135)) is used inconsistently with
    respect to surrounding spacing. The version with the trailing space is
    fine mid-sentence but produces double spaces at the start of one. A
    single macro with `\xspace` would be cleaner.

---

## Summary

The conceptual contribution is genuine: a Monte Carlo–friendly probabilistic
generalisation of actual causality with explicit user-facing knobs (Γ, Δ, ci),
supported by both a formal correspondence with AC verdicts and a scalability
benchmark. The OBCB and signal-with-mediation worked examples are some of the
most careful in the SHAP-comparison literature I have seen. The two proof
gaps in §6 (Theorem 9 and the continuity argument in Theorem 7) are the only
mathematical issues I would block on; the rest are surface — but there is
enough surface to need a careful pass before camera-ready.

---

## Suggested triage order for revisions

1. Fix build (A1–A5) — required before any other reviewer can read it cleanly.
2. Patch proof gaps (B8, B9), state implicit factivity (B10).
3. Decide on D-MXY framing (B12).
4. Either ship AVM figures or remove the section (A7, C19).
5. Tighten §5 desideratum #13 tolerance (B13).
6. Editorial sweep (D20–D31).
7. Consider abstract rewrite + author block (A6, D29, D30).
