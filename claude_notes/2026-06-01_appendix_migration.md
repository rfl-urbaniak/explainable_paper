# Session note 2026-06-01 — Computations migrated to appendices

Migrated explicit running-example value-computations from the main body to three
new appendices to shorten the (preprint, no page limit) paper. Design principle:
**punchlines stay inline (claim + comparison table + interpretation), derivations
move**. Scope was "Moderate"; benchmark numerics (sec5/8/9) left in body as
empirical results.

## New files (input after `\bibliography` under `\appendix` in main.tex)
- `sections/appA_obcb.tex` (`\label{app:obcb}`) — Alice individual PN trace
  (`app:obcb_pn`), Alice PS trace (`app:obcb_ps`), Option-A SHAP value function
  (`app:obcb_vS`). Sources: sec2 PN/PS tracings, sec4 Option-A v(S) block.
- `sections/appB_signal.tex` (`\label{app:signal}`) — distributional facts
  (`app:signal_distfacts`), conditional expectations (`app:signal_condexp`),
  plain SHAP (`app:signal_plain`), Causal SHAP (`app:signal_causal`), PCI per-pair
  (`app:signal_pci`). Source: sec4 signal computation blocks. **Biggest win.**
- `sections/appC_desert.tex` (`\label{app:desert}`) — basic-DT structural model
  + PCI component spec. **Lightest**: sec7b's arithmetic already lived in the
  companion notebook, so only the model/spec setup moved; eq:nsj, tables, and the
  three observations stay in body.

## Body changes
- sec2: Alice PN/PS itemize+multline derivations → one-line pointers keeping the
  punchline numbers (0.045, 1). Table `tab:pn_ps_pns` stays.
- sec4: removed Plain/Causal SHAP per-target computations and PCI per-pair table;
  removed labels `sec:causal_shap_signal`, `sec:pci_signal_computation` (no
  dangling refs). Kept the plain-SHAP φ_Y=0.139 interpretive note (D-YX needs it),
  all tables, desiderata analysis. Option-A v(S) align block → pointer.
- sec7b: structural eqs + component itemize → 1 paragraph + pointer.

Build: `latexmk -pdf` clean, 65 pp, all `app:*` refs resolve.

Possible follow-up if more trimming wanted: sec4 Option-B NaN per-coalition
tracing (kept in body as it's the Causal-SHAP-breaks-on-Bob argument), and
sec5/8/9 result-table detail.
