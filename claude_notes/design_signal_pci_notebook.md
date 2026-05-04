# Design note — Signal-with-mediation PCI notebook

**Date:** 2026-05-04
**Goal:** Add a separate notebook
`claude_notes/signal_mediation_computations.ipynb` that verifies every value
in Section 4.4 (plain SHAP), Section 4.5 (Causal SHAP), and Section 4.6
(desiderata table) of `main.tex`, and that computes a third column for PCI
with witness = the third variable, so the desiderata table in Section 4.6
can grow a PCI column.

This note is the design plan; nothing committed to `main.tex` yet.

---

## 1. The model

Signal-with-mediation chain:
- $X \sim \mathcal{N}(0.5,\, 0.25)$
- $M = X + \varepsilon_M$, $\varepsilon_M \sim \mathcal{N}(0,\, 0.1)$
- $Y = M + \varepsilon_Y$, $\varepsilon_Y \sim \mathcal{N}(0,\, 0.1)$

All noise terms are independent of each other and of $X$. Everything is jointly
Gaussian; conditional expectations are linear in the conditioning variables and
SHAP/Causal SHAP have closed-form values.

**Joint moments:**
- $\mathbb{E}[X] = \mathbb{E}[M] = \mathbb{E}[Y] = 0.5$
- $\mathrm{Var}(X) = 0.25$, $\mathrm{Var}(M) = 0.35$, $\mathrm{Var}(Y) = 0.45$
- $\mathrm{Cov}(X, M) = 0.25$, $\mathrm{Cov}(X, Y) = 0.25$, $\mathrm{Cov}(M, Y) = 0.35$

**Optimal predictors** (conditional expectations, derived in Section 4.3):
- $f_Y(X, M) = \mathbb{E}[Y \mid X, M] = M$
- $f_M(X, Y) = \mathbb{E}[M \mid X, Y] = 0.5 X + 0.5 Y$
- $f_X(M, Y) = \mathbb{E}[X \mid M, Y] = 0.5 + (5/7)(M - 0.5)$
  (no $Y$ dependence: $X \perp Y \mid M$)

**Factual instance:** $X = M = Y = 1$.

---

## 2. SHAP and Causal SHAP coalition values to verify

For each target $B \in \{Y, M, X\}$ the feature set is the other two variables,
giving a 2-feature game with four coalition values per target. Table values from
`main.tex` Sections 4.4 and 4.5:

### Target $Y$, features $\{X, M\}$
|                          | $v_{\mathrm{plain}}$ | $v_{\mathrm{causal}}$ |
|--------------------------|----------------------|-----------------------|
| $\emptyset$              | $0.5$                | $0.5$                 |
| $\{X\}$                  | $1$                  | $1$                   |
| $\{M\}$                  | $1$                  | $1$                   |
| $\{X, M\}$               | $1$                  | $1$                   |

Resulting Shapley: $\phi_X = \phi_M = 0.25$ (both methods identical).

### Target $M$, features $\{X, Y\}$
|                          | $v_{\mathrm{plain}}$ | $v_{\mathrm{causal}}$ |
|--------------------------|----------------------|-----------------------|
| $\emptyset$              | $0.5$                | $0.5$                 |
| $\{X\}$                  | $1$                  | $1$                   |
| $\{Y\}$                  | $0.889$              | $0.75$                |
| $\{X, Y\}$               | $1$                  | $1$                   |

Plain Shapley: $\phi_X = 0.306$, $\phi_Y = 0.194$.
Causal Shapley: $\phi_X = 0.375$, $\phi_Y = 0.125$.

### Target $X$, features $\{M, Y\}$
|                          | $v_{\mathrm{plain}}$ | $v_{\mathrm{causal}}$ |
|--------------------------|----------------------|-----------------------|
| $\emptyset$              | $0.5$                | $0.5$                 |
| $\{M\}$                  | $0.857$              | $0.857$               |
| $\{Y\}$                  | $0.778$              | $0.5$                 |
| $\{M, Y\}$               | $0.857$              | $0.857$               |

Plain Shapley: $\phi_M = 0.219$, $\phi_Y = 0.139$.
Causal Shapley: $\phi_M = 0.357$, $\phi_Y = 0$.

The notebook should reproduce all four columns above exactly (closed-form
Gaussian conditioning), and assert against the paper values to within $10^{-3}$.

---

## 3. PCI with witness = third variable

The user's request: for each cell $\phi_A^B$ in `tab:signal_desid`, also compute
PCI(A → B) with witness W = the third variable (the one that's neither A nor
B), held at its factual value.

### Mapping cells to (target, suspect, witness)

| Desideratum | Cell        | Target $B$ | Suspect $A$ | Witness $W$ |
|-------------|-------------|------------|-------------|-------------|
| DXY         | $\phi_X^Y$  | $Y$        | $X$         | $M$         |
| DMY         | $\phi_M^Y$  | $Y$        | $M$         | $X$         |
| DXM         | $\phi_X^M$  | $M$        | $X$         | $Y$         |
| DMX         | $\phi_M^X$  | $X$        | $M$         | $Y$         |
| DYX         | $\phi_Y^X$  | $X$        | $Y$         | $M$         |
| DYM         | $\phi_Y^M$  | $M$        | $Y$         | $X$         |

Note: DMXY and D-ind don't fit this single-cell template; they are summary
desiderata across cells and don't get an independent PCI value. The notebook
should report PCI for the six single-cell desiderata.

### PCI specification for the signal example

PCI ingredients we need (Definitions in Section 3 of the paper):
- **Causal model:** the SCM above.
- **Suspect set $\mathbf{S}_k = \{A\}$** for the target.
- **Witness set $\mathbf{W} = \{W\}$**, fixed at factual value $w^\star = 1$.
- **Variable selection distribution $\Gamma_s$:** point mass on $\{A\}$.
  (We are computing a single-suspect PCI per cell, no Shapley-style averaging
  over suspect subsets.)
- **Witness selection distribution $\Gamma_w$:** point mass on $\{W\}$.
  (We are committing to the third variable as witness; $\emptyset$ option
  not included.)
- **Alternative-value distribution $\Delta$ for $A$:** marginal $P(A)$.
  Continuous, so we take $A' \sim P(A)$.
- **Causal impact function $ci$:** Absolute Difference Impact Score
  (Example~\ref{ex:absolute_score} in the paper):
  $$ci(y^s, y^n, y^\star) = |y^n - y^\star| - |y^s - y^\star|.$$
  In the sufficiency world $A=a^\star$ is held at factual, $W=w^\star$, so
  $y^s = B^\star = 1$ and $|y^s - y^\star| = 0$.
  In the necessity world $A=a'$ is set to alternative, $W=w^\star$, and
  $y^n$ is sampled from the SCM under those interventions.

So the PCI value reduces to
$$\mathrm{PCI}(A \to B \mid W) \;=\; \mathbb{E}_{A' \sim P(A),\, U}\bigl[\,|y^n(A', U) - B^\star|\,\bigr],$$
where $U$ collects the SCM's free noise terms (which ones are free depends on
how partial vs full abduction is configured; we'll use partial abduction —
sample the noise terms not constrained by witness/factual observation).

### Per-cell mechanics

For each of the six cells, we'll need to derive $y^n$ analytically.

**DXY: PCI(X → Y | W = M).**
- Sufficiency world: $X = 1$, $M = 1$ (witness), $\varepsilon_Y \sim \mathcal{N}(0, 0.1)$.
  $y^s = M + \varepsilon_Y$. $\mathbb{E}[y^s] = 1$. (Or we could fix $\varepsilon_Y$
  at factual; under partial abduction with no $Y$-side observation we sample.)
- Necessity world: $X = X' \sim P(X)$, $M = 1$ (witness override), $\varepsilon_Y$
  sampled. $y^n = M + \varepsilon_Y = 1 + \varepsilon_Y$.
- $|y^n - y^\star|$ in expectation: $\mathbb{E}|\varepsilon_Y| = \sqrt{0.1}\sqrt{2/\pi} \approx 0.252$.
- The witness blocks the $X \to M \to Y$ path: changing $X$ has no effect on
  $Y$ when $M$ is held fixed. So PCI(X → Y | M) is just the noise-driven
  baseline $\mathbb{E}|\varepsilon_Y|$.

This already exposes a tension: with $M$ as witness, the witness blocks the
only causal path from $X$ to $Y$, so PCI registers only random noise rather
than a substantive signal. The closed-form prediction is $\approx 0.252$.

**DMY: PCI(M → Y | W = X).**
- Sufficiency world: $M = 1$ (suspect at factual), $X = 1$ (witness), $\varepsilon_Y$ sampled.
  $y^s = M + \varepsilon_Y = 1 + \varepsilon_Y$.
- Necessity world: $M = M'$ where $M'$ is sampled from $P(M)$ (alternative for
  the suspect). $X = 1$ (witness). $y^n = M' + \varepsilon_Y$.
- Wait — but $M$ has a structural equation $M = X + \varepsilon_M$. If we
  intervene $do(M = M')$, the structural equation is overridden. So under
  $do(M = M')$, $M$ is forced to $M'$, and $Y = M + \varepsilon_Y = M' + \varepsilon_Y$.
- $|y^n - y^\star| = |M' + \varepsilon_Y - 1|$. With $M' \sim \mathcal{N}(0.5, 0.35)$
  and $\varepsilon_Y \sim \mathcal{N}(0, 0.1)$, $M' + \varepsilon_Y - 1 \sim \mathcal{N}(-0.5, 0.45)$.
  Expected absolute value of $\mathcal{N}(\mu, \sigma^2)$:
  $\sigma\sqrt{2/\pi}\exp(-\mu^2/(2\sigma^2)) + \mu(1 - 2\Phi(-\mu/\sigma))$.

The notebook should compute these numerically (Monte Carlo + closed-form for
verification) and assemble all six PCI values. That gives us a column for
the desiderata table.

### Notes / open questions

- Whether to use full or partial abduction is a modelling choice. Partial
  matches what the OBCB notebook uses. We'll go with partial abduction.
- For the upstream-cause cells (DYX, DYM, DMX), where the suspect $A$ is
  downstream of the target $B$ in the SCM, intervening on $A$ has no
  structural effect on $B$. PCI should naturally return a small/zero
  attribution; this is the desideratum these cells encode. The notebook
  should confirm.
- The `ci` choice (Absolute Difference) is one option. We could also try a
  signed difference $y^n - y^\star$ if the user wants directional info. Stick
  with absolute for now to align with the desiderata's $|\phi|$ comparison.

---

## 4. Notebook structure (planned)

1. **Setup.** Import numpy, scipy.stats; declare model parameters and factual
   instance.
2. **Joint moments and predictors.** Verify $\mathbb{E}[B]$, $\mathrm{Var}$,
   $\mathrm{Cov}$ match the paper; verify $f_Y, f_M, f_X$ closed forms via
   linear regression on Monte Carlo samples.
3. **SHAP coalition values.** Compute the four cells per target (3 targets ×
   4 = 12 values); assert against paper values.
4. **Causal SHAP coalition values.** Same 12 values under do-conditioning;
   assert against paper.
5. **Shapley assembly.** Plug coalition values into the closed-form 2-feature
   Shapley; verify all 6 plain and 6 causal $\phi_A^B$ match the paper.
6. **PCI with third-variable witness.** Compute the 6 cells using:
   - Closed-form Gaussian when possible (necessity world is Gaussian, so
     $|y^n - y^\star|$ has a known mean).
   - Monte Carlo sanity check.
7. **Desiderata table reproduction.** Reassemble the table from
   `tab:signal_desid` with an extra PCI column.

The notebook should be executable end-to-end and produce a final summary
table that the paper's `tab:signal_desid` extension can refer back to.

---

## 5. After the notebook

Once the notebook runs and PCI values are stable:

1. Add a "PCI (witness = third var)" column to `tab:signal_desid` in
   `main.tex` Section 4.6.
2. Adjust the desideratum-by-desideratum prose discussion below the table
   to comment on what PCI does on each cell — particularly: does the witness
   mechanism over-block (DXY, DMY, DXM) by registering only noise; does it
   correctly zero out the non-causal cells (DYX, DYM, DMX); does DMXY hold;
   what about D-ind.
3. The closing paragraph of Section 4 (post-Section-4.6 wrap-up) may need
   a sentence about how PCI fares on the signal example, to balance the
   already-stated "neither SHAP variant reliably satisfies the desiderata"
   claim.

---

## 6. Risks / things to flag

- **Witness over-blocking on DXY.** If PCI(X → Y | W = M) ≈ 0 because the
  witness blocks the only causal path, this is a genuine result but may
  require careful framing: it is not a failure of PCI but a consequence of
  the chosen witness set. The natural mitigation is to also report PCI with
  $\Gamma_w$ uniform over $\{\emptyset, \{W\}\}$ (as we did for OBCB).
- **Continuous $\Delta$.** The OBCB notebook used a binary alternative-value
  distribution. For Gaussian features the natural $\Delta$ is the marginal
  $P(A)$. The paper's Section 3 allows this (Definition of $\Delta$ permits
  any probability distribution on $\mathrm{dom}(A)$).
- **Norming.** PCI values from the Absolute Difference $ci$ are not directly
  comparable to SHAP values (different units). The desiderata table compares
  signs/zeros and ranks, so this should be OK, but worth stating explicitly
  in the notebook commentary.
