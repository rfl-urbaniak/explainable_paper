# Pearl's Probability of Actual Causation — discussion proposal

**Date:** 2026-05-26
**Status:** Draft for review — nothing inserted into the paper yet.

The document is organised in two parts. **Sections 1–7** are the conceptual
walk-through and motivate the comparison; they're meant to be read top-to-bottom.
**Proposal A** and **Proposal B** at the end are paste-ready LaTeX inserts for the
paper itself.

---

_________________________________________
## PART TO INCLUDE EARLY ON

Pearl \citep[Ch.~10]{causalityPearl} defines the **probability of actual causation**
as

$$P(\text{caused}(x,y\mid e)) \;=\; \frac{P(U_{xy}\cap U_e)}{P(U_e)}
\qquad \text{(Def. 10.3.5)}$$

where $U_{xy}$ is the set of noise states $\mathbf{u}$ in which the chosen
actual-causation predicate "$X{=}x$ is an actual cause of $Y{=}y$" holds,
and $U_e$ is the set of states compatible with the observed evidence $e$. The
actual-causality predicate plugged in is Pearl's causal-beam definition
\citep[Ch.~10]{causalityPearl} or the Halpern--Pearl AC1--AC3
\citep{actualCausalityHalpern}.

The formula is Bayes' rule applied to actual causation. Each $\mathbf{u}$ determines
a deterministic world through the structural equations, so "is $X{=}x$ an actual
cause of $Y{=}y$?" has a yes/no answer once $\mathbf{u}$ is fixed; $U_{xy}$ collects
the yes-states. The ratio rewrites cleanly to

$$P(\text{caused}(x,y\mid e)) \;=\; P(\mathbf{u}\in U_{xy}\mid \mathbf{u}\in U_e),$$

i.e. *the posterior probability, given the evidence, that we are in a noise state
for which the actual-cause verdict holds*. This is Pearl's bridge from
actual causality to *probability of* actual causality. As our paper proposes to build
a seemingly similar bridge with PCI, and as Pearl's notion gives similar answers in
typical simple cases, we spend some time comparing it to PCI in
Section~\ref{sub:pearl_actual_cause_prob}, once enough technical machinery is on the
page to make the discussion precise.

__________________________________

## Pearl's Probability of Actual Causation and PCI


Pearl's Definition 10.3.5 \citep[Ch.~10]{causalityPearl},

$$P(\text{caused}(x,y\mid e)) \;=\; \frac{P(U_{xy}\cap U_e)}{P(U_e)}
\qquad \text{(Def. 10.3.5)},$$

is the posterior probability, given the evidence, that we are in a noise state
for which the actual-cause verdict holds. How does it differ from PCI, and
how does PCI improve on it?

We proceed in three steps. We first replay Pearl's canonical desert-traveller
illustration and show that PCI, on the original deterministic version, recovers
Pearl's verdicts (and Pearl's rankings). We then exhibit one operational gap ---
PCI's expectation collapses Pearl's existential-over-witnesses step into a single
graded integral --- and one substantive gap, where introducing stochastic mechanism
noise downstream of one of the causes splits the two methods on the *strength* of
attribution, even though they still agree on the *rank*. We close with two further
differences that follow from the construction without needing their own worked
example.




### Pearl's worked example: the desert traveller

We start with the canonical illustration used by Pearl
\citep[\S 10.3.3]{causalityPearl}.

\begin{example}[Desert traveller]\label{ex:desert_traveler}
A traveller has two enemies.
Enemy 2 shoots and empties the canteen ($X{=}1$); enemy 1, unaware, poisons the
water in the canteen ($P{=}1$). The traveller dies ($Y{=}1$). Who is the actual
cause of death?
\end{example}

The example involves one source of uncertainty, packed into a binary noise variable $u$:

- $u=0$: the traveller drank the poisoned water before the canteen was emptied, the
  cyanide killed him; the shooter is irrelevant.
- $u=1$: the canteen was empty before he could drink, dehydration killed him; the
  poisoning is irrelevant.


Pearl's beam analysis \citep[Ch.~10]{causalityPearl} (or the Halpern--Pearl
AC1--AC3 of \citealp{actualCausalityHalpern}, equivalently) at each
deterministic state gives the actual-cause verdicts:

- In $u=1$: $X{=}1$ is the actual cause; $P{=}1$ is not.
- In $u=0$: $P{=}1$ is the actual cause; $X{=}1$ is not.

So $U_{x,y} = \{u=1\}$ and $U_{p,y} = \{u=0\}$.

Without any extra evidence, Def. 10.3.5 results in the prior:

$$P(\text{caused}(x,y\mid e)) = P(u=1\mid e), \qquad P(\text{caused}(p,y\mid e)) = P(u=0\mid e).$$

Say we sharpen the evidence with a forensic report, *no cyanide in the body*, which is
incompatible with $u=0$. Then $U_e = \{u=1\}$, and

$$P(\text{caused}(x,y\mid e)) \;=\; \frac{P(\{u=1\}\cap\{u=1\})}{P(\{u=1\})} \;=\; 1.$$

The forensics drives the probability that the shooter was the actual cause to
certainty. A toxicology report finding cyanide would do the symmetric thing for the
poisoner.




## PCI on the same example

To put the two methods side by side, we compute the responsibility assigned to the
shot by PCI on the same model.

The endogenous variables are $X, P, C, D, Y \in \{0,1\}$, where $X$ is the shooter's
action, $P$ the poisoner's action, $C$ the cyanide-killed-him path indicator, $D$ the
dehydration-killed-him path indicator, and $Y$ death. The exogenous binary $u$ encodes
which of the two mechanisms gets the chance to act first. The structural equations,
following \citep[p.~323]{causalityPearl}, are
$$c = p\,(u' \vee x'), \qquad d = x\,(u \vee p'), \qquad y = c \vee d,$$
where $u', x', p'$ denote complements.

The factual values are $X=1, P=1, Y=1$, with prior $P(u=0)=P(u=1)=\tfrac{1}{2}$. The
factual values of $(C, D)$ are $(1,0)$ at $u=0$ and $(0,1)$ at $u=1$. The suspect set
is $\mathbf{S} = \{X, P\}$. The witness pool $\mathbf{W} \subseteq \{C, D\}$ gives
four possible active witness sets $\mathbf{T} \in \{\emptyset, \{C\}, \{D\}, \{C,D\}\}$.
The alternative-value distribution $\Delta$ deterministically flips $1\to 0$. The
AC-aligned kernel $\Phi(\mathbf{C},\mathbf{T}) = p^\Gamma \cdot \mathbb{I}[Y^s] \cdot Y^n$
reduces, in this binary deterministic case, to $\Phi > 0$ iff $Y$ flips when
$\mathbf{C}$ is set to its alternative and $\mathbf{T}$ is pinned at its factual value.

At $u=1$, only $(\{X\}, \{C\})$ flips: the required witness $C=0$ pins the cyanide
path at its dormant value; $X$ is the actual cause.
At $u=0$, only $(\{P\}, \{D\})$ flips: the witness $D=0$ pins the
dehydration path; $P$ is the actual cause. So $U_{x,y} = \{u=1\}$ and $U_{p,y} =
\{u=0\}$, exactly Pearl's verdicts in his (10.15)--(10.16).

Before evidence arrives, under uniform $\Gamma$ over the
four $\mathbf{T}$s and the uniform prior on $u$:

$$\mathbb{E}[\Phi\mid\mathbf{C}{=}\{X\}] \;=\; \tfrac{1}{2}\cdot\tfrac{1}{4} +
\tfrac{1}{2}\cdot 0 \;=\; \tfrac{1}{8}, \qquad
\mathbb{E}[\Phi\mid\mathbf{C}{=}\{P\}] \;=\; \tfrac{1}{8}.$$

We have equal responsibilities assigned to the two candidates,
the expected reading for a symmetric preemption setup with a
symmetric prior. Pearl's Def. 10.3.5 says the same thing in its own units:
$P(\text{caused})=\tfrac{1}{2}$ for each cause.

Conditioning on the "no cyanide" forensic report (so $P(u=1\mid e) = 1$):

$$\mathbb{E}[\Phi\mid\mathbf{C}{=}\{X\}, e] \;=\; \tfrac{1}{4}, \qquad
\mathbb{E}[\Phi\mid\mathbf{C}{=}\{P\}, e] \;=\; 0;$$

Pearl's Def. 10.3.5 gives $1$ and $0$. Both methods flip the verdict to "shooter
caused death, poisoner did not". The sign and ordering match in both regimes. The
magnitudes do not --- PCI's $\tfrac{1}{4}$ is the $\Gamma$-fraction of witnesses
passing the necessity test, while Pearl's $1$ is the posterior probability of a
single binary claim. That is, PCI is sensitive to how many context settings
there are and what happens in those contexts, whereas Pearl's metric jumps to one as
soon as a single witness set exists that validates the actual-causality claim. The two
readings answer somewhat different questions, but they ride together on the ranking.

## PCI expectation vs the AC machinery

The agreement above is reassuring, but the more interesting observation is what
changes underneath. **AC's necessity clause is an existential**: *there exists* a
witness set $\mathbf{T}$ and an alternative value $c'$ such that the intervention
flips $Y$. To verify the existential, one in principle enumerates the candidate
$(\mathbf{T}, c')$ pairs --- this is the main source of AC's combinatorial hardness.
Pearl's definition then averages the indicator of that already-enumerated predicate
over noise states, inheriting the enumeration unchanged.

Instead, **PCI collapses both moves into one expectation.** $\Phi$ is averaged over
the joint $(\mathbf{C}, \mathbf{T}, \mathbf{u})$ under $\Gamma \otimes P_{\mathbf{U}}$.
The existential "does *some* witness work?" becomes the quantitative "*what
fraction* of witnesses work?" --- the information AC uses only one piece of,
exposed as a graded score with sampling-based estimation.

In the desert traveller this is small change: four candidate witness sets, all
trivially enumerable. The point is the scaling. In a model with $n$ candidate
witness variables, AC's existential demands checking up to $2^n$ subsets per noise
state, and Pearl's posterior over the AC indicator inherits that cost per
evaluation. PCI's expectation can be estimated at a Monte Carlo budget the user
controls, with variance independent of the problem's combinatorial size. The
desert-traveller agreement is the small-$n$ end of a curve whose large-$n$ end is
the computational gap that motivates PCI in the first place.

So far we have two conceptual differences that look mainly *operational*.
But there are also more substantive cases where the two methods give qualitatively
different answers about who is responsible.

## Desert traveller with asymmetric sufficiency

Note however that PCI involves sufficiency considerations that are absent in actual
causality and therefore in Pearl's probability of actual causality. The desert
traveller is a clean preemption case: at any single noise state, one of the two
enemies is uniquely responsible, so necessity and sufficiency coincide and PCI's
sufficiency factor $Y^s$ adds nothing to the ranking.

To exhibit the second contribution PCI makes --- a real sufficiency reading,
separable from necessity --- we introduce noise downstream of the poisoning so that
the poisoner's mechanism becomes statistically unreliable while the shooter's stays
deterministic.

\begin{example}[Desert traveller, weak poison]\label{ex:desert_traveler_weak_poison}
Same enemies as in Example~\ref{ex:desert_traveler}. The cyanide is a small dose
that turns out fatal only with probability $\alpha = 0.1$ (independent exogenous
noise $\xi$); $\alpha = 1$ recovers Pearl's original. The structural equations are
$$c = p\,(u' \vee x'), \quad v_C = c\cdot \xi, \quad d = x\,(u \vee p'),
\quad y = v_C \vee d.$$
\end{example}

Again, we have two scenarios under the same observation $X=1, P=1, Y=1$. Forensic
analysis identifies the mechanism after the fact:

- *Scenario A --- cyanide killed*: $u=0$ (drank early), $\xi=1$ (this dose happened
  to be fatal).
- *Scenario B --- dehydration killed*: $u=1$ (drank late), $\xi$ irrelevant.

Given the forensic evidence, the probability of actual causality makes the situation
symmetric. The AC indicator is true with posterior certainty in each case:
$P(\text{caused}(p,y\mid e_A)) = 1$ in A, $P(\text{caused}(x,y\mid e_B)) = 1$ in B.
Each enemy is the actual cause of his victim, full probability, end of story.

PCI's necessity factor matches Pearl's notion. The necessity component flips for $P$
in A and for $X$ in B, with the same witness analysis as before. So far the two
methods produce the same ranking.

However, PCI's sufficiency factor distinguishes the scenarios. The sufficiency
evaluation is sensitive to how reliably the cause produces the effect, and the two
enemies' sufficiency probabilities differ by an order of magnitude. With a $ci$ that
combines necessity and sufficiency (the PNS-product $ci$ of
Section~\ref{sec:causal_impact}, or any monotone composition of $Y^s$ and $Y^n$), the
two scenarios read:

| Scenario               | Def. 10.3.5    | PCI sufficiency $\mathrm{PS}$  | PCI composite |
|------------------------|:--------------:|:------------------------------:|:-------------:|
| A --- poisoner caused  | $1$ (certain)  | $0.1$                          | low           |
| B --- shooter caused   | $1$ (certain)  | $0.76$                         | high          |

The intuitive payoff is that a juror weighing the two deaths would plausibly treat
the shooter in B as more clearly to blame than the poisoner in A. The shooter's act
was overwhelmingly likely to bring about the outcome; the poisoner's was
overwhelmingly likely *not* to --- the poisoner needed both the right circumstances
(early drinking) and a lucky cyanide outcome on top, neither of which were robust
features of his action. PCI matches this intuition because it has a sufficiency
factor to read off. Def. 10.3.5 does not, because the AC indicator at its core does
not separate counterfactual reinstatement from counterfactual removal.

## Further differences between Pearl's probability of causation and PCI

Three further differences are worth flagging but don't need their own worked example.


**Breadth of aggregation.** Def. 10.3.5 aggregates over one dimension: the noise
$\mathbf{u}$, conditional on evidence $e$, for a claim $(X=x, Y=y)$ already fixed.
PCI aggregates over four: the suspect set $\mathbf{C}$ and witness set $\mathbf{T}$
via $\Gamma$, the alternative value $\mathbf{c}'$ via $\Delta$, and the noise via
$P_{\mathbf{U}}$. This is what supports per-feature comparison --- the same PCI
expectation, evaluated over varying $\mathbf{C}$ inside a fixed $\mathbf{S}$,
returns the relative responsibility of each candidate cause. Def. 10.3.5 cannot
perform that comparison without a separate evaluation per claim, each inheriting
the AC-predicate cost we already discussed.

**Continuous variables.** The AC predicates inside Def. 10.3.5 are event-shaped:
they read off $\mathrm{AC}(x,y;M,\mathbf{u})$ for propositional $x, y$, and
extensions to continuous variables require discretising the relevant event before
the predicate applies. PCI's $ci$ function is continuous-native (the
absolute-difference impact score of Section~\ref{sec:causal_impact}), and the
synthetic benchmark of Section~\ref{sub:synthetic} exercises that continuity
directly.

**Lineage in Pearl's PN/PS.** Def. 10.3.5 takes a binary actual-causation predicate
--- itself a structural-equation construction that does not appeal to PN/PS at all
--- and lifts it to a probability through $P(\mathbf{u}\mid e)$; PN, PS, and PNS
appear nowhere in the construction. PCI, by contrast, generalises PN and PS
directly: the necessity factor $Y^n$ is the counterfactual operation underlying
$\mathrm{PN}$, the sufficiency factor $Y^s$ is the counterfactual operation
underlying $\mathrm{PS}$, both indexed by the cause--witness--alternative--noise
tuple, with $\Gamma$ and $\Delta$ exposing the design choices PN/PS makes
implicitly. So Pearl's Def. 10.3.5 and PCI are not competing implementations of the
same idea: they inherit from different parts of Pearl's own toolkit, with PCI
playing the role of a context-sensitive, multi-variable extension of the PN/PS
machinery rather than a probabilistic wrapper around AC.


Pearl's Def. 10.3.5 and PCI share the same explanatory target --- probabilistic
responsibility built on top of structural information --- but instantiate it through
different mechanisms. Pearl's definition averages a binary AC indicator against a
posterior over noise; PCI integrates a continuous necessity--sufficiency kernel
against a joint distribution over suspect sets, witness sets, alternative values,
and noise.

On clean preemption cases, the two agree on ranking. They diverge in two
substantive places: (1) the expectation replaces an existential, and
(2) the sufficiency factor is separable in PCI but absent from Pearl's definition.
When two causes differ in how reliably they reinstate the outcome, as the shooter
and the poisoner do in the weak-poison variant, PCI separates them in a way that
matches juror-style responsibility intuitions. Pearl's Def. 10.3.5 does not.

So while inspired by similar problems and considerations,
the two constructions answer related but distinct questions, and the question PCI
answers is the one we need for ML-grade causal attribution.

---

## Placement (decided in conversation)

- **§2 motivations — early flag (Proposal A0)**: short forward-reference paragraph
  between the section's opening framing (`sections/sec2_motivations.tex:6`) and the
  `\subsection{Actual Causality}` heading (line 8). Job: plant a flag the first time
  the reader encounters §2, so the substantive discussion in §7 doesn't come out
  of nowhere.
- **§2 motivations — substantive paragraph (Proposal A)**: one paragraph between
  the PNS table and `\subsection{Path forward}` (inserted at
  `sections/sec2_motivations.tex:286`). Job: pre-empt "but Pearl already did this"
  before §3 opens.
- **§7 Relation with Actual Causality (Proposal B)**: new subsection appended after
  `sections/sec7_actual_causality.tex:454`. Job: the technical contrast in detail.

---


```

### Comments / questions for A0

> *(leave inline comments below this line)*

-

---

## Proposal A — §2 paragraph (LaTeX-ready)

```latex
Pearl himself addresses this gap in \citep[Ch.~10]{causalityPearl}: the \emph{probability
of actual causation}, $P(\text{caused}(x,y\mid e)) = P(U_{xy}\cap U_e)/P(U_e)$
(Definition~10.3.5), where $U_{xy}$ collects the noise states $\mathbf{u}$ in which a
chosen actual-cause predicate holds. The diagnosis is the one we just made: PN/PS/PNS are
global features of $Y_x(\mathbf{u})$ and miss the structural detail that distinguishes
Alice's case from Bob's, so probabilistic responsibility has to be built on top of a
scenario-specific predicate rather than on the marginal counterfactual alone. Pearl's
construction takes a binary actual-cause predicate --- the causal-beam definition or the
Halpern--Pearl variant \citep{actualCausalityHalpern} --- and averages its indicator
against the posterior $P(\mathbf{u}\mid e)$. \Ourapproach{} shares the diagnosis but
neither the binary indicator nor the AC predicate at its core: it integrates a continuous
necessity--sufficiency kernel against a joint distribution over suspect sets, witness
sets, alternative values, and noise. Section~\ref{sec:relation_with_ac} develops the
contrast in detail; we flag here only that Definition~10.3.5 is \ourapproach{}'s closest
neighbour in the literature, and that the two differ on each of the four axes that govern
tractability and the kind of question that can be answered.
```

### Comments / questions for A

> *(leave inline comments below this line)*

-

---

## Proposal B — §7 subsection (LaTeX-ready)

```latex
\subsection{Comparison with Pearl's Probability of Actual Causation}
\label{sub:pearl_actual_cause_prob}

Pearl's Definition~10.3.5 in \citep{causalityPearl} proposes a probabilistic generalisation
of actual causation that addresses the same diagnostic gap motivating \ourapproach{}: PN/PS/PNS
are global features of the response function $Y_x(\mathbf{u})$ and miss scenario-specific
structural information, so probabilistic responsibility ought to be built on top of a
structural predicate.\footnote{\label{fn:desert_notebook}The computations supporting the
comparison in this section --- both Pearl's original desert-traveller example and the
weak-poison variant used to separate necessity from sufficiency --- are collected in the
companion notebook \texttt{docs/source/desert\_traveler.ipynb}.} Concretely, for a chosen
actual-causation predicate
$\mathrm{AC}(x,y;M,\mathbf{u})$ --- the causal-beam definition of
\citep[Ch.~10]{causalityPearl} in the first edition, the Halpern--Pearl 2005 AC1--AC3 in the
second-edition postscript and in \citep{actualCausalityHalpern} --- Pearl defines
\begin{equation}\label{eq:pearl-paac}
P(\text{caused}(x,y\mid e)) \;=\; \frac{P(U_{xy}\cap U_e)}{P(U_e)},
\qquad U_{xy}=\{\mathbf{u}\colon \mathrm{AC}(x,y;M,\mathbf{u})\},
\end{equation}
the posterior expectation of the AC indicator under $P(\mathbf{u}\mid e)$. The same
construction underwrites Chockler and Halpern's degree of blame
\citep{chocklerResponsibilityBlameStructuralModel2004}, which substitutes a graded
responsibility share inside the expectation. Because \eqref{eq:pearl-paac} is the closest
construction in the literature to \ourapproach{}'s expectation in
Definition~\ref{def:jointnecsuf}, the differences are worth recording in detail.

\medskip\noindent\textit{(i) Estimand shape.}
\eqref{eq:pearl-paac} is a posterior expectation of the binary indicator
$\mathbb{I}[\mathrm{AC}]$; its value is a probability that a fixed claim ``$X=x$ caused
$Y=y$'' is true under the model. \ourapproach{}'s
$\mathbb{E}_{\Gamma\otimes\Delta\otimes P_{\mathbf{U}}}[\Phi]$ is the expectation of a
continuous kernel that combines a sufficiency factor $Y^s$ and a necessity factor $Y^n$ with
a $\Gamma$-weight; its value is a graded responsibility that ranks candidate causes against
each other, rather than evaluating a sharp predicate. The two notions answer different
questions: \eqref{eq:pearl-paac} answers ``how probable is the AC verdict?'' for a claim
already on the table; $\mathbb{E}[\Phi]$ answers ``how much of the outcome's causal weight
does this variable carry?'' \Ourapproach{} therefore returns interpretable comparisons
across features --- the use case for the synthetic and AVM evaluations of
Sections~\ref{sub:synthetic} and \ref{sec:avm} --- where \eqref{eq:pearl-paac} returns a
single number per pre-specified claim.

\medskip\noindent\textit{(ii) Tractability.}
The cost of evaluating \eqref{eq:pearl-paac} is the cost of the AC predicate it embeds, and
the structure-based AC family is intractable in general
\citep{eiterComplexityResultsStructurebased2002}: the existential ``does some witness work?''
inside AC1--AC3 in principle enumerates up to $2^{|\mathbf{T}|}$ subsets per noise state, and
the posterior averaging in \eqref{eq:pearl-paac} inherits that cost per noise sample.
\ourapproach{}'s kernel $\Phi$ embeds no AC oracle: evaluating it at a sampled
$(\mathbf{C},\mathbf{c}',\mathbf{T},\mathbf{u})$ costs one twin-network counterfactual per
factor, and the expectation is approximated by Monte Carlo over $\Gamma$, $\Delta$, and
$P_{\mathbf{U}}$ at a budget the user controls, with variance independent of the
combinatorial size of the model. Where AC alignment is specifically wanted,
Proposition~\ref{prop:filter} recovers it through the same expectation rather than through
the AC predicate itself.

\medskip\noindent\textit{(iii) Breadth of aggregation.}
\eqref{eq:pearl-paac} aggregates over a single dimension: the noise $\mathbf{u}$,
conditional on evidence $e$, for a claim $(X=x, Y=y)$ already fixed. \Ourapproach{}
aggregates over four: the suspect set $\mathbf{C}$ and witness set $\mathbf{T}$ via
$\Gamma$, the alternative value $\mathbf{c}'$ via $\Delta$, and the noise $\mathbf{u}$ via
$P_{\mathbf{U}}$. This is what supports per-feature comparison: $\Gamma$ exposes the choice
of which set to ask about as a first-class component, so the same expectation evaluated over
varying $\mathbf{C}$ inside a fixed $\mathbf{S}$ returns the relative responsibility of each
candidate. \eqref{eq:pearl-paac} cannot perform this comparison without a separate
evaluation per claim, each inheriting (ii)'s cost.

\medskip\noindent\textit{(iv) Continuous variables and symmetric sufficiency.}
The AC predicates inside \eqref{eq:pearl-paac} are event-shaped: they read off
$\mathrm{AC}(x,y;M,\mathbf{u})$ for propositional $x,y$, with extensions to continuous
variables discretising the relevant event. \Ourapproach{}'s $ci$ function is
continuous-native (see the absolute-difference impact score in
Section~\ref{sec:causal_impact}), and the synthetic benchmark of Section~\ref{sub:synthetic}
exercises that continuity directly. Sufficiency enters \eqref{eq:pearl-paac} only as part of
the AC predicate's sustenance leg, which is a structural-contingency check rather than a
counterfactual-reinstatement probability \`a la Pearl PS; \ourapproach{} keeps the necessity
factor $Y^n$ and the sufficiency factor $Y^s$ separate in the kernel, both indexed by
$(\mathbf{C},\mathbf{c}',\mathbf{T},\mathbf{u})$, and the synthetic benchmark shows that the
two factors discriminate causal archetypes that PNS-only and AC-only scores conflate. The
desert-traveller weak-poison variant in the notebook of footnote~\ref{fn:desert_notebook}
makes the point concretely: with full forensic evidence, both shooter and poisoner receive
$P(\text{caused}) = 1$ under \eqref{eq:pearl-paac} even though the shooter's mechanism is
deterministic and the poisoner's fires with probability $\alpha = 0.1$; \ourapproach{}'s
sufficiency factor separates them, ranking the shooter strictly above the poisoner.

\medskip\noindent\textit{(v) Lineage in Pearl's PN/PS.}
\eqref{eq:pearl-paac} and \ourapproach{} are not competing implementations of the same
underlying object: they inherit from different parts of Pearl's own toolkit.
\eqref{eq:pearl-paac} takes a binary actual-causation predicate --- itself a
structural-equation construction that does not appeal to $\mathrm{PN}$ or $\mathrm{PS}$ ---
and lifts it to a probability through $P(\mathbf{u}\mid e)$; $\mathrm{PN}$, $\mathrm{PS}$,
and $\mathrm{PNS}$ appear nowhere in the formula. \ourapproach{} generalises Pearl's
$\mathrm{PN}$ and $\mathrm{PS}$ directly: the necessity factor $Y^n$ is the counterfactual
operation underlying $\mathrm{PN}$, the sufficiency factor $Y^s$ is the counterfactual
operation underlying $\mathrm{PS}$, both indexed by the cause--witness--alternative--noise
tuple, with $\Gamma$ and $\Delta$ exposing as design distributions the choices
$\mathrm{PN}/\mathrm{PS}$ makes implicitly. In this sense \ourapproach{} plays the role of
a context-sensitive, multi-variable extension of $\mathrm{PN}/\mathrm{PS}$ rather than a
probabilistic wrapper around AC, and the difference shows up exactly where AC and
$\mathrm{PN}/\mathrm{PS}$ themselves diverge.

\medskip
Taken together, Pearl's Definition~10.3.5 and \ourapproach{} share an explanatory target ---
probabilistic responsibility built on top of structural information --- but instantiate it
through different mechanisms: an AC-indicator expectation against $P(\mathbf{u}\mid e)$ in
the first case, a continuous necessity--sufficiency kernel against
$\Gamma\otimes\Delta\otimes P_{\mathbf{U}}$ in the second. The first is the natural
construction if a sharp AC predicate is given and a posterior over $\mathbf{u}$ is the only
object one wants to integrate against; the second is the natural construction if
responsibility is to scale to ML-grade models and to support cross-feature comparison.
```

### Comments / questions for B

> *(leave inline comments below this line)*

-

---

## Notes / flags

- **Citations used** (all already in `references.bib`, no new entries needed):
  `causalityPearl`, `actualCausalityHalpern`,
  `chocklerResponsibilityBlameStructuralModel2004`,
  `eiterComplexityResultsStructurebased2002`. (The earlier draft also cited
  `Halpern2015-HALGCA` and `icardNormalityActualCausal2017` in the stability point;
  those are no longer needed in this subsection now that the stability argument is
  dropped --- they remain available if we want them in §11.)

- **Stability point dropped.** The previous (ii) bundled tractability with a
  "definitional stability" argument (AC predicate has shifted across editions;
  PCI's estimand is invariant). That argument cuts both ways: PCI is itself a
  family of constructions indexed by the $ci$ function, and saying the AC predicate
  is unstable while $ci$ is stable would be inviting an answer in kind. (ii) now
  reads as tractability only.

- **New point added: (v) Lineage in Pearl's PN/PS.** \eqref{eq:pearl-paac} is built
  on the AC predicate and does not appeal to $\mathrm{PN}/\mathrm{PS}$ at all; PCI
  generalises $\mathrm{PN}$ and $\mathrm{PS}$ directly. So the two are not
  competing implementations of one idea --- they inherit from different parts of
  Pearl's toolkit. This frames PCI as a context-sensitive extension of the
  $\mathrm{PN}/\mathrm{PS}$ machinery rather than as a competitor to AC.

- **Notebook footnote.** Subsection B now footnotes
  `docs/source/desert_traveler.ipynb` for the computations supporting both the
  original desert-traveller agreement and the weak-poison sufficiency split.

- **Soft complexity claim, deliberate.** I wrote "structure-based AC family is
  intractable in general" rather than naming a complexity class.
  Eiter--Lukasiewicz 2002 proves $\Sigma_2^P$-completeness for HP 2001; HP 2005
  and the normality-refined variants are in the same neighbourhood but the
  cleanest defensible claim is intractability. Promote to the sharper
  $\Sigma_2^P$ claim?

- **Length.** Subsection B is now ~95 lines of LaTeX with the new (v) and the
  notebook footnote, ~3/4 page. Cut path if needed: collapse (i)+(iii) into one
  paragraph, or shorten (v) to two sentences.

- **Chockler--Halpern degree of blame.** Cited once in B.0 but really is the
  existing literature's "graded responsibility" notion. Should it get its own
  paragraph in §11 related-work, or a footnote in §7? Currently a one-line
  mention is light.

- **Forward references resolved.** A: `sec:relation_with_ac`. B: `def:jointnecsuf`,
  `prop:filter`, `sub:synthetic`, `sec:avm`, `sec:causal_impact`. All exist.

- **Decisions pending.** (1) approve A, B, both, or revisions; (2) promote
  complexity claim?; (3) Chockler--Halpern follow-up; (4) any preferred trim of B;
  (5) the new (v) is the natural place to mention that PCI's $\Gamma$/$\Delta$
  surface the design choices PN/PS leaves implicit --- worth a single explicit
  forward reference to Section~3 ($\Gamma$ / $\Delta$ definitions), or kept terse?
