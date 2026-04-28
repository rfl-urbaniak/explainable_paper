




### CODE


- [ ] Implementations
    - [ ] SHAP as applicable to our models (see roc360 experiments for a prototype)
    - [ ] Causal SHAP as applicable to our models (see the causal SHAP papers)


- [ ] Notebook 1. Basic categorical cases
    - [ ] Notebook 1. with the basic categorical case comparing actual causality, SHAP cSHAP. 
    - [ ] Notebook 1. Add variants to Notebook 1 that clearly emphasize/validate necessity and sufficiency separately, benchmark against SHAP and cSHAP
    
- [ ] Notebook 2. Categorical benchmarking on generalized stone-throwing example
    - [ ] Notebook 2. Comparison to SHAP
    - [ ] Notebook 2. Comparison to cSHAP

- Notebook 3.

- [ ] Notebook 4. Gradient-based attributions
    - [ ] Notebook 4. Potentially comparison to SHAP
    - [ ] Notebook 4. Potentially comparison to cSHAP

- [ ] Notebook 5. SIR benchmark
    - [ ] Notebook 5. Potentially comparisont to SHAP
    - [ ] Notebook 5. Potentially comparison to cSHAP

- [ ] Notebook 6. Harder example (?)


- [ ] The sufficiency component of PCI is never validated. The rebuttal's distinction from causal Shapley rests on the sufficiency component : that PCI measures "how a feature contributes to a unit's prediction being close to factual value," not just the counterfactual effect of changing it (necessity, ). But sufficiency is precisely the component the paper never validates. 

- [ ] we will expand the continuous-variable example currently presented in the appendix into a more tutorial-style example that illustrates the PCI construction step by step, including a parallel comparison with SHAP **repeated in writing on purpose**

- [ ] The -excised distribution is not motivated over the observational distribution. At , the excised distribution reduces to the observational distribution ( is empty), which is what SHAP uses as its baseline. The rebuttal motivates excision intuitively but provides no formal justification for or guidance for choosing . Like , this is a free parameter that changes scores, with no selection criterion. **repeated in writing on purpose**



### WRITING


#### Search parameters

- [ ] The paper uses K=4, but point 3b remains unaddressed: K is a free parameter that changes attribution scores, with no selection criterion offered. The rebuttal's analogy to interaction terms in regression does not resolve this, as regression provides model selection criteria for choosing interaction order.


- [ ] convert the method description in Section 4 into a more formalized algorithm

- [ ] *In a sufficiently interesting model with many variables, sampling interventions uniformly from the powerset would have the consequence of the expected size of intervention set being \(d/2\), where \(d\) is the number of variables. That would mean, say, that to judge the role of a single feature in a model with 50 variables, one would be looking mostly at interventions involving 25 variables and trying to disentangle the role of that feature. In a realistic model the noise involved will not allow one to do this reliably. This would, moreover, not be conceptually the right thing to do. Imagine a model with 50 variables each corresponding to a particular voter (out of a group of 50) voting "yes"/"no". Suppose the total outcome is "yes" and we want to inspect the role of a particular voter. Sampling uniformly from the powerset, the expected size of alternatives intervened on would be 25, so this procedure would force us to bias the number of voters changing their mind in the alternative worlds we consider to be close to 25. Why use sets at all instead of single witnesses and single interventions? This is analogous to model-building choices: many models go beyond "single feature roles" by allowing pairwise interaction terms. Why only pairwise? Because it's hard to add all possible \(n\)-feature interaction terms. In allowing for up to \(n>2\) terms, we generalize slightly, allowing for up to four-term interactions, but limiting considerations are analogous.*

_____





- [ ]  the proposed approach appears to require the complete causal model (structural equations and noise distributions) to sample counterfactual outcomes. This is strictly stronger than the requirements of the approaches PCI claims to generalize: PN/PS/PNS, despite also being counterfactual quantities, can be bounded from observational or interventional data given a causal graph, and causal Shapley methods require observational data and a graph or interventional data. 


#### Comparison to SHAP

- [ ] However, Example 1 contains Shapley weighting as a special case: at , the two are mathematically equivalent (Eq. 8 of Lundberg et al., 2017). SHAP or Shapley weights are not credited in the paper, and this is the same combinatorial averaging the rebuttal claims PCI avoids. 


- [ ] *Referring to papers cited by the referee, Janzing et al. argue that conditioning is conceptually wrong for feature removal, and that dropping a feature should be interpreted causally as an intervention. Sharma et al. propose CF-Shapley to attribute a change in a system metric from a reference $x^\text{ref}$ to an observed $x$, using a structural causal model. For input with observed $x$ and reference $x^\text{ref}$, the CF-Shapley value is the average counterfactual effect of setting $x_j$ to its reference value across all subsets of other variables:
$$
\phi_j = \frac{1}{|S|} \sum_{S \subseteq \{1,\dots,d\}\setminus \{j\}} \big( f(x_S, x_j^\text{ref}) - f(x_S, x_j) \big).
$$
A similar estimand is introduced by Heskes et al., except it's defined in terms of permutations.
We agree that introducing interventional semantics is an improvement, but there are key conceptual differences between this approach and PCI: **Different questions.** SHAP (and causal SHAP) measures a feature’s contribution relative to a group baseline. PCI measures how a feature contributes to a unit’s prediction being close to the factual value and how changing it would lead that prediction away from it. For a nearly average observation where $y \approx \bar{y}$, SHAP attempts to explain near-zero values, while PCI asks which features keep the prediction near $y$ and which would move it away. The only "baseline" in PCI is the local factual value, which makes the attribution even more focused on token-level causality. **Computation.** SHAP requires combinatorial averaging over permutations/subsets. PCI estimands are expectations over probabilistic models, enabling standard inference methods. **Mediation-like witness blocking.** PCI uses mechanisms inspired by overdetermination and undercutting to break symmetries (as illustrated in benchmarks) and yield finer responsibility attributions. Causal SHAP lacks these. **Connection.** Our conjecture is that with null witness proposals, empirical distributions, uniform sampling over candidate sets, and a baseline-distance metric, PCI can in fact be used to approximate estimands closely related to causal SHAP values. We will add an explicit discussion of these points and of Causal SHAP more generally to the paper.*



- [ ] The paper defines PCI with a general function incorporating both sufficiency and necessity, claiming to generalize PNS. Yet Theorems 10 and 11 require 𝟙, which ignores entirely. Section 4 also explicitly drops sufficiency. 

#### Theorem conditions



    *We apologize for any lack of clarity in the statement of the theorems. They require that the alternative values of intervention candidates have full support, that is, that any non-factual value from the range considered as reasonable alternative interventions has a non-zero probability/density. In the stone-throwing example, this means that the probability of Sally not throwing and that the probability of Bill not throwing should be non-zero. These conditions are satisfied in the stochastic model under consideration. What is not required by the theorems is that the outcome variable conditional on each particular interventional setting should have full support. This will obviously be false in any deterministic SCM, but it is not required by the theorems.*

We will inspect the wording of the theorems and the discussion thereof, ensuring that it doesn't suggest this misinterpretation.

- [ ]  The theorem conditions still appear to be violated on the benchmark. The rebuttal clarifies that Theorems 10–11 require full support on non-factive values of intervention candidates, not on the outcome. However, suspects and witnesses are defined as potentially overlapping subsets of all endogenous variables, without excluding deterministic intermediate variables. In the stone-throwing benchmark, let $S$ = "Sally throws" and $H$ = "the bottle is hit". Then $H$ is a deterministic function of $S$ (and possibly other variables), e.g. $H = f(S, \dots)$. If both $S$ and $H$ are included in the suspect set, the joint alternative distribution over counterfactual values cannot have full support. In particular, assignments such as $(S = 0, H = 1)$ are impossible under the structural equations, so $P(S = 0, H = 1) = 0$. However, the theorem requires joint support: “full support for the alternative value distribution at $\mathbf{s}'$” (Appendix B), where $\mathbf{s}'$ is a joint assignment over the active suspect set $\mathbf{S}$. The rebuttal addresses only marginal support (e.g., “the probability of Sally not throwing should be non-zero”), but this is insufficient: the condition requires $P(\mathbf{S} = \mathbf{s}') > 0$ for all admissible $\mathbf{s}'$, not merely that each component has non-zero marginal probability.


#### Excision and normality

*The core insight from the probabilities of causal necessity and sufficiency is that responsibility attributions have two components: how much causal power does a property have to make things happen the way they did, and how much causal power does a property have to make things different if we intervene for it to have a different value. Conceptually this is a generalization of the common pattern in causal effect estimands: one always compares treatment to some baseline different from the treatment. The generalization here is that instead of comparing to a single sometimes somewhat arbitrarily chosen baseline, in a continuous case we integrate over alternative values together given an informed posterior over them. To build up intuition, consider the limiting case of a binary variable. For instance, in asking whether a person's health outcome is impacted by a given drug dosage, the intuition is that we should compare this to regimes in which we excise, i.e. do not give that drug to the subject. To investigate the sufficiency of the drug, we separately investigate the consequence of giving the drug to the subject.
Some need to choose a meaningful alternative value already comes up as soon as one wants to distinguish treatments in a more interesting non-binary space: a decision about what a meaningfully different treatment is has to be tacitly made anyway - our framework forces the user to be more explicit about it.
*


- [ ] Why not just use the empirical distribution?

    *Here the answer is widely Bayesian: whatever reasons one might have to form the posteriors using priors and conditioning on data, are the reasons why one should be using a posterior distribution instead of empirical distribution. A good guide for reasons to go this way can be found, for example, in Kruschke's book on Bayesian data analysis.*

- [ ] Why only use information upstream in shaping alternative distributions?

    *The base distributions for a given site are only informed by upstream variable values. This means that the exploratory search for alternative values is relatively wide but not unrealistic. We build in the assumption that the range of realistic alternative values for a feature is constrained by other factual features that have impact on it, but not by features below it in the causal diagram. Some alternatives to this, which can also be implemented are:*


- [ ] The -excised distribution is not motivated over the observational distribution. At , the excised distribution reduces to the observational distribution ( is empty), which is what SHAP uses as its baseline. The rebuttal motivates excision intuitively but provides no formal justification for or guidance for choosing . Like , this is a free parameter that changes scores, with no selection criterion. **repeated in code on purpose**

----------


- [ ] Normality references were withdrawn rather than supported. The paper's introduction frames PCI as "inspired by the study of descriptive normality (Halpern and Hitchcock, 2015; Icard et al., 2017)." The rebuttal states these will be dropped because "discussion is too involved for this paper." This withdraws a motivating claim without replacing it.

- [ ] we will expand the continuous-variable example currently presented in the appendix into a more tutorial-style example that illustrates the PCI construction step by step, including a parallel comparison with SHAP **repeated in code on purpose**



#### Causal impact function

- [ ] *The metric we use here is a generalization of standard estimands that rely on L1 distance. In a whole array of causal effect literature, L1 distance between outcomes is an outcome of interest (e.g. in Average Treatment Effect, Conditional Treatment Effect, L1 distance is also used in SHAP). While we depart from those approaches by moving away from having a single baseline value and integrating over the empirically informed conditional distribution of alternative values, we preserve the spirit thereof by still using L1, except now in two dimensions. This being said, if the users have a reason to conceptualize treatment effect using L2 or some other distance measure, this is fully composable and they are free to do so. We will clarify this further in the text.*




#### Cardinality-constrained search




*In a sufficiently interesting model with many variables, sampling interventions uniformly from the powerset implies that the expected size of the intervention set is approximately \( n/2 \), where \( n \) is the number of variables. This has an important consequence: to judge the role of a single feature in a model with, say, 50 variables, one would mostly be examining interventions involving around 25 variables. The task then becomes disentangling the role of a single variable from large, high-order interventions. In realistic settings, noise will generally make this unreliable. Moreover, this is not conceptually the right approach. Consider a model with 50 variables, each representing a voter choosing "yes" or "no". Suppose the aggregate outcome is "yes", and we want to evaluate the role of a specific voter. If we sample uniformly from the powerset, the expected number of voters whose choices are altered in counterfactual scenarios is 25. This artificially biases the analysis toward large coalitions of changes, rather than isolating the contribution of individual voters. This raises a more general question: why use sets of interventions at all, instead of focusing on single interventions (or minimal witnesses)? There is a parallel with model-building. Many models extend beyond single-feature effects by including interaction terms. Typically, these are limited to pairwise interactions—not because higher-order interactions are conceptually invalid, but because including all \( n \)-way interactions is computationally and statistically intractable. Allowing interactions up to order \( n > 2 \) (e.g., up to four-way interactions) is a modest generalization, but the same limiting considerations apply.*


### MINOR EDITORIAL

- [ ] SHAP or Shapley weights are not credited in the paper