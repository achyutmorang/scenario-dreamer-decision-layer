# Problem Formulation: Distributional Policy Improvement Under Multi-Agent Uncertainty

## One-Sentence Formulation
Can a **distribution-aware, tail-risk-constrained ego decision layer** improve the closed-loop safety-progress tradeoff in Scenario Dreamer environments when the environment generator and reactive traffic model are both frozen?

## Why This Problem Exists
The practical gap is not "how to train a larger driving model." The practical gap is:

- pretrained closed-loop stacks already produce plausible behavior,
- reactive traffic induces **multi-modal conditional futures** for the ego,
- selecting actions by a single predicted outcome or a single rollout can be brittle,
- compute-constrained researchers often cannot retrain the backbone, but can still modify the **decision rule**.

This makes the scientifically sharp question:

> Given a frozen generative-reactive simulator stack, what statistic of the future outcome distribution should the ego optimize at decision time: mean return, worst-case return, or tail risk?

That is a real problem. It is not classical trajectory optimization in disguise unless the future distribution collapses to a single deterministic rollout.

## Fixed Experimental Substrate
The following remain fixed throughout the main comparison:

- **Environment source**: Scenario Dreamer pregenerated Waymo environment pack
- **Traffic model**: pretrained CtRL-Sim checkpoint
- **Baseline ego planner family**: IDM-derived baseline proposal
- **Scenario subsets and seeds**: smoke/dev/report tiers already defined in this repo
- **Closed-loop rollout settings**: same horizon, simulator step size, and metric schema

Only the **ego decision layer** is allowed to change.

## Baseline Semantics
For this project, baseline should be read at two levels:

- **System baseline:** fixed Scenario Dreamer environments + frozen CtRL-Sim traffic + stock IDM ego.
- **Experiment baseline:** the same frozen system baseline with **no added selector, reranker, critic, or other method-specific logic**.

This distinction matters because the contribution here is not a new simulator backbone. It is a modification to the ego decision rule on top of a fixed closed-loop stack.

## Core State and Transition Model
Let:

- $e \in \mathcal{E}$ be a scenario from the fixed Scenario Dreamer environment set,
- $x_t \in \mathcal{X}$ be the simulator state at decision time $t$,
- $a_t \in \mathcal{A}$ be the ego action or short-horizon ego plan chosen at time $t$,
- $\pi_{\mathrm{traffic}}$ be the frozen CtRL-Sim traffic policy,
- $P_e^{\pi_{\mathrm{traffic}}}(\cdot \mid x_t, a_t, \xi)$ be the closed-loop transition law induced by environment $e$, ego choice $a_t$, frozen traffic, and traffic-model sampling noise $\xi$.

The presence of $\xi$ is essential. If the traffic stack is effectively deterministic, there is no distributional decision problem, only deterministic replanning. This yields an immediate prerequisite experiment:

> **Experiment 0:** verify that conditioned on the same $(x_t, a_t)$, the frozen simulator-policy stack produces non-trivial diversity over sampled futures.

## Candidate Generation
At each decision time $t$, construct a **bounded, structured candidate set**:

$$
A_t = \{a_t^{(0)}, a_t^{(1)}, \dots, a_t^{(M-1)}\}.
$$

Equivalently, define a fixed proposal generator

$$
\mathcal{G}(x_t; \Theta) = \{a_t(\theta_0), \dots, a_t(\theta_{M-1})\},
$$

where $\Theta = \{\theta_0, \dots, \theta_{M-1}\}$ is a small, fixed family of ego-behavior parameters.

This set must be concrete, cheap, and reproducible. For the first method family in this repo:

- $a_t^{(0)}$: baseline nominal IDM-style ego action or nominal short-horizon ego rollout,
- $a_t^{(1)}$: conservative longitudinal variant with larger headway or lower target speed,
- $a_t^{(2)}$: aggressive longitudinal variant with smaller headway or higher target speed,
- $a_t^{(3)}, a_t^{(4)}$: optional route- or lane-biased variants when the substrate exposes distinct feasible lane-following continuations.

The exact parameterization can change with implementation, but the proposal family must remain fixed across all compared selectors. The method is not allowed to assume access to a perfect planner or full-horizon optimizer.

## Multi-Future Evaluation Per Candidate
For each candidate $a_t^{(m)} \in A_t$, draw $K$ futures over a short receding horizon $H$:

$$
\tau_{t:t+H}^{(m,1)}, \dots, \tau_{t:t+H}^{(m,K)}
\sim
P_e^{\pi_{\mathrm{traffic}}}(\cdot \mid x_t, a_t^{(m)}, \xi).
$$

Each future is a closed-loop rollout in which:

- the first ego decision is fixed by candidate $a_t^{(m)}$,
- traffic remains reactive under frozen CtRL-Sim,
- subsequent evolution is simulated under the fixed substrate.

This is where Scenario Dreamer matters: it supplies a controlled but diverse evaluation distribution of scenes, and CtRL-Sim supplies reactive traffic dynamics so that conditional futures are not trivial.

## Short-Horizon Utility and Cost Functionals
For each sampled future $\tau_{t:t+H}^{(m,k)}$, define:

- progress utility:

$$
R^{(m,k)}_H
=
\sum_{h=0}^{H-1} \gamma^h \, r_{\mathrm{prog}}(x_{t+h}, a_{t+h}),
$$

- short-horizon proximity risk surrogate:

$$
d_{\min,h}^{(m,k)}
=
\min_{j \in \mathcal{N}_{t+h}}
\mathrm{dist}\!\left(x_{t+h}^{\mathrm{ego}}, x_{t+h}^{(j)}\right),
$$

$$
C^{(m,k)}_{\mathrm{prox}}
=
\sum_{h=0}^{H-1}
\frac{1}{d_{\min,h}^{(m,k)} + \varepsilon_d},
$$

- route-deviation surrogate:

$$
C^{(m,k)}_{\mathrm{route}}
=
\max_{0 \le h < H} \mathrm{dist}_{\mathrm{route}}\!\left(x_{t+h}^{\mathrm{ego}}\right),
$$

- optional interaction-disturbance cost:

$$
C^{(m,k)}_{\mathrm{int}}
=
\sum_{h=0}^{H-1} \sum_{j \in \mathcal{N}_{t+h}}
\left\|
x_{t+h}^{(j)}\!\left(a_t^{(m)}\right)
-
x_{t+h}^{(j)}\!\left(a_t^{(0)}\right)
\right\|_2,
$$

which measures how strongly the ego candidate perturbs the traffic response relative to the nominal baseline candidate.

The first two costs are the core ones. $C_{\mathrm{int}}$ is a secondary regularizer, not a mandatory primary contribution.

### Report-Time Binary Metrics
The final paper must still report the substrate-native binary episode metrics:

- collision indicator,
- off-route indicator,
- completion indicator,
- progress.

The key distinction is:

- **decision-time risk variables** are continuous short-horizon surrogates,
- **report-time safety metrics** remain binary episode outcomes.

This separation is necessary because CVaR over binary collision indicators is too weak to encode severity.

## Distributional Statistics
For each candidate $a_t^{(m)}$, estimate:

- expected progress:

$$
\hat{\mu}_{\mathrm{prog}}^{(m)}
=
\frac{1}{K} \sum_{k=1}^K R_H^{(m,k)},
$$

- tail proximity risk:

$$
\widehat{\mathrm{CVaR}}_{\alpha}^{(m,\mathrm{prox})}
=
\widehat{\mathrm{CVaR}}_{\alpha}\big(C_{\mathrm{prox}}^{(m,1)}, \dots, C_{\mathrm{prox}}^{(m,K)}\big),
$$

- tail route-deviation risk:

$$
\widehat{\mathrm{CVaR}}_{\alpha}^{(m,\mathrm{route})}
=
\widehat{\mathrm{CVaR}}_{\alpha}\big(C_{\mathrm{route}}^{(m,1)}, \dots, C_{\mathrm{route}}^{(m,K)}\big).
$$

If interaction disturbance is included, also estimate

$$
\hat{\mu}_{\mathrm{int}}^{(m)}
=
\frac{1}{K} \sum_{k=1}^{K} C_{\mathrm{int}}^{(m,k)}.
$$

Using CVaR rather than the mean is the key shift. The method is not merely "scoring actions." It is comparing **distributions of closed-loop futures**.

## Decision Rule
The ego selector solves, at each decision time $t$:

$$
m_t^*
=
\arg\max_{m \in \{0,\dots,M-1\}}
\hat{\mu}_{\mathrm{prog}}^{(m)} - \lambda_{\mathrm{int}} \hat{\mu}_{\mathrm{int}}^{(m)}
$$

subject to:

$$
\widehat{\mathrm{CVaR}}_{\alpha}^{(m,\mathrm{prox})} \le \varepsilon_{\mathrm{prox}},
\qquad
\widehat{\mathrm{CVaR}}_{\alpha}^{(m,\mathrm{route})} \le \varepsilon_{\mathrm{route}}.
$$

If the interaction term is omitted, set $\lambda_{\mathrm{int}} = 0$. If no candidate is feasible, the selector falls back to the baseline candidate $a_t^{(0)}$.

This is a **receding-horizon** controller:

1. build candidates at time $t$,
2. sample $K$ futures for each candidate over horizon $H$,
3. choose $a_t^{(m_t^*)}$,
4. execute only the first control,
5. replan at time $t+1$.

That makes the method computationally feasible on Colab and consistent with online decision making.

## Horizon Validity Assumption
The method assumes that short-horizon risk surrogates correlate with longer-horizon closed-loop outcomes. This is not automatic and must be validated.

The minimum validation is a horizon ablation:

- $H = 1$ second,
- $H = 3$ seconds,
- $H = 5$ seconds,

and a check that improvements in $C_{\mathrm{prox}}$ and $C_{\mathrm{route}}$ predict improvements in final collision/off-route rates.

## Learning Problem
There are two viable implementation levels:

### Level 1: Pure simulation-time selector
No additional learned model. Compute empirical mean/tail-risk directly from sampled futures and choose a feasible candidate.

### Level 2: Learned surrogate selector
Train a small scorer $s_\phi(x_t, a_t)$ or pair of heads
$g_\phi^{\mathrm{risk}}(x_t, a_t)$, $h_\phi^{\mathrm{prog}}(x_t, a_t)$
to predict the statistics above from cached sampled futures.

The learned version is still tractable because:

- the backbone remains frozen,
- data generation uses the fixed simulator-policy stack,
- only a small head or reranker is trained.

## Main Scientific Claim
The strong claim is not:

> constrained planning improves driving

That is too generic.

The stronger and cleaner claim is:

> Under a frozen generative-reactive driving stack, **distributional policy improvement via tail-risk-aware action selection over multiple simulated futures** can improve closed-loop safety without disproportionate loss in route progress.

This makes the paper about **which statistic of future uncertainty should guide a safe ego policy**, not about building a larger model.

## What Is Actually Novel Here
The novelty is the combination of:

1. **frozen pretrained reactive traffic** rather than retrained traffic,
2. **distribution-aware candidate evaluation** rather than single-rollout scoring,
3. **tail-risk-constrained selection over continuous short-horizon risk surrogates** rather than pure expected-value maximization,
4. **evaluation on generated long-tail environments** rather than only logged replay or deterministic maps.

Any one of these alone is weak. The combination is the actual contribution.

## Minimal Hypotheses
The first paper version should test:

### H0
Conditioned on the same state-action pair, the frozen substrate exhibits non-trivial future diversity; otherwise the distributional formulation collapses.

### H1
Single-rollout or mean-only selectors improve nominal progress but are brittle under rare adverse futures.

### H2
CVaR-constrained selectors reduce collision and off-route rates relative to mean-based selection at comparable or mildly reduced progress.

### H3
The benefit of distribution-aware selection is largest in highly interactive or branching scenarios, not in trivial scenes.

### H4
Increasing the future-sample budget $K$ improves safety only when the additional sampled futures carry non-trivial diversity.

## Required Baselines and Ablations
At minimum, compare:

0. **Diversity audit**
   - measure variance across sampled futures for fixed $(x_t, a_t)$
   - report whether multi-future reasoning is empirically justified

1. **Baseline IDM ego**
   - the stock frozen system baseline
   - fixed Scenario Dreamer scene, frozen CtRL-Sim traffic, stock IDM ego
   - no distribution-aware selection or add-on logic

2. **Single-rollout selector**
   - choose action using one sampled future per candidate

3. **Mean-only selector**
   - maximize $\hat{\mu}_{\mathrm{prog}}^{(m)}$ without tail-risk constraints

4. **Worst-case selector**
   - optimize against max or worst sampled cost

5. **CVaR-constrained selector**
   - the proposed method

Also include:

- **horizon ablation** over $H$,
- **future-budget ablation** over $K$,
- **scenario stratification** by interaction intensity, such as local traffic density, route branching, or intersection-like geometry when available.

This isolates the central question:

> Is mean, worst-case, or tail-risk the right statistic for closed-loop decision making under reactive uncertainty?

For perturbation-based diagnostics, use the same logic:

1. **Reference rollout**
   - stock frozen system baseline on the original scene
2. **Counterfactual baseline**
   - stock frozen system baseline on the perturbed scene
3. **Method rollout**
   - perturbed scene plus the proposed decision-layer or diagnostic add-on

This prevents the perturbation itself from being confused with the method effect.

## Near-Term Evaluation Plan
Before any selector is implemented, the evaluation plan is:

1. **Smoke validation**
   - verify the frozen baseline runs end to end
   - persist artifacts and videos to Drive-backed storage
2. **Experiment 0: diversity audit**
   - choose one fixed scenario index
   - rerun the same stock baseline under multiple seeds
   - estimate whether the frozen stack exhibits non-trivial outcome spread for identical high-level conditions
3. **Dev-tier baseline**
   - evaluate the stock baseline on the fixed dev subset
   - lock the baseline metrics and runtime envelope that future selectors must beat
4. **Selector comparisons**
   - only after diversity is empirically justified
   - compare single-rollout, mean, worst-case, and CVaR-based decision rules under equal future budget and equal candidate budget
5. **Report-tier run**
   - only after the selector is stable on dev

This sequence matters because a distribution-aware selector is only scientifically justified if the frozen substrate actually branches in a measurable way.

## Evaluation Contract
To keep the claim clean, the following must remain fixed across all methods:

- same CtRL-Sim checkpoint,
- same Scenario Dreamer environment pack,
- same scenario tiers and seeds,
- same candidate budget $M$,
- same future sample budget $K$,
- same horizon $H$,
- same metrics.

Primary metrics:

- collision rate,
- off-route rate,
- completed rate,
- progress,
- runtime throughput.

Secondary metrics:

- short-horizon proximity risk,
- route-deviation risk,
- optional interaction disturbance.

## Statistical Reporting
Use paired comparisons at the scenario level.

For each scenario $e$, record the metric difference:

$$
\Delta_e = \mathrm{metric}_e(\text{method}) - \mathrm{metric}_e(\text{baseline}).
$$

Then report:

- mean paired difference,
- bootstrap confidence interval over environments,
- win/loss counts per scenario class if available.

This is stronger than only reporting global averages.

## Compute-Aware Complexity
If there are $M$ candidates, $K$ sampled futures per candidate, and short horizon $H$, then the online decision cost is:

$$
\mathcal{O}(M K H)
$$

simulator steps per decision.

That is expensive enough to matter, but still feasible if:

- $M$ is small,
- $K$ is small,
- $H$ is short,
- the report tier is run offline rather than interactively.

This matches the Colab constraint surface much better than backbone retraining.

## Failure Boundaries
This formulation breaks down if any of the following are true:

1. the traffic model is effectively deterministic, so multi-future uncertainty is absent,
2. the candidate set has too little structured diversity, so selection cannot matter,
3. short-horizon surrogates do not correlate with long-horizon outcomes,
4. scenario diversity is too low, so long-tail safety does not appear,
5. the runtime budget is too small to estimate tail risk reliably.

These are not minor details. They determine whether the problem is genuinely distributional or only nominal.

## Final Problem Statement
The tight version to carry into the paper is:

> We study **distribution-aware, risk-constrained ego action selection** in a frozen generative-reactive driving stack. At each decision step, the ego evaluates a bounded candidate set using multiple sampled short-horizon futures under frozen CtRL-Sim traffic within Scenario Dreamer environments, and selects the action that maximizes expected progress subject to tail-risk constraints on collision and off-route events. The central scientific question is whether tail-risk-aware selection over future distributions yields better closed-loop safety-progress tradeoffs than deterministic or mean-based selection under fixed compute and fixed pretrained backbones.
> We study **distributional policy improvement for ego decision making under multi-agent uncertainty** in a frozen generative-reactive driving stack. At each decision step, the ego evaluates a bounded, structured candidate set using multiple sampled short-horizon futures under frozen CtRL-Sim traffic within Scenario Dreamer environments, and selects the action that maximizes expected progress subject to tail-risk constraints on continuous short-horizon safety surrogates. The central scientific question is whether tail-risk-aware selection over future distributions yields better closed-loop safety-progress tradeoffs than deterministic, single-rollout, mean-based, or worst-case selection under fixed compute and fixed pretrained backbones.

## Reviewer-Proof Summary
This is **not**:

- full-policy retraining,
- generic constrained planning,
- open-loop reranking,
- a simulator benchmark paper.

This **is**:

- safe policy improvement over a fixed proposal class,
- under multi-agent closed-loop uncertainty,
- with explicit tail-risk statistics,
- evaluated on a fixed generated-environment substrate.
