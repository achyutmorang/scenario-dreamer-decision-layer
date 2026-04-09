# Scenario Dreamer Decision-Layer Research Plan

## Motivation
The earlier SMART/WOSAC line established an important fact: the infrastructure problem was largely solved, but full backbone training on Colab-class GPUs is not operationally viable. The pivot here is deliberate: keep the scientific objective of improving **closed-loop multi-agent behavior**, but move to a stack with a usable pretrained baseline so GPU time is spent on methodology rather than reproducing a large model from scratch.

## Baseline Definition
The fixed v1 baseline is:
- **Simulator:** Scenario Dreamer
- **Behavior model:** pretrained CtRL-Sim checkpoint (`ctrl_sim_waymo_1M_steps`)
- **Baseline policy:** IDM ego planner with CtRL-Sim-controlled traffic agents
- **Evaluation domains:** Scenario Dreamer pregenerated Waymo environments, organized into smoke/dev/report tiers

Baseline semantics for this repo:
- **System baseline:** the fixed stack above.
- **Experiment baseline:** the same frozen stack with no added decision layer, reranker, critic, or other method-specific logic.

For perturbation or interaction-breakdown studies, keep the distinction explicit:
- the **reference rollout** is the stock frozen stack on the original scene,
- the **counterfactual baseline** is the same stock frozen stack on the perturbed scene,
- the **method** is the perturbed scene plus the proposed add-on.

What stays frozen in v1:
- Scenario Dreamer environment generation stack
- CtRL-Sim behavior backbone
- baseline environment pack, seeds, and rollout settings

What is allowed to change later:
- a lightweight decision layer on top of the frozen baseline
- candidate generation / reranking logic
- small-head risk/progress scoring modules
- transfer-check adapters into Waymax/WOSAC-style interfaces

## First Method Trajectory
The first recommended method is **risk-aware decision-layer selection**.

Core idea:
1. query the frozen baseline for `K` candidate short-horizon actions or rollout continuations
2. compute a small set of selection features such as collision risk, route progress, smoothness, and model confidence
3. use a lightweight selector or reranker to choose the candidate actually executed in closed loop

This is intentionally chosen because it is tractable on limited GPUs. It avoids retraining the expensive generator and focuses research effort on the part of the system where a small method can still produce measurable closed-loop gains.

## Baseline-to-Method Comparison Protocol
All future methods must keep these fixed:
- same frozen CtRL-Sim checkpoint
- same environment pack and subset definitions
- same random seeds for each evaluation tier
- same ego planner family unless the experiment explicitly changes it
- same rollout horizon and metric schema

Default comparison rule:
- if the contribution is a **decision layer**, compare against the same frozen stack without the decision layer.
- if the contribution is a **diagnostic perturbation study**, compare stock-vs-stock first before introducing any method.

The normalized outputs in `results/runs/<run_id>/` are the contract for comparison.

## Evaluation Strategy
### Smoke
- 1-3 environments
- visualization-first
- goal: confirm end-to-end setup, video path, and artifact schema

### Dev
- fixed small subset from the pregenerated Waymo environment pack
- goal: iteration on analysis and later on decision-layer prototypes

### Report
- full available pregenerated pack in the repo baseline config
- fixed seeds
- goal: stable baseline numbers, representative videos, and future comparative analysis

Tracked metrics:
- collision rate
- off route rate
- completed rate
- progress
- runtime throughput
- scenario count and elapsed time

## Near-Term Experimental Plan
The near-term question is:

> Can a **distribution-aware, risk-constrained ego decision rule** improve closed-loop safety and progress over the stock frozen baseline under equal compute budget?

The immediate execution ladder is:

1. **Smoke baseline validation**
   - rerun the stock frozen baseline on 3 scenarios
   - confirm Drive-backed persistence for `run_manifest.json`, `metrics.json`, `config_snapshot.json`, and at least one MP4
2. **Experiment 0: diversity audit**
   - hold one fixed scenario constant
   - rerun the same stock ego policy under multiple seeds
   - measure whether closed-loop outcome metrics vary across seeds
   - if no metric-level variation appears, do not rush into a distributional selector; first add trajectory-level audit instrumentation
3. **Risk-variance study**
   - expand from one scenario to a fixed small scene set
   - quantify per-scene variance in short-horizon risk proxies such as minimum TTC and minimum ego-agent distance
   - measure an offline selector upper bound by choosing the safest sampled future among the first $K$ rollouts
   - use this to decide whether multi-sample selection has enough signal to justify a stronger online selector
4. **Dev-tier baseline lock**
   - run the fixed 12-scenario dev subset
   - record the stock baseline metrics and runtime envelope
   - use this as the default comparator for early selector work
5. **Selector baseline ladder**
   - once candidate generation exists, compare:
     - stock IDM ego
     - single-rollout selection
     - mean-based selection
     - worst-case selection
     - CVaR-constrained selection
6. **Report-tier evaluation**
   - only after the selector is stable on the dev subset
   - preserve seeds, scenario pack, and metric schema

Decision gate:
- if Experiment 0 shows no meaningful future diversity, the project should shift toward deterministic or robust replanning rather than tail-risk statistics.
- if the risk-variance study shows no consistent improvement even for the offline selector upper bound, do not invest in a more expensive online selector yet.

## Phased Milestones
1. **Baseline bootstrap**
   - pin upstream Scenario Dreamer
   - verify pretrained CtRL-Sim checkpoint path
   - verify pregenerated environment pack path
2. **Smoke evaluation + video**
   - run 1-3 environments
   - emit normalized artifacts
   - save at least one demo video
3. **Normalized report evaluation**
   - run the full report tier
   - lock baseline metrics and manifests
4. **Decision-layer prototype**
   - add candidate selection without changing the backbone
5. **Transfer-check hook**
   - map baseline outputs into a later Waymax/WOSAC comparison scaffold

## Stop Conditions
- no backbone retraining in v1
- no full WOSAC reproduction inside this repo
- no large literature expansion beyond the curated core set
- no new simulator stack inside this repo unless Scenario Dreamer is abandoned entirely

## Why This Direction Is Feasible
This trajectory matches the actual constraint surface:
- limited Colab GPU time
- pretrained baseline available in Scenario Dreamer
- strong need for a method that can be evaluated quickly and iterated repeatedly

The scientific bet is that **selection quality**, not full backbone retraining, is the highest-leverage innovation target under the current compute budget.
