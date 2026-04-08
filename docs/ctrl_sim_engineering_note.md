# CtRL-Sim Engineering Note

## Purpose
This note summarizes the **CtRL-Sim architecture**, **training setup**, and the **operational implications** for this repo. The intent is not to reproduce CtRL-Sim training. The intent is to make clear which parts of the upstream system are expensive, which parts stay frozen, and why the decision-layer research direction is the right scope for Colab-class hardware.

## Canonical Source
Primary paper:
- *CtRL-Sim: Reactive and Controllable Driving Agents with Offline Reinforcement Learning*
- arXiv: [2403.19918](https://arxiv.org/abs/2403.19918)
- OpenReview PDF: [CoRL 2024 paper](https://openreview.net/pdf/f0911020539f35ff4c2c738d2fb2c8a38b429761.pdf)

## Architecture Summary
CtRL-Sim is a **multi-agent encoder-decoder Transformer** for closed-loop traffic simulation.

### Encoder
The encoder consumes the initial scene context at $t = 0$:
- per-agent initial state
- per-agent goal state
- map / lane geometry

Per-agent initial state includes:
- position
- velocity
- heading
- agent type

Per-agent goal state is represented by the logged final:
- position
- velocity
- heading

The map is encoded with a **polyline lane encoder** and transformed into lane-segment embeddings. Initial agent embeddings and map embeddings are concatenated and passed through Transformer encoder blocks.

### Decoder
The decoder is autoregressive over **agents and timesteps**. For each agent-time pair, it models tokens built from:
- state
- factorized return-to-go
- action

The decoder predicts:
- return tokens
- action tokens
- future states

This is important: CtRL-Sim is not just a reactive policy head. It also models **future state evolution**, which helps it behave like a closed-loop simulator policy rather than a pure one-step controller.

### Reward Factorization and Control
CtRL-Sim factorizes the return-to-go into three axes:
- goal-reaching
- vehicle-vehicle collision
- vehicle-road-edge collision

At inference, it uses **exponential tilting** over the predicted return distributions. This is how the model exposes controllability over behavior style without retraining the network.

## Key Model Hyperparameters
Reported in the paper appendix:
- hidden size $d = 256$
- encoder blocks $E = 2$
- decoder blocks $D = 4$
- model size: about **8.3M parameters**
- context length $H = 32$
- max controlled-agent subset size $N = 24$
- lane segments: up to **200** nearby segments
- lane polyline points per segment: **100**

## Dataset Construction and Representation
The training dataset is derived from WOMD scenes executed in a **physics-enhanced Nocturne simulator**.

Paper-reported preprocessing constraints:
- pedestrians and cyclists removed
- scenes with traffic lights removed
- actions obtained through an inverse bicycle-model-style reconstruction from state transitions
- acceleration clipped to $[-10, 10]$
- steering clipped to $[-0.7, 0.7]$ radians

Reported dataset sizes:
- train: **134,150** scenes
- validation: **9,678** scenes
- test: **2,492** scenes

Action tokenization:
- acceleration bins: **20**
- steering bins: **50**
- total action tokens: **1000**

Return tokenization:
- each return component is discretized into **350** bins

## Training Details
Paper-reported training setup:
- optimizer: **AdamW**
- learning rate: linearly decayed from **$5 \times 10^{-4}$**
- total steps: **200k**
- batch size: **64**
- goal dropout: **10%**
- supervise only trajectories of **moving agents**
- future-state prediction weighted by $\alpha = 1/100$

Loss structure:
$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{action}}
+
\mathcal{L}_{\mathrm{return}}
+
\alpha \mathcal{L}_{\mathrm{state}}.
$$

Reported training cost:
- about **20 hours on 4 NVIDIA A100 GPUs**

## Inference Details That Matter Here
CtRL-Sim can simulate scenes with more than 24 controlled agents by processing overlapping **24-agent subsets** at each timestep.

Paper-reported inference behavior:
- use the most recent **32** timesteps as context
- sample actions with temperature **1.5** in the appendix training-details section
- roll scenes in closed loop after an initial logged prefix

Two practical implications follow:
1. The model is designed for **reactive multi-agent closed-loop use**, not only open-loop prediction.
2. Reproducing training on weak hardware is unattractive even though the parameter count is modest.

## Why This Matters for This Repo
This repo should treat CtRL-Sim as a **frozen backbone**, not a training target.

That decision is justified by the paper's compute profile:
- the network is only 8.3M parameters,
- but the full stack still required substantial multi-GPU training,
- and its value comes from the learned closed-loop reactive traffic behavior, not from being easy to retrain.

For Colab-class experimentation, the feasible path is:
- freeze Scenario Dreamer environments,
- freeze the pretrained CtRL-Sim checkpoint,
- innovate only in the ego decision layer.

## Research Implication for the Decision-Layer Project
The right question for this repo is not:
- "Can we retrain CtRL-Sim better on Colab?"

The right question is:
- "Given a frozen generative-reactive stack, can we improve ego decisions by selecting among multiple candidate actions using distributional outcome statistics?"

That scope is defensible because it preserves:
- the pretrained traffic dynamics,
- the environment distribution,
- the evaluation contract,
- and the computational feasibility of running experiments on Colab.

## What Stays Frozen vs What Can Change
### Frozen
- Scenario Dreamer environment pack
- pretrained CtRL-Sim checkpoint
- baseline rollout and evaluation settings
- seeds and subset definitions for smoke/dev/report tiers

### Allowed to change
- ego candidate generator
- decision statistic over sampled futures
- selector / reranker parameters
- decision-time constraints and risk functional

## Operational Guidance
For this repo, treat CtRL-Sim as a **simulation prior**:
- do not plan around full retraining,
- do not treat parameter count alone as evidence of cheap reproducibility,
- do use it as the fixed reactive world model for baseline and future comparative experiments.

## Bottom Line
CtRL-Sim is small enough to be deployable, but not cheap enough to be a sensible Colab training target. Its architecture and training setup support exactly the pivot this repo is making: **freeze the expensive reactive simulator-policy backbone and study decision-layer improvement on top of it**.
