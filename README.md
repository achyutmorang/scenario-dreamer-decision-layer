# Scenario Dreamer Decision Layer

A local research workspace for **Scenario Dreamer + pretrained CtRL-Sim** baseline experiments, with a clean path toward a future decision-layer method (risk-aware reranking / selection) and a minimal Waymax/WOSAC transfer hook.

## What This Repo Is
- A **baseline bootstrap** workspace, not a backbone training repo.
- The baseline is fixed to:
  - **Simulator stack:** Scenario Dreamer
  - **Behavior model:** pretrained CtRL-Sim checkpoint
  - **Planner policy for baseline eval:** IDM ego policy, CtRL-Sim for other agents
- The research target is a future **decision-layer method** on top of a frozen baseline.

## Baseline Semantics
This repo uses the word **baseline** in two distinct but related ways:

- **System baseline:** the fixed stack used across the project:
  - Scenario Dreamer pregenerated Waymo environments
  - frozen pretrained CtRL-Sim traffic model
  - stock IDM ego policy
- **Experiment baseline:** the exact same frozen stack **without** any new method-specific logic added on top.

For example, if the experiment is an interaction-breakdown study or a decision-layer comparison, the fairest comparator is:

- same scene,
- same seeds,
- same frozen CtRL-Sim traffic,
- same IDM ego family,
- but **no added reranker, selector, critic, or perturbation-handling logic** beyond the stock setup.

## What This Repo Is Not
- Not a full SMART reproduction repo.
- Not a full WOSAC submission repo.
- Not a full Scenario Dreamer training repo.

## Repository Map
- [`docs/research_plan.md`](./docs/research_plan.md): research direction and milestone ladder
- [`docs/problem_formulation.md`](./docs/problem_formulation.md): rigorous problem statement for the decision-layer method
- [`docs/ctrl_sim_engineering_note.md`](./docs/ctrl_sim_engineering_note.md): CtRL-Sim architecture, training details, and Colab-scope implications
- [`docs/closed_loop_simulation_field_reverse_engineering.md`](./docs/closed_loop_simulation_field_reverse_engineering.md): critical field analysis, failure modes, and research hooks
- [`configs/baselines/scenario_dreamer_ctrlsim.yaml`](./configs/baselines/scenario_dreamer_ctrlsim.yaml): single source of truth baseline config (JSON-compatible YAML)
- [`references/papers/`](./references/papers): curated core paper packet
- [`references/papers_manifest.json`](./references/papers_manifest.json): paper metadata + integrity
- [`scripts/bootstrap_baseline.py`](./scripts/bootstrap_baseline.py): clone and pin upstream Scenario Dreamer, verify asset locations, write lock metadata
- [`scripts/download_references.py`](./scripts/download_references.py): download the curated paper set and refresh the manifest
- [`scripts/download_assets.py`](./scripts/download_assets.py): verify or optionally download baseline assets
- [`scripts/run_smoke_eval.py`](./scripts/run_smoke_eval.py): 1-3 environment smoke run
- [`scripts/run_dev_eval.py`](./scripts/run_dev_eval.py): fixed small subset development run
- [`scripts/run_report_eval.py`](./scripts/run_report_eval.py): full report-tier run
- [`scripts/run_diversity_audit.py`](./scripts/run_diversity_audit.py): Experiment 0 seed-sweep audit on one fixed scenario
- [`scripts/render_demo.py`](./scripts/render_demo.py): single-environment MP4 demo run
- [`scripts/setup_colab_runtime.py`](./scripts/setup_colab_runtime.py): lean Colab runtime bootstrap for baseline simulation
- [`notebooks/`](./notebooks): Colab notebooks for Drive-backed asset prep, smoke baseline, diversity audit, and dev/report evaluation

## Quickstart
### 1. Download papers
```bash
python3 scripts/download_references.py
python3 scripts/validate_papers.py
```

### 2. Clone and pin upstream Scenario Dreamer
```bash
python3 scripts/bootstrap_baseline.py --clone-upstream --write-lock
```

### 3. Verify or fetch baseline assets
```bash
python3 scripts/download_assets.py --mode checkpoint --download
python3 scripts/download_assets.py --mode envs
```

Notes:
- The checkpoint download can be automated with `gdown` if it is installed.
- The 75-environment Waymo pack may require manual placement depending on the Google Drive layout exposed by the upstream release.

### 4. Dry-run the baseline
```bash
python3 scripts/run_smoke_eval.py --dry-run
```

### 5. Run the smoke baseline
```bash
python3 scripts/run_smoke_eval.py
```

### 6. Render a demo video
```bash
python3 scripts/render_demo.py
```

### 7. Run Experiment 0: diversity audit
```bash
python3 scripts/run_diversity_audit.py --scenario-index 0 --seeds 0,1,2,3
```

### 8. Run the dev tier
```bash
python3 scripts/run_dev_eval.py
```

## Colab Path
If you want a persistent Colab workflow with Google Drive-backed assets and run artifacts:
- use [`notebooks/scenario_dreamer_assets_colab.ipynb`](./notebooks/scenario_dreamer_assets_colab.ipynb) to bind the canonical repo asset layout to Drive and optionally fetch the CtRL-Sim checkpoint,
- then use [`notebooks/scenario_dreamer_baseline_colab.ipynb`](./notebooks/scenario_dreamer_baseline_colab.ipynb) to install a lean simulation runtime, run the smoke baseline, and render a demo MP4 directly into Drive-backed results storage,
- then use [`notebooks/scenario_dreamer_diversity_audit_colab.ipynb`](./notebooks/scenario_dreamer_diversity_audit_colab.ipynb) for Experiment 0,
- and use [`notebooks/scenario_dreamer_dev_eval_colab.ipynb`](./notebooks/scenario_dreamer_dev_eval_colab.ipynb) to run the fixed dev subset and, later, the report tier.

## Normalized Output Contract
Each run writes under `results/runs/<run_id>/` by default, or under `$SCENARIO_DREAMER_RESULTS_ROOT/<run_id>/` when the Drive-backed override is set:
- `run_manifest.json`
- `metrics.json`
- `config_snapshot.json`
- `stdout.log`
- `stderr.log`
- optional `video_manifest.json`

This schema is method-agnostic so later decision-layer methods can be compared against the same baseline without redesigning downstream analysis.

## Implementation Notes
- The baseline config file uses a `.yaml` extension but is stored as **JSON-compatible YAML** so it can be loaded with Python stdlib only.
- Upstream code is cloned into `external/scenario-dreamer` and is not vendored into this repo.
