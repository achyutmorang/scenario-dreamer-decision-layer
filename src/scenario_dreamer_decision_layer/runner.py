from __future__ import annotations

import ast
import json
import os
import re
import shutil
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

from .artifacts import create_run_bundle, sha256_jsonable, write_json, write_manifest, write_video_manifest
from .config import load_config, project_root, resolve_repo_relative
from .contracts import BaselineIdentity, RunManifest, VideoManifest, TransferHookRequest


METRIC_PATTERNS = {
    "collision_rate": re.compile(r"collision rate:\s*([0-9.]+)"),
    "off_route_rate": re.compile(r"off route rate:\s*([0-9.]+)"),
    "completed_rate": re.compile(r"completed rate:\s*([0-9.]+)"),
    "progress": re.compile(r"progress:\s*([0-9.]+)"),
}

OUTCOME_DIVERSITY_METRICS = (
    "collision_rate",
    "off_route_rate",
    "completed_rate",
    "progress",
)

RISK_SCORE_DIRECTIONS = {
    "min_ttc_proxy_s": "max",
    "min_ego_agent_distance_m": "max",
    "hard_brake_count": "min",
}

ProgressCallback = Callable[[Dict[str, Any]], None]


def _emit_progress(progress: ProgressCallback | None, event: str, **payload: Any) -> None:
    if progress is None:
        return
    progress({"event": event, **payload})


def _resolve_common_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    upstream_dir = resolve_repo_relative(config["upstream"]["repo_dir"])
    checkpoint_path = resolve_repo_relative(config["assets"]["checkpoint"]["relative_ckpt_path"])
    env_pickles_dir = resolve_repo_relative(config["assets"]["simulation_envs"]["pickles_dir"])
    env_jsons_dir = resolve_repo_relative(config["assets"]["simulation_envs"]["jsons_dir"])
    scratch_root = resolve_repo_relative(config["assets"]["scratch_root"])
    movies_dir = resolve_repo_relative(config["assets"]["movies_dir"])
    dataset_root = resolve_repo_relative(config["assets"]["dataset_root"])
    return {
        "upstream_dir": upstream_dir,
        "checkpoint_path": checkpoint_path,
        "env_pickles_dir": env_pickles_dir,
        "env_jsons_dir": env_jsons_dir,
        "scratch_root": scratch_root,
        "movies_dir": movies_dir,
        "dataset_root": dataset_root,
    }


def _list_pickles(path: Path) -> List[Path]:
    return sorted(path.glob("*.pkl"))


def _select_pickle_files(source_dir: Path, limit: int, *, offset: int = 0, indices: Iterable[int] | None = None) -> List[Path]:
    files = _list_pickles(source_dir)
    if indices is not None:
        selected: List[Path] = []
        for idx in indices:
            if idx < 0 or idx >= len(files):
                raise IndexError(f"Scenario index out of range: {idx} (available: 0..{len(files) - 1})")
            selected.append(files[idx])
        return selected
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")
    if limit < 0:
        return files[offset:]
    return files[offset : offset + limit]


def _materialize_files(selected: List[Path], subset_dir: Path) -> Tuple[Path, int]:
    subset_dir.mkdir(parents=True, exist_ok=True)
    for item in subset_dir.iterdir():
        if item.is_symlink() or item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    for item in selected:
        target = subset_dir / item.name
        target.symlink_to(item)
    return subset_dir, len(selected)


def _materialize_subset(source_dir: Path, subset_dir: Path, limit: int, *, offset: int = 0, indices: Iterable[int] | None = None) -> Tuple[Path, int]:
    selected = _select_pickle_files(source_dir, limit, offset=offset, indices=indices)
    return _materialize_files(selected, subset_dir)


def _build_command(config: Dict[str, Any], tier: str, dataset_path: Path, movie_dir: Path, visualize: bool, lightweight: bool, seed: int) -> List[str]:
    paths = _resolve_common_paths(config)
    baseline = config["baseline"]
    cmd = [
        "python",
        "run_simulation.py",
        f"sim.mode={baseline['sim_mode']}",
        f"sim.dataset_path={dataset_path}",
        f"sim.behaviour_model.run_name={baseline['behaviour_model_run_name']}",
        f"sim.behaviour_model.model_path={paths['checkpoint_path']}",
        f"sim.visualize={str(visualize)}",
        f"sim.lightweight={str(lightweight)}",
        f"sim.verbose={str(baseline['verbose'])}",
        f"sim.simulate_vehicles_only={str(baseline['simulate_vehicles_only'])}",
        f"sim.policy={baseline['policy']}",
        f"sim.steps={baseline['steps']}",
        f"sim.dt={baseline['dt']}",
        f"sim.agent_scale={baseline['agent_scale']}",
        f"sim.behaviour_model.tilt={baseline['tilt']}",
        f"sim.behaviour_model.action_temperature={baseline['action_temperature']}",
        f"sim.behaviour_model.use_rtg={str(baseline['use_rtg'])}",
        f"sim.behaviour_model.predict_rtgs={str(baseline['predict_rtgs'])}",
        f"sim.behaviour_model.compute_metrics={str(baseline['compute_behaviour_metrics'])}",
        f"sim.movie_path={movie_dir}",
        f"sim.seed={seed}",
    ]
    if paths["env_jsons_dir"].exists():
        cmd.append(f"sim.json_path={paths['env_jsons_dir']}")
    return cmd


def _parse_metrics(stdout_text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, pattern in METRIC_PATTERNS.items():
        matches = pattern.findall(stdout_text)
        if matches:
            metrics[key] = float(matches[-1])
    if len(metrics) == len(METRIC_PATTERNS):
        return metrics
    for line in reversed(stdout_text.splitlines()):
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            try:
                values = ast.literal_eval(line)
            except Exception:
                continue
            for item in values:
                if not isinstance(item, str):
                    continue
                for key, pattern in METRIC_PATTERNS.items():
                    match = pattern.search(item)
                    if match:
                        metrics[key] = float(match.group(1))
            if len(metrics) == len(METRIC_PATTERNS):
                return metrics
    return metrics


def _missing_assets(paths: Dict[str, Path]) -> Dict[str, str]:
    missing: Dict[str, str] = {}
    if not paths["upstream_dir"].exists():
        missing["upstream_dir"] = str(paths["upstream_dir"])
    if not paths["checkpoint_path"].exists():
        missing["checkpoint_path"] = str(paths["checkpoint_path"])
    if not paths["env_pickles_dir"].exists():
        missing["env_pickles_dir"] = str(paths["env_pickles_dir"])
    return missing


def _simulation_env(paths: Dict[str, Path]) -> Dict[str, str]:
    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(paths["upstream_dir"])
    env["SCRATCH_ROOT"] = str(paths["scratch_root"])
    env["DATASET_ROOT"] = str(paths["dataset_root"])
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{paths['upstream_dir']}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(paths["upstream_dir"])
    )
    return env


def _execute_simulation(
    *,
    paths: Dict[str, Path],
    cmd: List[str],
    scenario_count: int,
    stdout_path: Path,
    stderr_path: Path,
) -> Tuple[Dict[str, float], float]:
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=paths["upstream_dir"],
        env=_simulation_env(paths),
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.time() - started
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Simulation failed with code {proc.returncode}. See {stderr_path}")

    metrics = _parse_metrics(proc.stdout)
    metrics["runtime_throughput_scenarios_per_sec"] = (scenario_count / elapsed) if elapsed > 0 else 0.0
    metrics["num_scenarios"] = scenario_count
    metrics["elapsed_seconds"] = elapsed
    return metrics, elapsed


def _compute_numeric_spread(metric_values: Dict[str, List[float]]) -> Dict[str, Dict[str, float | int]]:
    spread: Dict[str, Dict[str, float | int]] = {}
    for key, values in metric_values.items():
        if not values:
            continue
        spread[key] = {
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
            "unique_values": len({round(v, 10) for v in values}),
        }
    return spread


def _compute_categorical_spread(metric_values: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    spread: Dict[str, Dict[str, Any]] = {}
    for key, values in metric_values.items():
        if not values:
            continue
        unique = sorted({str(value) for value in values})
        spread[key] = {
            "unique_values": len(unique),
            "values": unique,
        }
    return spread


def _risk_value_from_run(run_payload: Dict[str, Any], risk_key: str) -> float | None:
    trajectory_summary = run_payload.get("trajectory_summary", {})
    trajectory_metrics = trajectory_summary.get("trajectory_metrics", {})
    value = trajectory_metrics.get(risk_key)
    if value is None:
        return None
    return float(value)


def _variance_summary(values: List[float]) -> Dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
            "range": None,
            "variance": None,
            "stdev": None,
        }
    if len(values) == 1:
        variance = 0.0
        stdev = 0.0
    else:
        variance = statistics.pvariance(values)
        stdev = statistics.pstdev(values)
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "range": float(max(values) - min(values)),
        "variance": float(variance),
        "stdev": float(stdev),
    }


def _selector_pick(
    runs: List[Dict[str, Any]],
    *,
    k: int,
    risk_key: str,
) -> Dict[str, Any]:
    candidates = runs[: max(1, min(k, len(runs)))]
    direction = RISK_SCORE_DIRECTIONS[risk_key]

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for item in candidates:
        value = _risk_value_from_run(item, risk_key)
        if value is None:
            continue
        scored.append((value, item))

    if not scored:
        return {
            "k": k,
            "selected_seed": None,
            "selected_risk_value": None,
            "baseline_seed": candidates[0]["seed"] if candidates else None,
            "baseline_risk_value": None,
            "risk_improvement": None,
            "outcome_changed": False,
            "selected_metrics": {},
            "baseline_metrics": candidates[0].get("metrics", {}) if candidates else {},
        }

    baseline = candidates[0]
    baseline_value = _risk_value_from_run(baseline, risk_key)
    if direction == "max":
        selected_value, selected = max(scored, key=lambda item: item[0])
        improvement = None if baseline_value is None else float(selected_value - baseline_value)
    else:
        selected_value, selected = min(scored, key=lambda item: item[0])
        improvement = None if baseline_value is None else float(baseline_value - selected_value)

    baseline_metrics = baseline.get("metrics", {})
    selected_metrics = selected.get("metrics", {})
    outcome_changed = any(
        baseline_metrics.get(key) != selected_metrics.get(key)
        for key in OUTCOME_DIVERSITY_METRICS
    )
    return {
        "k": int(k),
        "selected_seed": int(selected["seed"]),
        "selected_risk_value": float(selected_value),
        "baseline_seed": int(baseline["seed"]),
        "baseline_risk_value": None if baseline_value is None else float(baseline_value),
        "risk_improvement": improvement,
        "outcome_changed": outcome_changed,
        "selected_metrics": selected_metrics,
        "baseline_metrics": baseline_metrics,
    }


def _execute_trajectory_audit(
    *,
    paths: Dict[str, Path],
    cmd: List[str],
    stdout_path: Path,
    stderr_path: Path,
) -> Dict[str, Any]:
    worker_script = project_root() / "scripts" / "_diversity_trace_worker.py"
    payload_path = stdout_path.with_name("trajectory_payload.json")
    worker_cmd = [
        "python",
        str(worker_script),
        "--config-dir",
        str(paths["upstream_dir"] / "cfgs"),
        "--output-json",
        str(payload_path),
        *cmd[2:],
    ]
    proc = subprocess.run(
        worker_cmd,
        cwd=project_root(),
        env=_simulation_env(paths),
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Trajectory audit failed with code {proc.returncode}. See {stderr_path}")
    if not payload_path.exists():
        raise RuntimeError(
            "Trajectory audit worker exited successfully but did not produce a payload file. "
            f"See {stdout_path} and {stderr_path}"
        )
    try:
        return json.loads(payload_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Trajectory audit worker produced an invalid payload file. "
            f"See {payload_path}, {stdout_path}, and {stderr_path}"
        ) from exc


def run_tier(
    tier: str,
    dry_run: bool = False,
    force_visualize: bool | None = None,
    force_lightweight: bool | None = None,
    num_envs: int | None = None,
    seed_override: int | None = None,
    scenario_offset: int = 0,
    scenario_indices: Iterable[int] | None = None,
) -> Dict[str, Any]:
    config = load_config()
    paths = _resolve_common_paths(config)
    tier_cfg = dict(config["evaluation"]["tiers"][tier])
    seed = int(seed_override if seed_override is not None else config["evaluation"]["seeds"][tier])
    limit = int(num_envs) if num_envs is not None else int(tier_cfg["num_envs"])
    visualize = bool(force_visualize if force_visualize is not None else tier_cfg["visualize"])
    lightweight = bool(force_lightweight if force_lightweight is not None else tier_cfg["lightweight"])

    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle = create_run_bundle(tag=tag, tier=tier)
    missing_assets = _missing_assets(paths)

    if paths["env_pickles_dir"].exists():
        if limit > 0:
            dataset_path, scenario_count = _materialize_subset(
                paths["env_pickles_dir"],
                bundle["subset_dir"],
                limit,
                offset=scenario_offset,
                indices=scenario_indices,
            )
        else:
            dataset_path = paths["env_pickles_dir"]
            scenario_count = len(_list_pickles(dataset_path))
    else:
        dataset_path = paths["env_pickles_dir"]
        scenario_count = 0

    movie_dir = bundle["movie_dir"]
    movie_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_command(
        config,
        tier=tier,
        dataset_path=dataset_path,
        movie_dir=movie_dir,
        visualize=visualize,
        lightweight=lightweight,
        seed=seed,
    )
    config_snapshot = {
        "config": config,
        "tier": tier,
        "resolved": {
            "upstream_dir": str(paths["upstream_dir"]),
            "checkpoint_path": str(paths["checkpoint_path"]),
            "dataset_path": str(dataset_path),
            "movie_dir": str(movie_dir),
            "scenario_count": scenario_count,
            "visualize": visualize,
            "lightweight": lightweight,
            "seed": seed,
            "missing_assets": missing_assets,
        },
        "command": cmd,
    }
    write_json(bundle["config_snapshot"], config_snapshot)

    if dry_run:
        return {
            "run_id": bundle["run_id"],
            "tier": tier,
            "scenario_count": scenario_count,
            "missing_assets": missing_assets,
            "command": cmd,
            "config_snapshot": str(bundle["config_snapshot"]),
        }

    if missing_assets:
        raise FileNotFoundError(f"Missing required baseline assets: {missing_assets}")
    metrics, elapsed = _execute_simulation(
        paths=paths,
        cmd=cmd,
        scenario_count=scenario_count,
        stdout_path=bundle["stdout"],
        stderr_path=bundle["stderr"],
    )
    write_json(bundle["metrics"], metrics)

    baseline_identity = BaselineIdentity(
        name=config["baseline_name"],
        upstream_repo_url=config["upstream"]["repo_url"],
        upstream_repo_commit=config["upstream"]["repo_commit"],
        checkpoint_path=str(paths["checkpoint_path"]),
        environment_path=str(dataset_path),
        tier=tier,
        seed=seed,
    )
    manifest = RunManifest(
        run_id=bundle["run_id"],
        created_utc=datetime.now(timezone.utc).isoformat(),
        baseline=baseline_identity,
        command=cmd,
        config_path=str(project_root() / "configs" / "baselines" / "scenario_dreamer_ctrlsim.yaml"),
        config_hash=sha256_jsonable(config_snapshot),
        num_scenarios=scenario_count,
        elapsed_seconds=elapsed,
        stdout_log=str(bundle["stdout"]),
        stderr_log=str(bundle["stderr"]),
    )
    write_manifest(bundle["run_manifest"], manifest)

    if visualize:
        files = sorted(str(p.relative_to(bundle["run_dir"])) for p in movie_dir.rglob("*.mp4"))
        video_manifest = VideoManifest(run_id=bundle["run_id"], movie_dir=str(movie_dir), files=files, lightweight=lightweight)
        write_video_manifest(bundle["video_manifest"], video_manifest)

    return {
        "run_id": bundle["run_id"],
        "run_dir": str(bundle["run_dir"]),
        "metrics": metrics,
        "scenario_count": scenario_count,
    }


def run_diversity_audit(
    *,
    scenario_index: int = 0,
    seeds: Iterable[int] = (0, 1, 2, 3),
    visualize: bool = False,
    lightweight: bool = True,
    progress: ProgressCallback | None = None,
) -> Dict[str, Any]:
    config = load_config()
    paths = _resolve_common_paths(config)
    missing_assets = _missing_assets(paths)
    if missing_assets:
        raise FileNotFoundError(f"Missing required baseline assets: {missing_assets}")

    all_files = _list_pickles(paths["env_pickles_dir"])
    if not all_files:
        raise FileNotFoundError(f"No scenario pickle files found under {paths['env_pickles_dir']}")
    if scenario_index < 0 or scenario_index >= len(all_files):
        raise IndexError(f"Scenario index out of range: {scenario_index} (available: 0..{len(all_files) - 1})")

    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle = create_run_bundle(tag=tag, tier="diversity_audit")
    scenario_path = all_files[scenario_index]
    dataset_path, scenario_count = _materialize_files([scenario_path], bundle["subset_dir"])
    requested_seeds = [int(seed) for seed in seeds]
    movie_root = bundle["movie_dir"]
    movie_root.mkdir(parents=True, exist_ok=True)
    _emit_progress(
        progress,
        "diversity_audit_started",
        scenario_index=scenario_index,
        scenario_name=scenario_path.name,
        seed_count=len(requested_seeds),
        visualize=visualize,
        lightweight=lightweight,
    )

    run_payloads: List[Dict[str, Any]] = []
    for seed_position, seed in enumerate(requested_seeds, start=1):
        seed_dir = bundle["run_dir"] / f"seed_{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        movie_dir = movie_root / f"seed_{seed:03d}"
        movie_dir.mkdir(parents=True, exist_ok=True)
        _emit_progress(
            progress,
            "seed_started",
            scenario_index=scenario_index,
            scenario_name=scenario_path.name,
            seed=seed,
            seed_position=seed_position,
            seed_total=len(requested_seeds),
        )
        cmd = _build_command(
            config,
            tier="diversity_audit",
            dataset_path=dataset_path,
            movie_dir=movie_dir,
            visualize=visualize,
            lightweight=lightweight,
            seed=seed,
        )
        started = time.time()
        trace_payload = _execute_trajectory_audit(
            paths=paths,
            cmd=cmd,
            stdout_path=seed_dir / "stdout.log",
            stderr_path=seed_dir / "stderr.log",
        )
        elapsed = time.time() - started
        metrics = {
            **trace_payload["metrics"],
            "runtime_throughput_scenarios_per_sec": (scenario_count / elapsed) if elapsed > 0 else 0.0,
            "num_scenarios": scenario_count,
            "elapsed_seconds": elapsed,
        }
        seed_payload = {
            "seed": seed,
            "command": cmd,
            "metrics": metrics,
            "elapsed_seconds": elapsed,
            "stdout_log": str(seed_dir / "stdout.log"),
            "stderr_log": str(seed_dir / "stderr.log"),
            "movie_dir": str(movie_dir),
            "trajectory_summary": trace_payload["trajectory_summary"],
        }
        write_json(seed_dir / "metrics.json", metrics)
        write_json(seed_dir / "config_snapshot.json", {"seed": seed, "scenario_path": str(scenario_path), "command": cmd})
        write_json(seed_dir / "trajectory_summary.json", trace_payload["trajectory_summary"])
        run_payloads.append(seed_payload)
        _emit_progress(
            progress,
            "seed_completed",
            scenario_index=scenario_index,
            scenario_name=scenario_path.name,
            seed=seed,
            seed_position=seed_position,
            seed_total=len(requested_seeds),
            elapsed_seconds=elapsed,
            metrics=metrics,
            trajectory_metrics=trace_payload["trajectory_summary"].get("trajectory_metrics", {}),
            termination_reason=trace_payload["trajectory_summary"].get("trajectory_categories", {}).get("termination_reason"),
        )

    metric_keys = sorted(config["evaluation"]["metrics"])
    metric_values: Dict[str, List[float]] = {key: [] for key in metric_keys}
    for payload in run_payloads:
        for key in metric_keys:
            value = payload["metrics"].get(key)
            if value is not None:
                metric_values[key].append(float(value))

    spread = _compute_numeric_spread(metric_values)
    metric_level_diversity_detected = any(
        spread.get(metric_name, {}).get("range", 0.0) > 0 for metric_name in OUTCOME_DIVERSITY_METRICS
    )

    trajectory_metric_values: Dict[str, List[float]] = defaultdict(list)
    trajectory_category_values: Dict[str, List[str]] = defaultdict(list)
    for payload in run_payloads:
        trajectory_summary = payload.get("trajectory_summary", {})
        for key, value in trajectory_summary.get("trajectory_metrics", {}).items():
            if value is not None:
                trajectory_metric_values[key].append(float(value))
        for key, value in trajectory_summary.get("trajectory_categories", {}).items():
            if value is not None:
                trajectory_category_values[key].append(str(value))

    trajectory_metric_spread = _compute_numeric_spread(dict(trajectory_metric_values))
    trajectory_category_spread = _compute_categorical_spread(dict(trajectory_category_values))
    trajectory_level_diversity_detected = (
        any(item["range"] > 0 for item in trajectory_metric_spread.values())
        or any(item["unique_values"] > 1 for item in trajectory_category_spread.values())
    )
    diversity_detected = metric_level_diversity_detected or trajectory_level_diversity_detected

    if metric_level_diversity_detected:
        decision = "distributional_method_justified_at_metric_level"
    elif trajectory_level_diversity_detected:
        decision = "trajectory_level_diversity_detected_metric_level_flat"
    else:
        decision = "metric_level_diversity_not_detected; trajectory_level_diversity_not_detected"

    summary = {
        "run_id": bundle["run_id"],
        "run_dir": str(bundle["run_dir"]),
        "scenario_index": scenario_index,
        "scenario_name": scenario_path.name,
        "scenario_path": str(scenario_path),
        "num_scenarios": scenario_count,
        "seeds": requested_seeds,
        "visualize": visualize,
        "lightweight": lightweight,
        "runs": run_payloads,
        "metric_spread": spread,
        "trajectory_metric_spread": trajectory_metric_spread,
        "trajectory_category_spread": trajectory_category_spread,
        "metric_level_diversity_detected": metric_level_diversity_detected,
        "trajectory_level_diversity_detected": trajectory_level_diversity_detected,
        "diversity_detected": diversity_detected,
        "decision": decision,
    }
    write_json(bundle["run_dir"] / "diversity_audit_summary.json", summary)
    _emit_progress(
        progress,
        "diversity_audit_completed",
        scenario_index=scenario_index,
        scenario_name=scenario_path.name,
        decision=decision,
        metric_level_diversity_detected=metric_level_diversity_detected,
        trajectory_level_diversity_detected=trajectory_level_diversity_detected,
    )
    return summary


def run_risk_variance_study(
    *,
    scenario_indices: Iterable[int],
    seeds: Iterable[int] = tuple(range(10)),
    selector_k_values: Iterable[int] = (1, 2, 5, 10),
    risk_key: str = "min_ttc_proxy_s",
    visualize: bool = False,
    lightweight: bool = True,
    progress: ProgressCallback | None = None,
) -> Dict[str, Any]:
    if risk_key not in RISK_SCORE_DIRECTIONS:
        raise ValueError(f"Unsupported risk_key={risk_key!r}; expected one of {sorted(RISK_SCORE_DIRECTIONS)}")

    requested_scene_indices = [int(index) for index in scenario_indices]
    if not requested_scene_indices:
        raise ValueError("Expected at least one scenario index.")

    requested_seeds = [int(seed) for seed in seeds]
    if not requested_seeds:
        raise ValueError("Expected at least one seed.")

    requested_k_values = sorted({int(value) for value in selector_k_values if int(value) > 0})
    if not requested_k_values:
        raise ValueError("Expected at least one positive selector K value.")
    _emit_progress(
        progress,
        "risk_study_started",
        scenario_indices=requested_scene_indices,
        scene_count=len(requested_scene_indices),
        seeds=requested_seeds,
        seed_count=len(requested_seeds),
        selector_k_values=requested_k_values,
        risk_key=risk_key,
        visualize=visualize,
        lightweight=lightweight,
    )

    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle = create_run_bundle(tag=tag, tier="risk_variance_study")
    scene_payloads: List[Dict[str, Any]] = []
    selector_aggregate: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for scene_position, scenario_index in enumerate(requested_scene_indices, start=1):
        _emit_progress(
            progress,
            "scene_started",
            scenario_index=scenario_index,
            scene_position=scene_position,
            scene_total=len(requested_scene_indices),
            seed_count=len(requested_seeds),
        )

        def _scene_progress(payload: Dict[str, Any]) -> None:
            merged = {
                "scene_position": scene_position,
                "scene_total": len(requested_scene_indices),
                **payload,
            }
            _emit_progress(progress, merged.pop("event"), **merged)

        audit_summary = run_diversity_audit(
            scenario_index=scenario_index,
            seeds=requested_seeds,
            visualize=visualize,
            lightweight=lightweight,
            progress=_scene_progress,
        )
        runs = list(audit_summary.get("runs", []))
        risk_values = [
            value
            for value in (_risk_value_from_run(run_payload, risk_key) for run_payload in runs)
            if value is not None
        ]
        risk_variance = _variance_summary(risk_values)
        selector_probe = []
        for k in requested_k_values:
            probe = _selector_pick(runs, k=k, risk_key=risk_key)
            selector_probe.append(probe)
            selector_aggregate[int(k)].append(probe)

        _emit_progress(
            progress,
            "scene_completed",
            scenario_index=scenario_index,
            scene_position=scene_position,
            scene_total=len(requested_scene_indices),
            scenario_name=audit_summary.get("scenario_name", ""),
            decision=audit_summary.get("decision", ""),
            risk_variance=risk_variance,
            selector_probe=selector_probe,
        )

        scene_payloads.append(
            {
                "scenario_index": int(scenario_index),
                "scenario_name": audit_summary.get("scenario_name", ""),
                "audit_run_id": audit_summary.get("run_id", ""),
                "audit_run_dir": audit_summary.get("run_dir", ""),
                "metric_level_diversity_detected": bool(audit_summary.get("metric_level_diversity_detected", False)),
                "trajectory_level_diversity_detected": bool(audit_summary.get("trajectory_level_diversity_detected", False)),
                "decision": audit_summary.get("decision", ""),
                "risk_key": risk_key,
                "risk_variance": risk_variance,
                "metric_spread": audit_summary.get("metric_spread", {}),
                "trajectory_metric_spread": audit_summary.get("trajectory_metric_spread", {}),
                "trajectory_category_spread": audit_summary.get("trajectory_category_spread", {}),
                "selector_probe": selector_probe,
                "runs": runs,
            }
        )

    selector_summary = {}
    for k, probes in selector_aggregate.items():
        improvements = [float(item["risk_improvement"]) for item in probes if item.get("risk_improvement") is not None]
        selector_summary[str(k)] = {
            "num_scenes": len(probes),
            "num_improved": sum(1 for item in probes if (item.get("risk_improvement") or 0.0) > 0),
            "num_outcome_changed": sum(1 for item in probes if item.get("outcome_changed")),
            "mean_risk_improvement": float(statistics.fmean(improvements)) if improvements else None,
            "max_risk_improvement": float(max(improvements)) if improvements else None,
            "min_risk_improvement": float(min(improvements)) if improvements else None,
        }

    aggregate_risk_ranges = [
        float(scene["risk_variance"]["range"])
        for scene in scene_payloads
        if scene["risk_variance"]["range"] is not None
    ]
    summary = {
        "run_id": bundle["run_id"],
        "run_dir": str(bundle["run_dir"]),
        "scenario_indices": requested_scene_indices,
        "seeds": requested_seeds,
        "selector_k_values": requested_k_values,
        "risk_key": risk_key,
        "risk_direction": RISK_SCORE_DIRECTIONS[risk_key],
        "visualize": visualize,
        "lightweight": lightweight,
        "scene_count": len(scene_payloads),
        "scene_payloads": scene_payloads,
        "aggregate": {
            "risk_range_mean": float(statistics.fmean(aggregate_risk_ranges)) if aggregate_risk_ranges else None,
            "risk_range_max": float(max(aggregate_risk_ranges)) if aggregate_risk_ranges else None,
            "num_metric_level_diverse_scenes": sum(
                1 for scene in scene_payloads if scene["metric_level_diversity_detected"]
            ),
            "num_trajectory_level_diverse_scenes": sum(
                1 for scene in scene_payloads if scene["trajectory_level_diversity_detected"]
            ),
        },
        "selector_summary": selector_summary,
        "study_type": "offline_risk_variance_and_selector_upper_bound",
        "study_note": (
            "Selector probe is an offline upper bound over sampled futures, not an online control policy. "
            "It tests whether sampling exposes safer realized futures under the same fixed baseline stack."
        ),
    }
    write_json(
        bundle["config_snapshot"],
        {
            "scenario_indices": requested_scene_indices,
            "seeds": requested_seeds,
            "selector_k_values": requested_k_values,
            "risk_key": risk_key,
            "visualize": visualize,
            "lightweight": lightweight,
        },
    )
    write_json(bundle["run_dir"] / "risk_variance_study_summary.json", summary)
    _emit_progress(
        progress,
        "risk_study_completed",
        run_id=bundle["run_id"],
        run_dir=str(bundle["run_dir"]),
        scene_count=len(scene_payloads),
        aggregate=summary["aggregate"],
        selector_summary=selector_summary,
    )
    return summary


def write_transfer_hook_request(run_manifest_path: Path) -> Path:
    payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    request = TransferHookRequest(
        source_run_id=payload["run_id"],
        baseline_name=payload["baseline"]["name"],
        metrics_path=str(run_manifest_path.parent / "metrics.json"),
        run_manifest_path=str(run_manifest_path),
    )
    out = run_manifest_path.parent / "transfer_hook_request.json"
    out.write_text(json.dumps(request.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return out
