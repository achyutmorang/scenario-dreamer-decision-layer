from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .artifacts import create_run_bundle, sha256_jsonable, write_json, write_manifest, write_video_manifest
from .config import load_config, project_root, resolve_repo_relative
from .contracts import BaselineIdentity, RunManifest, VideoManifest, TransferHookRequest


METRIC_PATTERNS = {
    "collision_rate": re.compile(r"collision rate:\s*([0-9.]+)"),
    "off_route_rate": re.compile(r"off route rate:\s*([0-9.]+)"),
    "completed_rate": re.compile(r"completed rate:\s*([0-9.]+)"),
    "progress": re.compile(r"progress:\s*([0-9.]+)"),
}


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


def _materialize_subset(source_dir: Path, subset_dir: Path, limit: int) -> Tuple[Path, int]:
    subset_dir.mkdir(parents=True, exist_ok=True)
    files = _list_pickles(source_dir)
    selected = files if limit < 0 else files[:limit]
    for item in subset_dir.iterdir():
        if item.is_symlink() or item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    for item in selected:
        target = subset_dir / item.name
        target.symlink_to(item)
    return subset_dir, len(selected)


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


def run_tier(
    tier: str,
    dry_run: bool = False,
    force_visualize: bool | None = None,
    force_lightweight: bool | None = None,
    num_envs: int | None = None,
) -> Dict[str, Any]:
    config = load_config()
    paths = _resolve_common_paths(config)
    tier_cfg = dict(config["evaluation"]["tiers"][tier])
    seed = int(config["evaluation"]["seeds"][tier])
    limit = int(num_envs) if num_envs is not None else int(tier_cfg["num_envs"])
    visualize = bool(force_visualize if force_visualize is not None else tier_cfg["visualize"])
    lightweight = bool(force_lightweight if force_lightweight is not None else tier_cfg["lightweight"])

    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle = create_run_bundle(tag=tag, tier=tier)
    missing_assets = _missing_assets(paths)

    if paths["env_pickles_dir"].exists():
        if limit > 0:
            dataset_path, scenario_count = _materialize_subset(paths["env_pickles_dir"], bundle["subset_dir"], limit)
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

    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(paths["upstream_dir"])
    env["SCRATCH_ROOT"] = str(paths["scratch_root"])
    env["DATASET_ROOT"] = str(paths["dataset_root"])
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{paths['upstream_dir']}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(paths["upstream_dir"])
    )

    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=paths["upstream_dir"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.time() - started
    bundle["stdout"].write_text(proc.stdout, encoding="utf-8")
    bundle["stderr"].write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Simulation failed with code {proc.returncode}. See {bundle['stderr']}")

    metrics = _parse_metrics(proc.stdout)
    metrics["runtime_throughput_scenarios_per_sec"] = (scenario_count / elapsed) if elapsed > 0 else 0.0
    metrics["num_scenarios"] = scenario_count
    metrics["elapsed_seconds"] = elapsed
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
