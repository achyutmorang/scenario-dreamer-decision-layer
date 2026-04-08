from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

from .config import load_config, project_root, resolve_repo_relative
from .artifacts import write_json


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_upstream_clone(config: Dict[str, Any], clone: bool = False) -> Dict[str, Any]:
    repo_url = config["upstream"]["repo_url"]
    repo_commit = config["upstream"]["repo_commit"]
    repo_dir = resolve_repo_relative(config["upstream"]["repo_dir"])
    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {"repo_dir": str(repo_dir), "repo_url": repo_url, "target_commit": repo_commit}

    if not repo_dir.exists():
        result["exists"] = False
        if clone:
            subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
        else:
            result["status"] = "missing"
            return result

    git_dir = repo_dir / ".git"
    if not git_dir.exists():
        raise RuntimeError(f"Existing upstream path is not a git repo: {repo_dir}")

    subprocess.run(["git", "-C", str(repo_dir), "fetch", "origin"], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "checkout", repo_commit], check=True)
    actual = subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
    result.update({"exists": True, "status": "ready", "actual_commit": actual})
    return result


def inspect_assets(config: Dict[str, Any]) -> Dict[str, Any]:
    checkpoint_path = resolve_repo_relative(config["assets"]["checkpoint"]["relative_ckpt_path"])
    env_pickles_dir = resolve_repo_relative(config["assets"]["simulation_envs"]["pickles_dir"])
    env_jsons_dir = resolve_repo_relative(config["assets"]["simulation_envs"]["jsons_dir"])

    payload: Dict[str, Any] = {
        "checkpoint": {
            "path": str(checkpoint_path),
            "exists": checkpoint_path.exists(),
            "expected_sha256": config["assets"]["checkpoint"]["expected_sha256"],
            "google_drive_url": config["assets"]["checkpoint"]["google_drive_url"],
        },
        "simulation_envs": {
            "pickles_dir": str(env_pickles_dir),
            "pickles_exists": env_pickles_dir.exists(),
            "num_pickles": len(sorted(env_pickles_dir.glob("*.pkl"))) if env_pickles_dir.exists() else 0,
            "jsons_dir": str(env_jsons_dir),
            "jsons_exists": env_jsons_dir.exists(),
            "num_jsons": len(sorted(env_jsons_dir.glob("*.json"))) if env_jsons_dir.exists() else 0,
            "source_url": config["assets"]["simulation_envs"]["shared_drive_url"],
        },
    }
    if checkpoint_path.exists():
        payload["checkpoint"]["actual_sha256"] = _sha256_file(checkpoint_path)
    return payload


def write_bootstrap_lock(config: Dict[str, Any], clone_result: Dict[str, Any], asset_state: Dict[str, Any]) -> Path:
    lock_dir = project_root() / "artifacts" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "baseline_assets.json"
    payload = {
        "baseline_name": config["baseline_name"],
        "upstream": clone_result,
        "assets": asset_state,
    }
    write_json(lock_path, payload)
    return lock_path
