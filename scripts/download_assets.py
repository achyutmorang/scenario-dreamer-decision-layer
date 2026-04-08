#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from _script_utils import ROOT  # noqa: F401
from scenario_dreamer_decision_layer.bootstrap import inspect_assets
from scenario_dreamer_decision_layer.config import load_config, resolve_repo_relative


def _checkpoint_expected_path(config) -> Path:
    return resolve_repo_relative(config["assets"]["checkpoint"]["relative_ckpt_path"])


def _checkpoint_search_root(config) -> Path:
    scratch_root = resolve_repo_relative(config["assets"]["scratch_root"])
    return scratch_root / "checkpoints"


def _normalize_checkpoint_layout(config):
    expected = _checkpoint_expected_path(config)
    search_root = _checkpoint_search_root(config)
    result = {
        "expected_path": str(expected),
        "search_root": str(search_root),
    }
    if expected.exists():
        result["status"] = "already_expected"
        return result
    if not search_root.exists():
        result["status"] = "search_root_missing"
        return result

    candidates = sorted(path for path in search_root.rglob("*.ckpt") if path.is_file())
    result["candidates"] = [str(path) for path in candidates]
    if not candidates:
        result["status"] = "missing"
        return result
    if len(candidates) > 1:
        result["status"] = "ambiguous"
        return result

    candidate = candidates[0]
    expected.parent.mkdir(parents=True, exist_ok=True)
    if candidate != expected:
        shutil.move(str(candidate), str(expected))
        parent = candidate.parent
        while parent != search_root and parent.exists():
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    result["status"] = "moved_to_expected"
    result["moved_from"] = str(candidate)
    return result


def _download_checkpoint(config):
    gdown = shutil.which("gdown")
    if not gdown:
        return {"status": "gdown_missing", "hint": "Install gdown or place the checkpoint manually.", "url": config['assets']['checkpoint']['google_drive_url']}
    search_root = _checkpoint_search_root(config)
    search_root.mkdir(parents=True, exist_ok=True)
    cmd = [gdown, "--folder", config["assets"]["checkpoint"]["google_drive_url"], "-O", str(search_root)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    payload = {
        "status": "ok" if proc.returncode == 0 else "failed",
        "command": cmd,
        "stdout": proc.stdout[-800:],
        "stderr": proc.stderr[-800:],
    }
    payload["normalize"] = _normalize_checkpoint_layout(config)
    payload["expected_exists_after"] = _checkpoint_expected_path(config).exists()
    return payload


def _verification_errors(payload, mode: str) -> list[str]:
    errors: list[str] = []
    checkpoint = payload["after"]["checkpoint"]
    if mode in {"checkpoint", "all"} and not checkpoint["exists"]:
        errors.append(f"checkpoint_missing:{checkpoint['path']}")
    if mode in {"envs", "all"}:
        envs = payload["after"]["simulation_envs"]
        if not envs["pickles_exists"] or envs["num_pickles"] <= 0:
            errors.append(f"env_pickles_missing:{envs['pickles_dir']}")
        if not envs["jsons_exists"] or envs["num_jsons"] <= 0:
            errors.append(f"env_jsons_missing:{envs['jsons_dir']}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify or optionally download baseline assets.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", choices=["checkpoint", "envs", "all"], default="all")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    payload = {"before": inspect_assets(config)}
    if args.download:
        if args.mode in {"checkpoint", "all"}:
            payload["checkpoint_download"] = _download_checkpoint(config)
        if args.mode in {"envs", "all"}:
            payload["env_download"] = {
                "status": "manual_or_external",
                "hint": "Place the 75-environment Waymo pack under artifacts/assets/scenario_dreamer_waymo_200m_pickles. Use the shared Drive URL for the official release assets.",
                "url": config["assets"]["simulation_envs"]["shared_drive_url"],
            }
    payload["after"] = inspect_assets(config)
    payload["verification_errors"] = _verification_errors(payload, args.mode)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if payload["verification_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
