#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from _script_utils import ROOT  # noqa: F401
from scenario_dreamer_decision_layer.bootstrap import inspect_assets
from scenario_dreamer_decision_layer.config import load_config, resolve_repo_relative


def _checkpoint_expected_path(config) -> Path:
    return resolve_repo_relative(config["assets"]["checkpoint"]["relative_ckpt_path"])


def _checkpoint_search_root(config) -> Path:
    scratch_root = resolve_repo_relative(config["assets"]["scratch_root"])
    return scratch_root / "checkpoints"


def _env_expected_dirs(config) -> tuple[Path, Path]:
    return (
        resolve_repo_relative(config["assets"]["simulation_envs"]["pickles_dir"]),
        resolve_repo_relative(config["assets"]["simulation_envs"]["jsons_dir"]),
    )


def _env_download_root(config) -> Path:
    dataset_root = resolve_repo_relative(config["assets"]["dataset_root"])
    return dataset_root / "scenario_dreamer_release"


def _env_archive_path(config) -> Path:
    download_root = _env_download_root(config)
    return download_root / config["assets"]["simulation_envs"].get("archive_name", "simulation_environment_datasets.tar")


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


def _normalize_env_layout(config):
    pickles_dir, jsons_dir = _env_expected_dirs(config)
    download_root = _env_download_root(config)
    result = {
        "download_root": str(download_root),
        "pickles_dir": str(pickles_dir),
        "jsons_dir": str(jsons_dir),
    }
    pickles_dir.mkdir(parents=True, exist_ok=True)
    jsons_dir.mkdir(parents=True, exist_ok=True)
    if not download_root.exists():
        result["status"] = "download_root_missing"
        return result

    discovered_pickles_dirs = sorted(
        str(path) for path in download_root.rglob("*")
        if path.is_dir() and path.name.endswith("_pickles")
    )
    discovered_jsons_dirs = sorted(
        str(path) for path in download_root.rglob("*")
        if path.is_dir() and path.name.endswith("_jsons")
    )
    result["discovered_pickles_dirs"] = discovered_pickles_dirs
    result["discovered_jsons_dirs"] = discovered_jsons_dirs

    moved_pickles = 0
    moved_jsons = 0

    exact_pickles_sources = sorted(
        path for path in download_root.rglob(pickles_dir.name)
        if path.is_dir()
    )
    exact_jsons_sources = sorted(
        path for path in download_root.rglob(jsons_dir.name)
        if path.is_dir()
    )
    result["exact_pickles_sources"] = [str(path) for path in exact_pickles_sources]
    result["exact_jsons_sources"] = [str(path) for path in exact_jsons_sources]

    pickles_sources = exact_pickles_sources
    jsons_sources = exact_jsons_sources

    if not pickles_sources and len(discovered_pickles_dirs) == 1:
        pickles_sources = [Path(discovered_pickles_dirs[0])]
        result["pickles_fallback"] = "single_discovered_dir"
    if not jsons_sources and len(discovered_jsons_dirs) == 1:
        jsons_sources = [Path(discovered_jsons_dirs[0])]
        result["jsons_fallback"] = "single_discovered_dir"

    for source_dir in pickles_sources:
        for candidate in sorted(source_dir.glob("*.pkl")):
            target = pickles_dir / candidate.name
            if candidate == target or target.exists():
                continue
            shutil.move(str(candidate), str(target))
            moved_pickles += 1

    for source_dir in jsons_sources:
        for candidate in sorted(source_dir.glob("*.json")):
            target = jsons_dir / candidate.name
            if candidate == target or target.exists():
                continue
            shutil.move(str(candidate), str(target))
            moved_jsons += 1

    result["moved_pickles"] = moved_pickles
    result["moved_jsons"] = moved_jsons
    result["status"] = "normalized" if (pickles_sources or jsons_sources) else "no_matching_dirs"
    return result


def _extract_env_archive(config):
    archive_path = _env_archive_path(config)
    download_root = _env_download_root(config)
    extract_root = download_root / "extracted"
    result = {
        "archive_path": str(archive_path),
        "extract_root": str(extract_root),
    }
    if not archive_path.exists():
        result["status"] = "archive_missing"
        return result
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as tf:
        tf.extractall(extract_root)
    result["status"] = "extracted"
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


def _download_envs(config):
    gdown = shutil.which("gdown")
    if not gdown:
        return {
            "status": "gdown_missing",
            "hint": "Install gdown or place the environment pack manually.",
            "url": config["assets"]["simulation_envs"].get("archive_url") or config["assets"]["simulation_envs"]["shared_drive_url"],
        }
    download_root = _env_download_root(config)
    download_root.mkdir(parents=True, exist_ok=True)
    archive_url = config["assets"]["simulation_envs"].get("archive_url")
    archive_path = _env_archive_path(config)
    if archive_url:
        cmd = [gdown, archive_url, "-O", str(archive_path)]
    else:
        cmd = [gdown, "--folder", config["assets"]["simulation_envs"]["shared_drive_url"], "-O", str(download_root)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    payload = {
        "status": "ok" if proc.returncode == 0 else "failed",
        "command": cmd,
        "stdout": proc.stdout[-1200:],
        "stderr": proc.stderr[-1200:],
    }
    payload["archive_extract"] = _extract_env_archive(config)
    payload["normalize"] = _normalize_env_layout(config)
    pickles_dir, jsons_dir = _env_expected_dirs(config)
    payload["pickles_exists_after"] = pickles_dir.exists() and any(pickles_dir.glob("*.pkl"))
    payload["jsons_exists_after"] = jsons_dir.exists() and any(jsons_dir.glob("*.json"))
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
            payload["env_download"] = _download_envs(config)
    payload["after"] = inspect_assets(config)
    payload["verification_errors"] = _verification_errors(payload, args.mode)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if payload["verification_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
