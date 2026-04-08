from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict

from .config import load_config, project_root, resolve_repo_relative


def default_drive_layout(drive_root: str | Path) -> Dict[str, Path]:
    base = Path(drive_root).expanduser().resolve() / "scenario_dreamer_decision_layer"
    assets_root = base / "assets"
    return {
        "base": base,
        "assets_root": assets_root,
        "scratch_root": assets_root / "scenario-dreamer",
        "datasets_root": assets_root / "datasets",
        "env_pickles_dir": assets_root / "scenario_dreamer_waymo_200m_pickles",
        "env_jsons_dir": assets_root / "scenario_dreamer_waymo_200m_jsons",
        "results_root": base / "results" / "runs",
    }


def _canonical_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    return {
        "scratch_root": resolve_repo_relative(config["assets"]["scratch_root"]),
        "datasets_root": resolve_repo_relative(config["assets"]["dataset_root"]),
        "env_pickles_dir": resolve_repo_relative(config["assets"]["simulation_envs"]["pickles_dir"]),
        "env_jsons_dir": resolve_repo_relative(config["assets"]["simulation_envs"]["jsons_dir"]),
    }


def _replace_with_symlink(canonical: Path, actual: Path) -> Dict[str, Any]:
    actual.mkdir(parents=True, exist_ok=True)
    if canonical.is_symlink():
        current = canonical.resolve()
        if current == actual:
            return {"mode": "symlink", "status": "already_bound", "canonical": str(canonical), "actual": str(actual)}
        canonical.unlink()
    elif canonical.exists():
        if canonical.is_dir():
            if any(canonical.iterdir()):
                raise RuntimeError(f"Refusing to replace non-empty directory with symlink: {canonical}")
            canonical.rmdir()
        else:
            raise RuntimeError(f"Refusing to replace file with symlink: {canonical}")
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.symlink_to(actual, target_is_directory=True)
    return {"mode": "symlink", "status": "bound", "canonical": str(canonical), "actual": str(actual)}


def bind_drive_layout(drive_root: str | Path, *, config_path: str | Path | None = None) -> Dict[str, Any]:
    config = load_config(config_path)
    layout = default_drive_layout(drive_root)
    canonical = _canonical_paths(config)

    bindings = {
        "scratch_root": _replace_with_symlink(canonical["scratch_root"], layout["scratch_root"]),
        "datasets_root": _replace_with_symlink(canonical["datasets_root"], layout["datasets_root"]),
        "env_pickles_dir": _replace_with_symlink(canonical["env_pickles_dir"], layout["env_pickles_dir"]),
        "env_jsons_dir": _replace_with_symlink(canonical["env_jsons_dir"], layout["env_jsons_dir"]),
    }
    layout["results_root"].mkdir(parents=True, exist_ok=True)
    os.environ["SCENARIO_DREAMER_RESULTS_ROOT"] = str(layout["results_root"])
    return {
        "project_root": str(project_root()),
        "drive_root": str(Path(drive_root).expanduser().resolve()),
        "results_root": str(layout["results_root"]),
        "bindings": bindings,
    }


def seed_env_pack_from_upstream(*, config_path: str | Path | None = None) -> Dict[str, Any]:
    config = load_config(config_path)
    upstream_dir = resolve_repo_relative(config["upstream"]["repo_dir"])
    assert upstream_dir is not None
    metadata_root = upstream_dir / "metadata" / "simulation_environment_datasets"
    canonical = _canonical_paths(config)
    pickles_dir = canonical["env_pickles_dir"]
    jsons_dir = canonical["env_jsons_dir"]
    source_pickles = metadata_root / pickles_dir.name
    source_jsons = metadata_root / jsons_dir.name

    result: Dict[str, Any] = {
        "metadata_root": str(metadata_root),
        "source_pickles": str(source_pickles),
        "source_jsons": str(source_jsons),
        "target_pickles": str(pickles_dir),
        "target_jsons": str(jsons_dir),
    }

    copied_pickles = 0
    copied_jsons = 0
    pickles_dir.mkdir(parents=True, exist_ok=True)
    jsons_dir.mkdir(parents=True, exist_ok=True)

    if source_pickles.exists():
        for candidate in sorted(source_pickles.glob("*.pkl")):
            target = pickles_dir / candidate.name
            if target.exists():
                continue
            shutil.copy2(candidate, target)
            copied_pickles += 1

    if source_jsons.exists():
        for candidate in sorted(source_jsons.glob("*.json")):
            target = jsons_dir / candidate.name
            if target.exists():
                continue
            shutil.copy2(candidate, target)
            copied_jsons += 1

    result["copied_pickles"] = copied_pickles
    result["copied_jsons"] = copied_jsons
    result["num_pickles_after"] = len(sorted(pickles_dir.glob("*.pkl"))) if pickles_dir.exists() else 0
    result["num_jsons_after"] = len(sorted(jsons_dir.glob("*.json"))) if jsons_dir.exists() else 0
    result["status"] = "seeded" if result["num_pickles_after"] > 0 and result["num_jsons_after"] > 0 else "source_missing_or_empty"
    return result


def inspect_bound_layout(*, config_path: str | Path | None = None) -> Dict[str, Any]:
    config = load_config(config_path)
    canonical = _canonical_paths(config)
    payload: Dict[str, Any] = {}
    for name, path in canonical.items():
        payload[name] = {
            "canonical": str(path),
            "exists": path.exists(),
            "is_symlink": path.is_symlink(),
            "resolved": str(path.resolve()) if path.exists() or path.is_symlink() else "",
        }
    payload["results_root"] = os.environ.get("SCENARIO_DREAMER_RESULTS_ROOT", "")
    return payload
