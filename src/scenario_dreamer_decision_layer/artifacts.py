from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .contracts import RunManifest, VideoManifest
from .config import project_root


class RunBundle(dict):
    pass


def utc_now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_jsonable(payload: Dict[str, Any]) -> str:
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_run_bundle(tag: str, tier: str) -> RunBundle:
    run_id = f"{tag}_{tier}"
    run_dir = ensure_dir(project_root() / "results" / "runs" / run_id)
    return RunBundle(
        run_id=run_id,
        run_dir=run_dir,
        config_snapshot=run_dir / "config_snapshot.json",
        run_manifest=run_dir / "run_manifest.json",
        metrics=run_dir / "metrics.json",
        stdout=run_dir / "stdout.log",
        stderr=run_dir / "stderr.log",
        video_manifest=run_dir / "video_manifest.json",
        subset_dir=run_dir / "_subset_pickles",
        movie_dir=run_dir / "movies",
    )


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_manifest(path: Path, manifest: RunManifest) -> None:
    write_json(path, asdict(manifest))


def write_video_manifest(path: Path, manifest: VideoManifest) -> None:
    write_json(path, asdict(manifest))
