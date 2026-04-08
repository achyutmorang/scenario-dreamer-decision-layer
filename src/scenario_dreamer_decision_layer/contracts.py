from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class BaselineIdentity:
    name: str
    upstream_repo_url: str
    upstream_repo_commit: str
    checkpoint_path: str
    environment_path: str
    tier: str
    seed: int


@dataclass
class RunManifest:
    run_id: str
    created_utc: str
    baseline: BaselineIdentity
    command: list[str]
    config_path: str
    config_hash: str
    num_scenarios: int
    elapsed_seconds: float
    stdout_log: str
    stderr_log: str


@dataclass
class VideoManifest:
    run_id: str
    movie_dir: str
    files: list[str]
    lightweight: bool


@dataclass
class TransferHookRequest:
    source_run_id: str
    baseline_name: str
    metrics_path: str
    run_manifest_path: str
    target_interface: str = "waymax_wosac_placeholder"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
