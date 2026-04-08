from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .contracts import TransferHookRequest


def load_run_manifest(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_transfer_request(run_manifest_path: str | Path) -> TransferHookRequest:
    payload = load_run_manifest(run_manifest_path)
    manifest_path = Path(run_manifest_path)
    return TransferHookRequest(
        source_run_id=payload["run_id"],
        baseline_name=payload["baseline"]["name"],
        metrics_path=str(manifest_path.parent / "metrics.json"),
        run_manifest_path=str(manifest_path),
    )
