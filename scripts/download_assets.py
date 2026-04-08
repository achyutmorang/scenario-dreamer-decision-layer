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


def _download_checkpoint(config):
    gdown = shutil.which("gdown")
    if not gdown:
        return {"status": "gdown_missing", "hint": "Install gdown or place the checkpoint manually.", "url": config['assets']['checkpoint']['google_drive_url']}
    scratch_root = resolve_repo_relative(config["assets"]["scratch_root"])
    scratch_root.mkdir(parents=True, exist_ok=True)
    cmd = [gdown, "--folder", config["assets"]["checkpoint"]["google_drive_url"], "-O", str(scratch_root / "checkpoints")]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return {"status": "ok" if proc.returncode == 0 else "failed", "command": cmd, "stdout": proc.stdout[-800:], "stderr": proc.stderr[-800:]}


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
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
