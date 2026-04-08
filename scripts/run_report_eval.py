#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from _script_utils import ROOT  # noqa: F401
from scenario_dreamer_decision_layer.runner import run_tier, write_transfer_hook_request


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the report-tier Scenario Dreamer baseline.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-transfer-hook", action="store_true")
    args = parser.parse_args()
    payload = run_tier("report", dry_run=args.dry_run)
    if args.write_transfer_hook and not args.dry_run:
        from pathlib import Path
        hook_path = write_transfer_hook_request(Path(payload["run_dir"]) / "run_manifest.json")
        payload["transfer_hook_request"] = str(hook_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
