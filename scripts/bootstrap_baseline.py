#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from _script_utils import ROOT  # noqa: F401
from scenario_dreamer_decision_layer.bootstrap import ensure_upstream_clone, inspect_assets, write_bootstrap_lock
from scenario_dreamer_decision_layer.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Clone/pin upstream Scenario Dreamer and inspect baseline assets.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--clone-upstream", action="store_true")
    parser.add_argument("--write-lock", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    clone_result = ensure_upstream_clone(config, clone=args.clone_upstream and not args.dry_run)
    asset_state = inspect_assets(config)
    payload = {"upstream": clone_result, "assets": asset_state}
    if args.write_lock and not args.dry_run:
        payload["lock_path"] = str(write_bootstrap_lock(config, clone_result, asset_state))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
