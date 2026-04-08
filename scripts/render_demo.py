#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from _script_utils import ROOT  # noqa: F401
from scenario_dreamer_decision_layer.runner import run_tier


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a 1-environment lightweight demo video.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()
    payload = run_tier("demo", dry_run=args.dry_run, num_envs=args.num_envs, force_visualize=True, force_lightweight=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
