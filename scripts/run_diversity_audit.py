#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from _script_utils import ROOT  # noqa: F401
from scenario_dreamer_decision_layer.runner import run_diversity_audit


def _parse_seed_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Experiment 0: audit metric-level future diversity on one fixed scenario across multiple seeds."
    )
    parser.add_argument("--scenario-index", type=int, default=0, help="0-based scenario index within the bound environment pack.")
    parser.add_argument(
        "--seeds",
        type=_parse_seed_list,
        default=_parse_seed_list("0,1,2,3"),
        help="Comma-separated list of seeds to sweep.",
    )
    parser.add_argument("--visualize", action="store_true", help="Render per-seed videos during the audit.")
    parser.add_argument("--no-lightweight", dest="lightweight", action="store_false", help="Disable lightweight rendering.")
    parser.set_defaults(lightweight=True)
    args = parser.parse_args()
    payload = run_diversity_audit(
        scenario_index=args.scenario_index,
        seeds=args.seeds,
        visualize=args.visualize,
        lightweight=args.lightweight,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
