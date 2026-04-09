#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from _script_utils import ROOT  # noqa: F401
from scenario_dreamer_decision_layer.runner import run_risk_variance_study


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a multi-scene risk variance study and an offline selector upper-bound probe "
            "on the frozen Scenario Dreamer + CtRL-Sim + IDM stack."
        )
    )
    parser.add_argument(
        "--scenario-indices",
        type=_parse_int_list,
        default=_parse_int_list("0,1,2,3,4"),
        help="Comma-separated fixed scenario indices to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_int_list,
        default=_parse_int_list("0,1,2,3,4,5,6,7,8,9"),
        help="Comma-separated seed list for per-scene repeated rollouts.",
    )
    parser.add_argument(
        "--selector-k-values",
        type=_parse_int_list,
        default=_parse_int_list("1,2,5,10"),
        help="Comma-separated K values for the offline selector upper-bound analysis.",
    )
    parser.add_argument(
        "--risk-key",
        choices=["min_ttc_proxy_s", "min_ego_agent_distance_m", "hard_brake_count"],
        default="min_ttc_proxy_s",
        help="Risk proxy used for per-scene variance summaries and selector ranking.",
    )
    parser.add_argument("--visualize", action="store_true", help="Render per-seed videos during the study.")
    parser.add_argument("--no-lightweight", dest="lightweight", action="store_false", help="Disable lightweight mode.")
    parser.set_defaults(lightweight=True)
    args = parser.parse_args()

    payload = run_risk_variance_study(
        scenario_indices=args.scenario_indices,
        seeds=args.seeds,
        selector_k_values=args.selector_k_values,
        risk_key=args.risk_key,
        visualize=args.visualize,
        lightweight=args.lightweight,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
