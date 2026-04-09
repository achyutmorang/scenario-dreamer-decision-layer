#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _script_utils import ROOT  # noqa: F401
from scenario_dreamer_decision_layer.runner import run_risk_variance_study


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _format_float(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _print_progress(payload: dict[str, object]) -> None:
    event = str(payload.get("event", ""))
    prefix = "[risk-study]"
    scene_position = payload.get("scene_position")
    scene_total = payload.get("scene_total")
    scene_label = f" scene {scene_position}/{scene_total}" if scene_position and scene_total else ""
    scenario_index = payload.get("scenario_index")
    scenario_name = payload.get("scenario_name")
    scenario_suffix = ""
    if scenario_name:
        scenario_suffix = f" scenario={scenario_name}"
    elif scenario_index is not None:
        scenario_suffix = f" scenario_index={scenario_index}"

    if event == "risk_study_started":
        message = (
            f"{prefix} study started scenes={payload.get('scene_count')} "
            f"seeds={payload.get('seed_count')} risk_key={payload.get('risk_key')} "
            f"selector_k={payload.get('selector_k_values')}"
        )
    elif event == "scene_started":
        message = f"{prefix}{scene_label}{scenario_suffix} started"
    elif event == "diversity_audit_started":
        message = (
            f"{prefix}{scene_label}{scenario_suffix} audit started "
            f"seed_count={payload.get('seed_count')}"
        )
    elif event == "seed_started":
        message = (
            f"{prefix}{scene_label}{scenario_suffix} "
            f"seed {payload.get('seed_position')}/{payload.get('seed_total')} "
            f"seed={payload.get('seed')} started"
        )
    elif event == "seed_completed":
        metrics = payload.get("metrics", {}) or {}
        trajectory_metrics = payload.get("trajectory_metrics", {}) or {}
        message = (
            f"{prefix}{scene_label}{scenario_suffix} "
            f"seed {payload.get('seed_position')}/{payload.get('seed_total')} "
            f"seed={payload.get('seed')} done "
            f"elapsed={_format_float(payload.get('elapsed_seconds'))}s "
            f"collision={_format_float(metrics.get('collision_rate'))} "
            f"off_route={_format_float(metrics.get('off_route_rate'))} "
            f"completed={_format_float(metrics.get('completed_rate'))} "
            f"progress={_format_float(metrics.get('progress'))} "
            f"min_ttc={_format_float(trajectory_metrics.get('min_ttc_proxy_s'))} "
            f"min_dist={_format_float(trajectory_metrics.get('min_ego_agent_distance_m'))} "
            f"hard_brake={trajectory_metrics.get('hard_brake_count', 'n/a')} "
            f"termination={payload.get('termination_reason', 'n/a')}"
        )
    elif event == "diversity_audit_completed":
        message = (
            f"{prefix}{scene_label}{scenario_suffix} audit done "
            f"decision={payload.get('decision')} "
            f"metric_diversity={payload.get('metric_level_diversity_detected')} "
            f"trajectory_diversity={payload.get('trajectory_level_diversity_detected')}"
        )
    elif event == "scene_completed":
        risk_variance = payload.get("risk_variance", {}) or {}
        selector_probe = payload.get("selector_probe", []) or []
        selector_compact = [
            {
                "k": item.get("k"),
                "baseline": item.get("baseline_risk"),
                "selected": item.get("selected_risk"),
                "delta": item.get("risk_improvement"),
            }
            for item in selector_probe
        ]
        message = (
            f"{prefix}{scene_label}{scenario_suffix} completed "
            f"decision={payload.get('decision')} "
            f"risk_mean={_format_float(risk_variance.get('mean'))} "
            f"risk_range={_format_float(risk_variance.get('range'))} "
            f"selector={json.dumps(selector_compact, sort_keys=True)}"
        )
    elif event == "risk_study_completed":
        message = (
            f"{prefix} study completed run_id={payload.get('run_id')} "
            f"scene_count={payload.get('scene_count')} "
            f"aggregate={json.dumps(payload.get('aggregate', {}), sort_keys=True)} "
            f"selector_summary={json.dumps(payload.get('selector_summary', {}), sort_keys=True)}"
        )
    else:
        message = f"{prefix} {json.dumps(payload, sort_keys=True)}"
    print(message, file=sys.stderr, flush=True)


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
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write the final JSON payload. When set, stdout stays quiet and progress prints to stderr.",
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
        progress=_print_progress,
    )
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized, encoding="utf-8")
        print(f"[risk-study] wrote summary to {args.output_json}", file=sys.stderr, flush=True)
    else:
        print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
