#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from _script_utils import ROOT  # noqa: F401


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one fixed-scenario diversity trace and emit metric and trajectory summaries as JSON."
    )
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _compose_cfg(config_dir: str, overrides: List[str]):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name="config", overrides=overrides)


def _angle_delta(current: float, previous: float) -> float:
    return float((current - previous + math.pi) % (2 * math.pi) - math.pi)


def _speed(state: np.ndarray) -> float:
    return float(np.linalg.norm(state[2:4]))


def _min_ego_agent_distance(ego_state: np.ndarray, agent_states: np.ndarray, agent_active: np.ndarray) -> float | None:
    if not np.any(agent_active):
        return None
    active_agents = agent_states[agent_active]
    distances = np.linalg.norm(active_agents[:, :2] - ego_state[:2], axis=1)
    if distances.size == 0:
        return None
    return float(np.min(distances))


def _min_ttc_proxy(ego_state: np.ndarray, agent_states: np.ndarray, agent_active: np.ndarray) -> float | None:
    if not np.any(agent_active):
        return None
    active_agents = agent_states[agent_active]
    rel_pos = active_agents[:, :2] - ego_state[:2]
    distances = np.linalg.norm(rel_pos, axis=1)
    valid = distances > 1e-6
    if not np.any(valid):
        return None
    rel_pos = rel_pos[valid]
    distances = distances[valid]
    rel_vel = active_agents[valid, 2:4] - ego_state[2:4]
    rel_dir = rel_pos / distances[:, None]
    closing_speed = -np.sum(rel_vel * rel_dir, axis=1)
    approaching = closing_speed > 1e-6
    if not np.any(approaching):
        return None
    ttc_values = distances[approaching] / closing_speed[approaching]
    if ttc_values.size == 0:
        return None
    return float(np.min(ttc_values))


def _nearest_agent_index(ego_state: np.ndarray, agent_states: np.ndarray, agent_active: np.ndarray) -> int | None:
    if not np.any(agent_active):
        return None
    active_indices = np.where(agent_active)[0]
    active_agents = agent_states[active_indices]
    distances = np.linalg.norm(active_agents[:, :2] - ego_state[:2], axis=1)
    if distances.size == 0:
        return None
    return int(active_indices[int(np.argmin(distances))])


def _termination_reason(info: Dict[str, Any]) -> str:
    if info.get("collision"):
        return "collision"
    if info.get("off_route"):
        return "off_route"
    if info.get("completed"):
        return "completed"
    return "timeout"


def _trace_summary(trace: List[Dict[str, Any]], dt: float, info: Dict[str, Any], final_step: int) -> Dict[str, Any]:
    positions = np.array([step["ego_position"] for step in trace], dtype=float)
    speeds = np.array([step["ego_speed_mps"] for step in trace], dtype=float)
    headings = np.array([step["ego_heading_rad"] for step in trace], dtype=float)
    min_distances = [step["min_ego_agent_distance_m"] for step in trace if step["min_ego_agent_distance_m"] is not None]
    ttc_values = [step["min_ttc_proxy_s"] for step in trace if step["min_ttc_proxy_s"] is not None]
    nearest_agent_indices = [step["nearest_agent_index"] for step in trace]

    path_length = 0.0
    if len(positions) > 1:
        path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())

    accel = np.diff(speeds) / dt if len(speeds) > 1 else np.array([], dtype=float)
    hard_brake_count = int(np.sum(accel < -2.0))

    stop_count = 0
    was_stopped = False
    for speed in speeds:
        is_stopped = speed < 0.5
        if is_stopped and not was_stopped:
            stop_count += 1
        was_stopped = is_stopped

    nearest_agent_switch_count = 0
    previous_idx = None
    for current_idx in nearest_agent_indices:
        if current_idx is None:
            continue
        if previous_idx is not None and current_idx != previous_idx:
            nearest_agent_switch_count += 1
        previous_idx = current_idx

    heading_change_abs = 0.0
    if len(headings) > 1:
        heading_change_abs = float(
            sum(abs(_angle_delta(curr, prev)) for prev, curr in zip(headings[:-1], headings[1:]))
        )

    trajectory_metrics = {
        "path_length_m": path_length,
        "terminal_x_m": float(positions[-1, 0]),
        "terminal_y_m": float(positions[-1, 1]),
        "terminal_speed_mps": float(speeds[-1]),
        "min_ego_agent_distance_m": float(min(min_distances)) if min_distances else None,
        "min_ttc_proxy_s": float(min(ttc_values)) if ttc_values else None,
        "hard_brake_count": hard_brake_count,
        "stop_count": stop_count,
        "nearest_agent_switch_count": nearest_agent_switch_count,
        "heading_change_abs_rad": heading_change_abs,
        "terminated_step": int(final_step),
        "progress": float(info.get("progress", 0.0)),
    }
    trajectory_categories = {"termination_reason": _termination_reason(info)}
    return {
        "trajectory_metrics": trajectory_metrics,
        "trajectory_categories": trajectory_categories,
    }


def main() -> int:
    args = _parse_args()
    cfg = _compose_cfg(args.config_dir, args.overrides)

    import torch
    from policies.idm_policy import IDMPolicy
    from policies.rl_policy import RLPolicy
    from simulator import Simulator
    from utils.viz import generate_video

    torch.manual_seed(cfg.sim.seed)
    random.seed(cfg.sim.seed)
    np.random.seed(cfg.sim.seed)

    env = Simulator(cfg)
    if cfg.sim.policy == "rl":
        policy = RLPolicy(cfg.sim)
    else:
        policy = IDMPolicy(cfg, env)

    obs = env.reset(0)
    if hasattr(policy, "reset"):
        policy.reset(obs)

    trace: List[Dict[str, Any]] = []
    info: Dict[str, Any] = {}

    trace.append(
        {
            "step": 0,
            "ego_position": env.ego_state[:2].astype(float).tolist(),
            "ego_speed_mps": _speed(env.ego_state),
            "ego_heading_rad": float(env.ego_state[4]),
            "min_ego_agent_distance_m": _min_ego_agent_distance(env.ego_state, env.data_dict["agent"][-1], env.agent_active),
            "min_ttc_proxy_s": _min_ttc_proxy(env.ego_state, env.data_dict["agent"][-1], env.agent_active),
            "nearest_agent_index": _nearest_agent_index(env.ego_state, env.data_dict["agent"][-1], env.agent_active),
        }
    )

    final_step = 0
    for step in range(env.steps):
        if cfg.sim.visualize:
            render_frame = not cfg.sim.lightweight or step % 3 == 0
            if render_frame:
                env.render_state(name="0", movie_path=cfg.sim.movie_path)

        action = policy.act(obs)
        obs, terminated, info = env.step(action)
        final_step = env.t
        trace.append(
            {
                "step": int(env.t),
                "ego_position": env.ego_state[:2].astype(float).tolist(),
                "ego_speed_mps": _speed(env.ego_state),
                "ego_heading_rad": float(env.ego_state[4]),
                "min_ego_agent_distance_m": _min_ego_agent_distance(env.ego_state, env.data_dict["agent"][-1], env.agent_active),
                "min_ttc_proxy_s": _min_ttc_proxy(env.ego_state, env.data_dict["agent"][-1], env.agent_active),
                "nearest_agent_index": _nearest_agent_index(env.ego_state, env.data_dict["agent"][-1], env.agent_active),
            }
        )
        if terminated:
            break

    if cfg.sim.visualize:
        generate_video(name="0", output_dir=cfg.sim.movie_path, delete_images=True)

    metrics = {
        "collision_rate": float(info.get("collision", False)),
        "off_route_rate": float(info.get("off_route", False)),
        "completed_rate": float(info.get("completed", False)),
        "progress": float(info.get("progress", 0.0)),
    }
    payload = {
        "metrics": metrics,
        "trajectory_summary": _trace_summary(trace, float(cfg.sim.dt), info, final_step),
    }
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
