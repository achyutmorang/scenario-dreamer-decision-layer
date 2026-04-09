from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenario_dreamer_decision_layer.runner import _select_pickle_files, run_diversity_audit


class RunnerTests(unittest.TestCase):
    def test_select_pickle_files_supports_indices_and_offset(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for idx in range(5):
                (root / f"{idx:02d}.pkl").write_text("x", encoding="utf-8")
            selected = _select_pickle_files(root, 2, offset=1)
            self.assertEqual([p.name for p in selected], ["01.pkl", "02.pkl"])
            selected_indices = _select_pickle_files(root, 1, indices=[4, 1])
            self.assertEqual([p.name for p in selected_indices], ["04.pkl", "01.pkl"])

    def test_run_diversity_audit_summarizes_seed_spread(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp_root = Path(td)
            env_pickles = temp_root / "pickles"
            env_pickles.mkdir(parents=True, exist_ok=True)
            (env_pickles / "scene_0.pkl").write_text("pickle-bytes", encoding="utf-8")

            config = {
                "baseline_name": "scenario_dreamer_ctrlsim",
                "upstream": {
                    "repo_url": "https://example.com/upstream.git",
                    "repo_commit": "deadbeef",
                    "repo_dir": str(temp_root / "upstream"),
                },
                "assets": {
                    "checkpoint": {"relative_ckpt_path": str(temp_root / "ckpt" / "last.ckpt")},
                    "simulation_envs": {
                        "pickles_dir": str(env_pickles),
                        "jsons_dir": str(temp_root / "jsons"),
                    },
                    "scratch_root": str(temp_root / "scratch"),
                    "movies_dir": str(temp_root / "movies"),
                    "dataset_root": str(temp_root / "datasets"),
                },
                "baseline": {
                    "sim_mode": "scenario_dreamer",
                    "policy": "idm",
                    "behaviour_model_run_name": "ctrl_sim_waymo_1M_steps",
                    "simulate_vehicles_only": True,
                    "verbose": True,
                    "steps": 400,
                    "dt": 0.1,
                    "agent_scale": 1.0,
                    "tilt": 10,
                    "action_temperature": 1.0,
                    "use_rtg": True,
                    "predict_rtgs": True,
                    "compute_behaviour_metrics": False,
                },
                "evaluation": {
                    "metrics": [
                        "collision_rate",
                        "off_route_rate",
                        "completed_rate",
                        "progress",
                        "runtime_throughput_scenarios_per_sec",
                        "num_scenarios",
                        "elapsed_seconds",
                    ]
                },
            }
            upstream_dir = Path(config["upstream"]["repo_dir"])
            upstream_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = Path(config["assets"]["checkpoint"]["relative_ckpt_path"])
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt_path.write_text("checkpoint", encoding="utf-8")

            seed_to_progress = {0: 10.0, 1: 10.0, 2: 12.5}
            seed_to_terminal_y = {0: 1.0, 1: 1.5, 2: 2.0}

            def fake_run(cmd, cwd, env, text, capture_output, check):
                seed_token = next(part for part in cmd if part.startswith("sim.seed="))
                seed = int(seed_token.split("=", 1)[1])
                stdout = json.dumps(
                    {
                        "metrics": {
                            "collision_rate": 0.0,
                            "off_route_rate": 0.0,
                            "completed_rate": 1.0,
                            "progress": seed_to_progress[seed],
                        },
                        "trajectory_summary": {
                            "trajectory_metrics": {
                                "terminal_y_m": seed_to_terminal_y[seed],
                                "min_ego_agent_distance_m": 4.0,
                                "hard_brake_count": float(seed),
                            },
                            "trajectory_categories": {
                                "termination_reason": "completed",
                            },
                        },
                    }
                )

                class Result:
                    returncode = 0
                    stderr = ""

                    def __init__(self, out: str):
                        self.stdout = out

                return Result(stdout)

            with (
                patch("scenario_dreamer_decision_layer.runner.load_config", return_value=config),
                patch("scenario_dreamer_decision_layer.runner.project_root", return_value=temp_root),
                patch("scenario_dreamer_decision_layer.runner.subprocess.run", side_effect=fake_run),
            ):
                summary = run_diversity_audit(scenario_index=0, seeds=[0, 1, 2], visualize=False, lightweight=True)

            self.assertTrue(summary["diversity_detected"])
            self.assertEqual(summary["scenario_name"], "scene_0.pkl")
            self.assertEqual(summary["metric_spread"]["progress"]["range"], 2.5)
            self.assertEqual(summary["trajectory_metric_spread"]["terminal_y_m"]["range"], 1.0)
            self.assertTrue(summary["metric_level_diversity_detected"])
            self.assertTrue(summary["trajectory_level_diversity_detected"])
            self.assertTrue((Path(summary["run_dir"]) / "diversity_audit_summary.json").exists())

    def test_run_diversity_audit_detects_trajectory_level_diversity_when_episode_metrics_are_flat(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp_root = Path(td)
            env_pickles = temp_root / "pickles"
            env_pickles.mkdir(parents=True, exist_ok=True)
            (env_pickles / "scene_0.pkl").write_text("pickle-bytes", encoding="utf-8")

            config = {
                "baseline_name": "scenario_dreamer_ctrlsim",
                "upstream": {
                    "repo_url": "https://example.com/upstream.git",
                    "repo_commit": "deadbeef",
                    "repo_dir": str(temp_root / "upstream"),
                },
                "assets": {
                    "checkpoint": {"relative_ckpt_path": str(temp_root / "ckpt" / "last.ckpt")},
                    "simulation_envs": {
                        "pickles_dir": str(env_pickles),
                        "jsons_dir": str(temp_root / "jsons"),
                    },
                    "scratch_root": str(temp_root / "scratch"),
                    "movies_dir": str(temp_root / "movies"),
                    "dataset_root": str(temp_root / "datasets"),
                },
                "baseline": {
                    "sim_mode": "scenario_dreamer",
                    "policy": "idm",
                    "behaviour_model_run_name": "ctrl_sim_waymo_1M_steps",
                    "simulate_vehicles_only": True,
                    "verbose": True,
                    "steps": 400,
                    "dt": 0.1,
                    "agent_scale": 1.0,
                    "tilt": 10,
                    "action_temperature": 1.0,
                    "use_rtg": True,
                    "predict_rtgs": True,
                    "compute_behaviour_metrics": False,
                },
                "evaluation": {
                    "metrics": [
                        "collision_rate",
                        "off_route_rate",
                        "completed_rate",
                        "progress",
                        "runtime_throughput_scenarios_per_sec",
                        "num_scenarios",
                        "elapsed_seconds",
                    ]
                },
            }
            upstream_dir = Path(config["upstream"]["repo_dir"])
            upstream_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = Path(config["assets"]["checkpoint"]["relative_ckpt_path"])
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt_path.write_text("checkpoint", encoding="utf-8")

            seed_to_terminal_reason = {0: "completed", 1: "off_route", 2: "completed"}

            def fake_run(cmd, cwd, env, text, capture_output, check):
                seed_token = next(part for part in cmd if part.startswith("sim.seed="))
                seed = int(seed_token.split("=", 1)[1])
                stdout = json.dumps(
                    {
                        "metrics": {
                            "collision_rate": 0.0,
                            "off_route_rate": 0.0,
                            "completed_rate": 1.0,
                            "progress": 10.0,
                        },
                        "trajectory_summary": {
                            "trajectory_metrics": {
                                "terminal_y_m": float(seed),
                                "min_ego_agent_distance_m": 5.0,
                            },
                            "trajectory_categories": {
                                "termination_reason": seed_to_terminal_reason[seed],
                            },
                        },
                    }
                )

                class Result:
                    returncode = 0
                    stderr = ""

                    def __init__(self, out: str):
                        self.stdout = out

                return Result(stdout)

            with (
                patch("scenario_dreamer_decision_layer.runner.load_config", return_value=config),
                patch("scenario_dreamer_decision_layer.runner.project_root", return_value=temp_root),
                patch("scenario_dreamer_decision_layer.runner.subprocess.run", side_effect=fake_run),
            ):
                summary = run_diversity_audit(scenario_index=0, seeds=[0, 1, 2], visualize=False, lightweight=True)

            self.assertFalse(summary["metric_level_diversity_detected"])
            self.assertTrue(summary["trajectory_level_diversity_detected"])
            self.assertTrue(summary["diversity_detected"])
            self.assertEqual(summary["decision"], "trajectory_level_diversity_detected_metric_level_flat")
            self.assertEqual(summary["trajectory_category_spread"]["termination_reason"]["unique_values"], 2)


if __name__ == "__main__":
    unittest.main()
