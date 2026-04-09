from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenario_dreamer_decision_layer.runner import _select_pickle_files, run_diversity_audit, run_risk_variance_study


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
                output_path = Path(cmd[cmd.index("--output-json") + 1])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(
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
                    ),
                    encoding="utf-8",
                )

                class Result:
                    returncode = 0
                    stderr = ""

                    def __init__(self):
                        self.stdout = ""

                return Result()

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
                output_path = Path(cmd[cmd.index("--output-json") + 1])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(
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
                    ),
                    encoding="utf-8",
                )

                class Result:
                    returncode = 0
                    stderr = ""

                    def __init__(self):
                        self.stdout = ""

                return Result()

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

    def test_run_diversity_audit_fails_if_worker_exits_without_payload_file(self) -> None:
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

            class Result:
                returncode = 0
                stderr = ""
                stdout = ""

            with (
                patch.dict(os.environ, {"SCENARIO_DREAMER_RESULTS_ROOT": str(temp_root / "results" / "runs")}, clear=False),
                patch("scenario_dreamer_decision_layer.runner.load_config", return_value=config),
                patch("scenario_dreamer_decision_layer.runner.project_root", return_value=temp_root),
                patch("scenario_dreamer_decision_layer.runner.subprocess.run", return_value=Result()),
            ):
                with self.assertRaisesRegex(RuntimeError, "did not produce a payload file"):
                    run_diversity_audit(scenario_index=0, seeds=[0], visualize=False, lightweight=True)

    def test_run_risk_variance_study_aggregates_scene_risk_and_selector_probe(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp_root = Path(td)

            def fake_audit(*, scenario_index, seeds, visualize, lightweight):
                risk_by_seed = {
                    0: [0.20, 0.35, 0.30],
                    1: [0.40, 0.25, 0.50],
                }[scenario_index]
                runs = []
                for seed, risk in zip(seeds, risk_by_seed):
                    runs.append(
                        {
                            "seed": seed,
                            "metrics": {
                                "collision_rate": 0.0,
                                "off_route_rate": 0.0,
                                "completed_rate": 1.0,
                                "progress": 100.0,
                            },
                            "trajectory_summary": {
                                "trajectory_metrics": {
                                    "min_ttc_proxy_s": risk,
                                    "min_ego_agent_distance_m": risk + 1.0,
                                },
                                "trajectory_categories": {
                                    "termination_reason": "completed",
                                },
                            },
                        }
                    )
                return {
                    "run_id": f"scene_{scenario_index}",
                    "run_dir": str(temp_root / f"scene_{scenario_index}"),
                    "scenario_name": f"scene_{scenario_index}.pkl",
                    "metric_level_diversity_detected": False,
                    "trajectory_level_diversity_detected": True,
                    "decision": "trajectory_level_diversity_detected_metric_level_flat",
                    "metric_spread": {},
                    "trajectory_metric_spread": {
                        "min_ttc_proxy_s": {
                            "min": min(risk_by_seed),
                            "max": max(risk_by_seed),
                            "range": max(risk_by_seed) - min(risk_by_seed),
                            "unique_values": len(set(risk_by_seed)),
                        }
                    },
                    "trajectory_category_spread": {
                        "termination_reason": {
                            "unique_values": 1,
                            "values": ["completed"],
                        }
                    },
                    "runs": runs,
                }

            with (
                patch.dict(os.environ, {"SCENARIO_DREAMER_RESULTS_ROOT": str(temp_root / "results" / "runs")}, clear=False),
                patch("scenario_dreamer_decision_layer.runner.run_diversity_audit", side_effect=fake_audit),
            ):
                summary = run_risk_variance_study(
                    scenario_indices=[0, 1],
                    seeds=[0, 1, 2],
                    selector_k_values=[1, 2, 3],
                    risk_key="min_ttc_proxy_s",
                    visualize=False,
                    lightweight=True,
                )

            self.assertEqual(summary["scene_count"], 2)
            self.assertEqual(summary["risk_key"], "min_ttc_proxy_s")
            self.assertEqual(summary["selector_summary"]["1"]["num_improved"], 0)
            self.assertEqual(summary["selector_summary"]["2"]["num_improved"], 1)
            self.assertAlmostEqual(summary["selector_summary"]["3"]["mean_risk_improvement"], 0.125)
            self.assertTrue((Path(summary["run_dir"]) / "risk_variance_study_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
