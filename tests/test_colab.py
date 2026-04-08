from __future__ import annotations

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

from scenario_dreamer_decision_layer.colab import bind_drive_layout, default_drive_layout, seed_env_pack_from_upstream


class ColabBindingTests(unittest.TestCase):
    def test_default_drive_layout(self) -> None:
        layout = default_drive_layout("/tmp/drive-root")
        self.assertTrue(str(layout["scratch_root"]).endswith("scenario_dreamer_decision_layer/assets/scenario-dreamer"))
        self.assertTrue(str(layout["results_root"]).endswith("scenario_dreamer_decision_layer/results/runs"))

    def test_bind_drive_layout_sets_results_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp_root = Path(td).resolve()
            canonical_root = temp_root / "canonical"
            config = {
                "assets": {
                    "scratch_root": str(canonical_root / "scenario-dreamer"),
                    "dataset_root": str(canonical_root / "datasets"),
                    "simulation_envs": {
                        "pickles_dir": str(canonical_root / "pickles"),
                        "jsons_dir": str(canonical_root / "jsons"),
                    },
                }
            }
            with patch("scenario_dreamer_decision_layer.colab.load_config", return_value=config):
                payload = bind_drive_layout(temp_root / "drive")
            self.assertEqual(payload["bindings"]["scratch_root"]["status"], "bound")
            self.assertTrue((canonical_root / "scenario-dreamer").is_symlink())
            self.assertTrue((canonical_root / "pickles").is_symlink())
            self.assertEqual(os.environ["SCENARIO_DREAMER_RESULTS_ROOT"], str(temp_root / "drive" / "scenario_dreamer_decision_layer" / "results" / "runs"))

    def test_seed_env_pack_from_upstream_copies_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp_root = Path(td).resolve()
            upstream = temp_root / "external" / "scenario-dreamer" / "metadata" / "simulation_environment_datasets"
            (upstream / "scenario_dreamer_waymo_200m_pickles").mkdir(parents=True, exist_ok=True)
            (upstream / "scenario_dreamer_waymo_200m_jsons").mkdir(parents=True, exist_ok=True)
            (upstream / "scenario_dreamer_waymo_200m_pickles" / "0_0.pkl").write_text("pickle", encoding="utf-8")
            (upstream / "scenario_dreamer_waymo_200m_jsons" / "0_0.json").write_text("{}", encoding="utf-8")

            target_root = temp_root / "bound"
            config = {
                "upstream": {"repo_dir": str(temp_root / "external" / "scenario-dreamer")},
                "assets": {
                    "scratch_root": str(target_root / "scenario-dreamer"),
                    "dataset_root": str(target_root / "datasets"),
                    "simulation_envs": {
                        "pickles_dir": str(target_root / "scenario_dreamer_waymo_200m_pickles"),
                        "jsons_dir": str(target_root / "scenario_dreamer_waymo_200m_jsons"),
                    },
                },
            }
            with patch("scenario_dreamer_decision_layer.colab.load_config", return_value=config):
                payload = seed_env_pack_from_upstream()

            self.assertEqual(payload["status"], "seeded")
            self.assertTrue((target_root / "scenario_dreamer_waymo_200m_pickles" / "0_0.pkl").exists())
            self.assertTrue((target_root / "scenario_dreamer_waymo_200m_jsons" / "0_0.json").exists())


if __name__ == "__main__":
    unittest.main()
