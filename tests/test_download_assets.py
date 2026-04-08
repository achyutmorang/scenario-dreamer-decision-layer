from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load_download_assets_module():
    path = ROOT / "scripts" / "download_assets.py"
    spec = importlib.util.spec_from_file_location("download_assets_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DownloadAssetsTests(unittest.TestCase):
    def test_normalize_checkpoint_layout_moves_single_ckpt_to_expected_path(self) -> None:
        module = _load_download_assets_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = {
                "assets": {
                    "scratch_root": str(root / "artifacts" / "assets" / "scenario-dreamer"),
                    "checkpoint": {
                        "relative_ckpt_path": str(root / "artifacts" / "assets" / "scenario-dreamer" / "checkpoints" / "ctrl_sim_waymo_1M_steps" / "last.ckpt"),
                    },
                }
            }
            downloaded = root / "artifacts" / "assets" / "scenario-dreamer" / "checkpoints" / "last.ckpt"
            downloaded.parent.mkdir(parents=True, exist_ok=True)
            downloaded.write_text("checkpoint-bytes", encoding="utf-8")

            payload = module._normalize_checkpoint_layout(config)
            expected = Path(config["assets"]["checkpoint"]["relative_ckpt_path"])

            self.assertEqual(payload["status"], "moved_to_expected")
            self.assertTrue(expected.exists())
            self.assertFalse(downloaded.exists())

    def test_normalize_env_layout_moves_pickles_and_jsons_to_expected_dirs(self) -> None:
        module = _load_download_assets_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = {
                "assets": {
                    "dataset_root": str(root / "artifacts" / "assets" / "datasets"),
                    "simulation_envs": {
                        "pickles_dir": str(root / "artifacts" / "assets" / "scenario_dreamer_waymo_200m_pickles"),
                        "jsons_dir": str(root / "artifacts" / "assets" / "scenario_dreamer_waymo_200m_jsons"),
                    },
                }
            }
            download_root = Path(config["assets"]["dataset_root"]) / "scenario_dreamer_release" / "extracted"
            source_pickles = download_root / "scenario_dreamer_waymo_200m_pickles"
            source_jsons = download_root / "scenario_dreamer_waymo_200m_jsons"
            source_pickles.mkdir(parents=True, exist_ok=True)
            source_jsons.mkdir(parents=True, exist_ok=True)
            (source_pickles / "scene_a.pkl").write_text("pickle-bytes", encoding="utf-8")
            (source_jsons / "scene_a.json").write_text("{\"scene\": 1}", encoding="utf-8")

            payload = module._normalize_env_layout(config)
            pickles_dir = Path(config["assets"]["simulation_envs"]["pickles_dir"])
            jsons_dir = Path(config["assets"]["simulation_envs"]["jsons_dir"])

            self.assertEqual(payload["status"], "normalized")
            self.assertTrue((pickles_dir / "scene_a.pkl").exists())
            self.assertTrue((jsons_dir / "scene_a.json").exists())


if __name__ == "__main__":
    unittest.main()
