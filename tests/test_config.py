from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenario_dreamer_decision_layer.config import default_config_path, load_config


class ConfigTests(unittest.TestCase):
    def test_config_is_json_compatible_yaml(self) -> None:
        data = json.loads(default_config_path().read_text(encoding="utf-8"))
        self.assertEqual(data["baseline_name"], "scenario_dreamer_ctrlsim")

    def test_load_config(self) -> None:
        cfg = load_config()
        self.assertIn("upstream", cfg)
        self.assertIn("evaluation", cfg)


if __name__ == "__main__":
    unittest.main()
