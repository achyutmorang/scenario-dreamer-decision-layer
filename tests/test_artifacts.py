from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenario_dreamer_decision_layer.artifacts import create_run_bundle, runs_root, sha256_jsonable


class ArtifactTests(unittest.TestCase):
    def test_create_run_bundle_paths(self) -> None:
        bundle = create_run_bundle("20260408T000000Z", "smoke")
        self.assertTrue(str(bundle["run_dir"]).endswith("20260408T000000Z_smoke"))
        self.assertTrue(str(bundle["metrics"]).endswith("metrics.json"))

    def test_sha256_jsonable_stable(self) -> None:
        payload = {"b": 2, "a": 1}
        self.assertEqual(sha256_jsonable(payload), sha256_jsonable({"a": 1, "b": 2}))

    def test_results_root_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            old = os.environ.get("SCENARIO_DREAMER_RESULTS_ROOT")
            os.environ["SCENARIO_DREAMER_RESULTS_ROOT"] = str(Path(td) / "runs")
            try:
                root = runs_root()
                bundle = create_run_bundle("20260408T000001Z", "dev")
            finally:
                if old is None:
                    os.environ.pop("SCENARIO_DREAMER_RESULTS_ROOT", None)
                else:
                    os.environ["SCENARIO_DREAMER_RESULTS_ROOT"] = old
            self.assertEqual(root, Path(td) / "runs")
            self.assertEqual(bundle["run_dir"].parent, Path(td) / "runs")


if __name__ == "__main__":
    unittest.main()
