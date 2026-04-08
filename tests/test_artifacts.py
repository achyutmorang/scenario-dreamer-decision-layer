from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenario_dreamer_decision_layer.artifacts import create_run_bundle, sha256_jsonable


class ArtifactTests(unittest.TestCase):
    def test_create_run_bundle_paths(self) -> None:
        bundle = create_run_bundle("20260408T000000Z", "smoke")
        self.assertTrue(str(bundle["run_dir"]).endswith("20260408T000000Z_smoke"))
        self.assertTrue(str(bundle["metrics"]).endswith("metrics.json"))

    def test_sha256_jsonable_stable(self) -> None:
        payload = {"b": 2, "a": 1}
        self.assertEqual(sha256_jsonable(payload), sha256_jsonable({"a": 1, "b": 2}))


if __name__ == "__main__":
    unittest.main()
