from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenario_dreamer_decision_layer.transfer_hook import build_transfer_request


class TransferHookTests(unittest.TestCase):
    def test_build_transfer_request(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            manifest = run_dir / "run_manifest.json"
            manifest.write_text(json.dumps({
                "run_id": "abc",
                "baseline": {"name": "scenario_dreamer_ctrlsim"}
            }), encoding="utf-8")
            request = build_transfer_request(manifest)
            self.assertEqual(request.source_run_id, "abc")
            self.assertEqual(request.baseline_name, "scenario_dreamer_ctrlsim")


if __name__ == "__main__":
    unittest.main()
