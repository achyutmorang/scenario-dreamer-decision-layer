from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load_runtime_module():
    path = ROOT / "scripts" / "setup_colab_runtime.py"
    spec = importlib.util.spec_from_file_location("setup_colab_runtime_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SetupColabRuntimeTests(unittest.TestCase):
    def test_ensure_upstream_package_markers_creates_init_files(self) -> None:
        module = _load_runtime_module()
        with tempfile.TemporaryDirectory() as td:
            upstream = Path(td) / "external" / "scenario-dreamer"
            for relative in [
                "datasets/waymo",
                "datasets/nuplan",
                "datamodules/waymo",
                "datamodules/nuplan",
                "models",
                "policies",
                "nn_modules",
                "cfgs",
            ]:
                (upstream / relative).mkdir(parents=True, exist_ok=True)

            created = module._ensure_upstream_package_markers(upstream)

            self.assertTrue(created)
            self.assertTrue((upstream / "datasets" / "__init__.py").exists())
            self.assertTrue((upstream / "datasets" / "waymo" / "__init__.py").exists())
            self.assertTrue((upstream / "datamodules" / "__init__.py").exists())
            self.assertTrue((upstream / "models" / "__init__.py").exists())
            self.assertTrue((upstream / "cfgs" / "__init__.py").exists())


if __name__ == "__main__":
    unittest.main()
