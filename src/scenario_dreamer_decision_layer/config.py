from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return project_root() / "configs" / "baselines" / "scenario_dreamer_ctrlsim.yaml"


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else default_config_path()
    return json.loads(path.read_text(encoding="utf-8"))


def repo_path(*parts: str) -> Path:
    return project_root().joinpath(*parts)


def resolve_repo_relative(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else project_root() / path
