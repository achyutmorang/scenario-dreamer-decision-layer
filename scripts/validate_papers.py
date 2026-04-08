#!/usr/bin/env python3
from __future__ import annotations

import json

from _script_utils import ROOT  # noqa: F401
from scenario_dreamer_decision_layer.papers import validate_manifest


def main() -> int:
    payload = validate_manifest()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not payload["missing"] and not payload["mismatched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
