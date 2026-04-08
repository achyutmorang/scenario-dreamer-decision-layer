#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from _script_utils import ROOT  # noqa: F401
from scenario_dreamer_decision_layer.papers import download_curated_papers


def main() -> int:
    parser = argparse.ArgumentParser(description="Download curated open-access paper PDFs and refresh the manifest.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = download_curated_papers(force=args.force)
    print(json.dumps({"count": len(manifest), "manifest_path": str(ROOT / 'references' / 'papers_manifest.json')}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
