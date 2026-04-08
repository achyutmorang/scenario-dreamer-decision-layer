#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

from _script_utils import ROOT


SIMULATION_PACKAGES = [
    "hydra-core>=1.3,<1.4",
    "moviepy==1.0.3",
    "wandb>=0.18,<0.19",
    "tqdm>=4.66,<5",
    "matplotlib>=3.8,<4",
    "scipy>=1.10,<1.16",
    "numpy>=1.26,<2.3",
    "torch-ema==0.3",
    "pytorch-lightning>=2.4,<2.6",
    "networkx>=3.4,<4",
    "shapely>=2.0,<2.1",
    "einops>=0.8,<0.9",
    "Pillow>=9.2,<11",
    "opencv-python>=4.10,<5",
    "gdown>=5.2,<6",
]

PYG_BASE_PACKAGES = [
    "torch_scatter",
    "torch_cluster",
    "pyg_lib",
    "torch_sparse",
    "torch_geometric",
]


def _run(cmd: Sequence[str]) -> None:
    print("[setup-colab-runtime] $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _missing_modules(modules: Iterable[str]) -> list[str]:
    missing = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    return missing


def _torch_info() -> tuple[str, str]:
    import torch

    torch_version = torch.__version__.split("+", 1)[0]
    cuda = getattr(torch.version, "cuda", None)
    cuda_tag = "cpu" if not cuda else f"cu{cuda.replace('.', '')}"
    return torch_version, cuda_tag


def _install_pyg() -> None:
    missing = _missing_modules(["torch_geometric", "torch_scatter", "torch_sparse", "torch_cluster"])
    if not missing:
        return
    torch_version, cuda_tag = _torch_info()
    wheel_index = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_tag}.html"
    print(f"[setup-colab-runtime] using PyG wheel index: {wheel_index}")
    for package in PYG_BASE_PACKAGES:
        _run([sys.executable, "-m", "pip", "install", "--upgrade", package, "-f", wheel_index])


def main() -> int:
    parser = argparse.ArgumentParser(description="Install a lean Colab runtime for Scenario Dreamer baseline simulation.")
    parser.add_argument("--editable-project", action="store_true", help="Install the current repo in editable mode.")
    args = parser.parse_args()

    if _missing_modules(["torch"]):
        raise RuntimeError("Torch is not installed in this runtime. Use a GPU Colab runtime with PyTorch preinstalled.")

    pure_missing = _missing_modules(
        [
            "hydra",
            "moviepy",
            "wandb",
            "tqdm",
            "matplotlib",
            "scipy",
            "numpy",
            "pytorch_lightning",
            "networkx",
            "shapely",
            "einops",
            "PIL",
            "cv2",
            "gdown",
        ]
    )
    if pure_missing:
        _run([sys.executable, "-m", "pip", "install", "--upgrade", *SIMULATION_PACKAGES])

    _install_pyg()

    if args.editable_project:
        _run([sys.executable, "-m", "pip", "install", "-e", str(ROOT)])

    print("[setup-colab-runtime] runtime ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
