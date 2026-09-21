# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path


PHASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PHASE_DIR.parents[2]
DATASET_DIR = REPO_ROOT / "data" / "Dataset.1980Love"
IMAGE_DIR = DATASET_DIR / "imgs"


def first_dataset_sample():
    image_path = next(iter(sorted(IMAGE_DIR.glob("*.jpg"))), None)
    if image_path is None:
        raise FileNotFoundError(f"No JPG files found in {IMAGE_DIR}")

    meta_path = image_path.with_suffix(".json")
    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    return image_path, metadata


def qwen_prompt(metadata):
    boxes = metadata.get("boxes") or []
    return (
        "Describe this movie frame briefly. "
        f"The detector metadata contains {len(boxes)} boxes; mention visible people, objects, and scene context."
    )


def env_flag(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def skip(message):
    print(f"SKIP: {message}")
    raise SystemExit(0)


def discover_garnet_dll():
    explicit = os.environ.get("GARNET_DLL_PATH", "").strip()
    candidates = []
    if explicit:
        candidates.append(Path(explicit))

    candidates.extend([
        REPO_ROOT / "out" / "build" / "x64-Debug" / "bin" / "garnet.dll",
        REPO_ROOT / "out" / "build" / "x64-Release" / "bin" / "garnet.dll",
        REPO_ROOT / "out" / "install" / "x64-Debug" / "bin" / "garnet.dll",
        REPO_ROOT / "out" / "install" / "x64-Release" / "bin" / "garnet.dll",
        REPO_ROOT.parent / "out" / "build" / "x64-Debug" / "bin" / "garnet.dll",
        REPO_ROOT.parent / "out" / "build" / "x64-Release" / "bin" / "garnet.dll",
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def add_windows_dll_dirs(garnet_dll):
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return []

    handles = []
    candidate_dirs = [
        garnet_dll.parent if garnet_dll else None,
        REPO_ROOT / "out" / "build" / "x64-Debug" / "bin",
        REPO_ROOT / "out" / "build" / "x64-Release" / "bin",
        REPO_ROOT.parent / "xlang" / "out" / "build" / "x64-Debug" / "bin",
        REPO_ROOT.parent / "xlang" / "out" / "build" / "x64-Release" / "bin",
    ]

    for path in candidate_dirs:
        if path and path.exists():
            handles.append(os.add_dll_directory(str(path)))
    return handles
