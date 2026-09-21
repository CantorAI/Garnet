# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def add_dll_dirs():
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    for dll_dir in [
        REPO_ROOT / "out" / "build" / "x64-Debug" / "bin",
        REPO_ROOT.parent / "xlang" / "out" / "build" / "x64-Debug" / "bin",
        REPO_ROOT.parent / "out" / "build" / "x64-Debug" / "bin",
        REPO_ROOT.parent / "out" / "build" / "x64-debug" / "bin",
        REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "bin",
        REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "lib",
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin"),
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64"),
        Path("C:/Program Files/Microsoft Visual Studio/18/Community/VC/Redist/MSVC/14.51.36231/debug_nonredist/x64/Microsoft.VC145.DebugCRT"),
    ]:
        if dll_dir.exists():
            os.add_dll_directory(str(dll_dir))


def main():
    add_dll_dirs()
    try:
        import xlang3
    except Exception as exc:
        print(f"SKIP: xlang is not available: {exc}")
        raise SystemExit(0)

    garnet_dll = REPO_ROOT / "out" / "build" / "x64-Debug" / "bin" / "garnet.dll"
    if not garnet_dll.exists():
        print(f"SKIP: garnet.dll not found: {garnet_dll}")
        raise SystemExit(0)

    garnet = xlang3.importModule("garnet", fromPath=str(garnet_dll))
    kv = garnet.KVCacheManager(
        max_num_pages=16,
        page_size=8,
        head_dim=128,
        num_kv_heads=2,
        num_layers=4,
        dtype_bytes=2,
        device_id=0,
    )
    stats0 = kv.stats()
    assert stats0["gpu_allocated"] == 1
    assert stats0["free_pages"] == 16
    assert stats0["total_key_bytes"] == 262144
    assert stats0["total_value_bytes"] == 262144

    pages = kv.allocate(7, sequence_length=17)
    assert list(pages) == [0, 1, 2]
    stats1 = kv.stats()
    assert stats1["used_pages"] == 3
    assert stats1["sequence_count"] == 1

    append1 = kv.append(7, tokens=1)
    assert append1["logical_length"] == 18
    assert append1["page_count"] == 3

    append2 = kv.append(7, tokens=10)
    assert append2["logical_length"] == 28
    assert append2["page_count"] == 4
    assert kv.stats()["used_pages"] == 4

    assert kv.free(7) == 1
    stats2 = kv.stats()
    assert stats2["used_pages"] == 0
    assert stats2["free_pages"] == 16
    assert stats2["sequence_count"] == 0
    print("Phase 13: production-shaped KV cache manager passed.")


if __name__ == "__main__":
    main()
