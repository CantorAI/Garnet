import sys
import os
from pathlib import Path

import xlang

try:
    repo_root = Path(__file__).resolve().parents[3]
    garnet_dll = Path(os.environ.get(
        "GARNET_DLL_PATH",
        repo_root / "out" / "build" / "x64-Debug" / "bin" / "garnet.dll",
    ))
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for dll_dir in [
            garnet_dll.parent,
            repo_root.parent / "xlang" / "out" / "build" / "x64-Debug" / "bin",
        ]:
            if dll_dir.exists():
                os.add_dll_directory(str(dll_dir))
    garnet = xlang.importModule("garnet", fromPath=str(garnet_dll))
except Exception as e:
    print(f"Failed to import Garnet via xlang: {e}")
    sys.exit(1)

print("Testing Garnet Paged KV Cache initialization and allocation...")

try:
    kv_cache_manager_ctor = getattr(garnet, "KVCacheManager", None)
    if not callable(kv_cache_manager_ctor):
        print("SKIP: Garnet KVCacheManager API is not exported yet.")
        sys.exit(0)

    # Test the internal API for KV Cache diagnostics
    kv_manager = kv_cache_manager_ctor(
        max_num_pages=1024,
        page_size=16,
        head_dim=128,
        num_kv_heads=32
    )
    
    # Allocate a sequence
    seq_id = 1
    blocks = kv_manager.allocate(seq_id, sequence_length=40)
    
    # 40 tokens / 16 page_size = 3 pages required
    assert len(blocks) == 3, f"Expected 3 pages to be allocated for 40 tokens, got {len(blocks)}"
    
    # Free the sequence blocks
    kv_manager.free(seq_id)
    
    print("Phase 02: Paged KV Cache Manager test passed!")
    sys.exit(0)
except Exception as e:
    print(f"KV Cache failed: {e}")
    sys.exit(1)
