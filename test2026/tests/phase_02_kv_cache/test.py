import sys

import xlang

try:
    garnet = xlang.importModule("garnet")
except Exception as e:
    print(f"Failed to import Garnet via xlang: {e}")
    sys.exit(1)

print("Testing Garnet Paged KV Cache initialization and allocation...")

try:
    # Test the internal API for KV Cache diagnostics
    kv_manager = garnet.KVCacheManager(
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
