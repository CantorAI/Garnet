"""Checkpoint weight layout, lazy packing, cache identity and VRAM admission."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/gpt_oss'))
from pipeline import make_tensor_parallel_plan

fixture = Path(sys.argv[1])
devices = [dict(id=i, name='synthetic', total_bytes=1 << 30,
               free_bytes=1 << 30, compute_major=12, compute_minor=0,
               pci_bus_id=str(i), peer_access=[True, True]) for i in range(2)]
os.environ.pop('GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS', None)
full = make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)
os.environ['GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS'] = '1'
sharded = make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)
assert sharded['expert_weight_shards'] and not full['expert_weight_shards']
assert sharded['cache_key'] != full['cache_key']
storage = sharded['weight_storage_estimate']
# Fixture: 2 layers, 5 experts -> worst rank owns 3. Padded up 64x128,
# down 128x64, with MXFP4 nibbles plus one E8M0 byte per32 weights.
assert storage['lazy_marlin_repacked_bytes'] == 2 * 3 * (64 * 128 + 128 * 64) * 17 // 32
assert storage['original_constants_estimated_bytes'] < full['weight_storage_estimate']['original_constants_estimated_bytes']
assert storage['includes_lazy_marlin_repacking']
budget = sharded['estimated_per_gpu_bytes'] - 1
for device in devices:
    device['free_bytes'] = budget
try:
    make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)
    raise AssertionError('planner admitted a shape exceeding available VRAM')
except ValueError:
    pass
print('TP2 expert-storage and memory admission tests passed')
