"""Checkpoint weight layout, lazy packing, cache identity and VRAM admission."""
import os
import sys
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/gpt_oss'))
from pipeline import (make_tensor_parallel_plan, estimate_tp2_moe_workspace_bytes,
                      estimate_tp2_marlin_workspace_bytes)

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
os.environ['GARNET_GPT_OSS_MARLIN_MAX_TOKENS'] = '8'
os.environ['GARNET_GPT_OSS_MARLIN_PREFILL_BLOCK'] = '8'
small_layout = make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)
os.environ['GARNET_GPT_OSS_MARLIN_MAX_TOKENS'] = '4096'
large_layout = make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)
assert small_layout['cache_key'] != large_layout['cache_key']
os.environ['GARNET_GPT_OSS_MARLIN_PREFILL_BLOCK'] = '32'
large_tile = make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)
assert large_tile['cache_key'] != large_layout['cache_key']
if len(sys.argv) > 2:
    config = dict(hidden_size=2880, intermediate_size=2880, num_experts=128, experts_per_token=8)
    # The compiled plugin's actual allocation methods are the authority; this
    # checks the Python estimator across dispatch and tile-size boundaries.
    for block in ('8', '32'):
        os.environ['GARNET_GPT_OSS_MARLIN_PREFILL_BLOCK'] = block
        report = subprocess.check_output([sys.argv[2], '--workspace'], text=True)
        for line in report.splitlines():
            rows, grouped, marlin = map(int, line.split())
            assert estimate_tp2_moe_workspace_bytes(config, rows) == grouped, (rows, grouped)
            assert estimate_tp2_marlin_workspace_bytes(config, rows) == marlin, (rows, marlin)
budget = sharded['estimated_per_gpu_bytes'] - 1
for device in devices:
    device['free_bytes'] = budget
try:
    make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)
    raise AssertionError('planner admitted a shape exceeding available VRAM')
except ValueError:
    pass
print('TP2 expert-storage and memory admission tests passed')
