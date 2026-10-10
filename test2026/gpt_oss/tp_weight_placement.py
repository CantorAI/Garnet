"""Checkpoint weight layout, lazy packing, cache identity and VRAM admission."""
import os
import sys
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/gpt_oss'))
from pipeline import (make_tensor_parallel_plan, estimate_tp2_moe_workspace_bytes,
                      estimate_tp2_marlin_workspace_bytes, marlin_workspace_profile,
                      collective_workspace_profile, build_tensor_parallel)

fixture = Path(sys.argv[1])
devices = [dict(id=i, name='synthetic', total_bytes=1 << 30,
               free_bytes=1 << 30, compute_major=12, compute_minor=0,
               pci_bus_id=str(i), peer_access=[True, True]) for i in range(2)]
os.environ.pop('GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS', None)
os.environ.pop('GARNET_GPT_OSS_TP_MOE_INTERMEDIATE_SHARDS', None)
os.environ.pop('GARNET_GPT_OSS_MARLIN_PREPACKED', None)
os.environ.pop('GARNET_GPT_OSS_MARLIN_PREFILL_UP_CTAS_PER_SM', None)
os.environ.pop('GARNET_GPT_OSS_MARLIN_PREFILL_UP_STAGES', None)
os.environ.pop('GARNET_GPT_OSS_MARLIN_PREFILL_DOWN_CTAS_PER_SM', None)
os.environ.pop('GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK', None)
os.environ.pop('GARNET_GPT_OSS_MARLIN_BOUNDED_PREFILL', None)
full = make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)
assert full['collective_workspace_layout'] == collective_workspace_profile()
# Old collective layouts reject before any native builder/allocation, even
# with a correct Marlin layout. Candidate flag cannot change engine allocation.
from unittest.mock import patch
import types
for layout in (None, 'v9-prefill-only'):
    stale=dict(full)
    if layout is None: stale.pop('collective_workspace_layout')
    else: stale['collective_workspace_layout']=layout
    with patch.dict(sys.modules, {'garnet':types.SimpleNamespace()}):
        try:
            build_tensor_parallel(fixture,'unused-cache',stale,1,False)
            raise AssertionError('Old collective workspace admitted')
        except ValueError as error:
            assert 'Collective workspace' in str(error)
with patch.dict(os.environ, {'GARNET_GPT_OSS_BF16_DECODE_ALLREDUCE':'1'}):
    assert make_tensor_parallel_plan(fixture,devices,reserve_bytes=0)['cache_key']==full['cache_key']
with patch('pipeline.collective_workspace_profile',return_value='v9-prefill-only'):
    assert make_tensor_parallel_plan(fixture,devices,reserve_bytes=0)['cache_key']!=full['cache_key']
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
# Independent fixture byte count: E5, H32, I32, two layers. Rank0 owns3
# experts; biases/dense constants remain and must not be subtracted.
os.environ['GARNET_GPT_OSS_MARLIN_PREPACKED'] = '1'
prepacked = make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)
packed_storage = prepacked['weight_storage_estimate']
removed = 2 * 3 * (64 * 32 + 32 * 32) * 17 // 32
assert packed_storage['original_quant_constants_eliminated_bytes'] == removed
assert packed_storage['original_constants_estimated_bytes'] == storage['original_constants_estimated_bytes'] - removed
assert packed_storage['prepacked_marlin_constant_bytes'] == storage['lazy_marlin_repacked_bytes']
assert not packed_storage['includes_lazy_marlin_repacking']
assert prepacked['estimated_per_gpu_bytes'] < sharded['estimated_per_gpu_bytes']
assert prepacked['cache_key'] != sharded['cache_key']
for device in devices:
    device['free_bytes'] = prepacked['estimated_per_gpu_bytes']
make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)
os.environ['GARNET_GPT_OSS_MARLIN_PREPACKED'] = '0'
try:
    make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)
    raise AssertionError('Original plus lazy-packed storage exceeded the available budget')
except ValueError:
    pass
for device in devices:
    device['free_bytes'] = device['total_bytes']
os.environ['GARNET_GPT_OSS_MARLIN_PREPACKED'] = 'invalid'
try:
    make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)
    raise AssertionError('Unsupported storage flag was accepted')
except ValueError:
    pass
os.environ['GARNET_GPT_OSS_MARLIN_PREPACKED'] = '1'
os.environ['GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS'] = '0'
try:
    make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)
    raise AssertionError('Unpartitioned TP2 prepacking was accepted')
except ValueError:
    pass
os.environ.pop('GARNET_GPT_OSS_MARLIN_PREPACKED', None)
os.environ['GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS'] = '1'
print('Prepacked/original byte accounting, cache separation and admission gates passed')
os.environ['GARNET_GPT_OSS_MARLIN_MAX_TOKENS'] = '8'
os.environ['GARNET_GPT_OSS_MARLIN_PREFILL_BLOCK'] = '8'
os.environ.pop('GARNET_GPT_OSS_MARLIN_DECODE_BLOCK', None)
small_layout = make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)
os.environ['GARNET_GPT_OSS_MARLIN_MAX_TOKENS'] = '4096'
large_layout = make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)
assert small_layout['cache_key'] != large_layout['cache_key']
os.environ['GARNET_GPT_OSS_MARLIN_PREFILL_BLOCK'] = '32'
large_tile = make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)
assert large_tile['cache_key'] != large_layout['cache_key']
os.environ['GARNET_GPT_OSS_MARLIN_DECODE_BLOCK'] = '8'
decode_tile = make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)
assert decode_tile['cache_key'] != large_tile['cache_key']
assert decode_tile['marlin_workspace_layout']['decode_block_override'] == 8
default_profile = marlin_workspace_profile()
config = dict(hidden_size=2880, intermediate_size=2880, num_experts=128, experts_per_token=8)
default_workspaces = {(rows, phase): estimate_tp2_marlin_workspace_bytes(config, rows, prefill=phase)
                      for rows in (128, 1023, 1024, 4096, 8020) for phase in (False, True)}
os.environ['GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK'] = '64'
large64 = make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)
assert large64['cache_key'] != decode_tile['cache_key']
assert marlin_workspace_profile()['large_prefill_block'] == 64
large64_workspaces = {(rows, phase): estimate_tp2_marlin_workspace_bytes(config, rows, prefill=phase)
                      for rows in (128, 1024, 4096) for phase in (False, True)}
for (rows, phase), size in default_workspaces.items():
    actual = estimate_tp2_marlin_workspace_bytes(config, rows, prefill=phase)
    if phase and rows in (1024, 4096):
        assert actual != size, (rows, phase)
    else:
        assert actual == size, (rows, phase)
os.environ['GARNET_GPT_OSS_MARLIN_PREFILL_UP_CTAS_PER_SM'] = '2'
up_candidate = make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)
assert up_candidate['cache_key'] != large64['cache_key']
assert up_candidate['marlin_workspace_layout']['prefill_up_ctas_per_sm'] == 2
for key in ((128, False), (128, True), (1024, False), (1024, True), (4096, False), (4096, True)):
    assert estimate_tp2_marlin_workspace_bytes(config, key[0], prefill=key[1]) == large64_workspaces[key]
os.environ['GARNET_GPT_OSS_MARLIN_PREFILL_UP_STAGES'] = '2'
up_stage_candidate = make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)
assert up_stage_candidate['cache_key'] != up_candidate['cache_key']
assert up_stage_candidate['marlin_workspace_layout']['prefill_up_stages'] == 2
os.environ['GARNET_GPT_OSS_MARLIN_PREFILL_UP_STAGES'] = '3'
try:
    marlin_workspace_profile()
    raise AssertionError('Malformed up-stage policy was accepted')
except ValueError:
    pass
os.environ.pop('GARNET_GPT_OSS_MARLIN_PREFILL_UP_STAGES')
os.environ['GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK'] = '0'
os.environ['GARNET_GPT_OSS_MARLIN_PREFILL_UP_STAGES'] = '2'
try:
    marlin_workspace_profile()
    raise AssertionError('Up-stage policy without large-prefill64 was accepted')
except ValueError:
    pass
os.environ.pop('GARNET_GPT_OSS_MARLIN_PREFILL_UP_STAGES')
os.environ['GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK'] = '64'
os.environ['GARNET_GPT_OSS_MARLIN_PREFILL_DOWN_CTAS_PER_SM'] = '1'
split_candidate = make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)
assert split_candidate['cache_key'] != up_candidate['cache_key']
assert split_candidate['marlin_workspace_layout']['prefill_down_ctas_per_sm'] == 1
os.environ['GARNET_GPT_OSS_MARLIN_PREFILL_DOWN_CTAS_PER_SM'] = '3'
try:
    marlin_workspace_profile()
    raise AssertionError('Malformed projection CTA policy was accepted')
except ValueError:
    pass
os.environ.pop('GARNET_GPT_OSS_MARLIN_PREFILL_UP_CTAS_PER_SM')
os.environ.pop('GARNET_GPT_OSS_MARLIN_PREFILL_DOWN_CTAS_PER_SM')
os.environ['GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK'] = '0'
assert marlin_workspace_profile() == default_profile
assert make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)['cache_key'] == decode_tile['cache_key']
os.environ.pop('GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK')
os.environ['GARNET_GPT_OSS_MARLIN_BOUNDED_PREFILL']='1'
bounded_plan=make_tensor_parallel_plan(fixture,devices,reserve_bytes=0)
assert bounded_plan['cache_key']!=decode_tile['cache_key']
assert bounded_plan['marlin_workspace_layout']['bounded_prefill_subcall_rows']==4096
for rows in (4097,4608,7168,8020,8192):
    assert estimate_tp2_marlin_workspace_bytes(config,rows)==estimate_tp2_marlin_workspace_bytes(config,4096)
    assert estimate_tp2_marlin_workspace_bytes(config,rows,prefill=False)==0
assert estimate_tp2_marlin_workspace_bytes(config,8193)==0
os.environ['GARNET_GPT_OSS_MARLIN_BOUNDED_PREFILL']='0'
assert make_tensor_parallel_plan(fixture,devices,reserve_bytes=0)['cache_key']==decode_tile['cache_key']
if len(sys.argv) > 2:
    config = dict(hidden_size=2880, intermediate_size=2880, num_experts=128, experts_per_token=8)
    # The compiled plugin's actual allocation methods are the authority; this
    # checks the Python estimator across dispatch and tile-size boundaries.
    for large in ('0', '64'):
        os.environ['GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK'] = large
        for block in ('8', '32'):
            os.environ['GARNET_GPT_OSS_MARLIN_PREFILL_BLOCK'] = block
            for decode_block in ('0', '8', '32'):
                os.environ['GARNET_GPT_OSS_MARLIN_DECODE_BLOCK'] = decode_block
                report = subprocess.check_output([sys.argv[2], '--workspace'], text=True)
                for line in report.splitlines():
                    rows, grouped, marlin, decode = map(int, line.split())
                    assert estimate_tp2_moe_workspace_bytes(config, rows) == grouped, (rows, grouped)
                    assert estimate_tp2_marlin_workspace_bytes(config, rows) == marlin, (rows, marlin)
                    assert estimate_tp2_marlin_workspace_bytes(config, rows, prefill=False) == decode, (rows, decode)
    os.environ.pop('GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK')
    os.environ['GARNET_GPT_OSS_MARLIN_BOUNDED_PREFILL']='1'
    for block in ('32','64'):
        os.environ['GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK']=block
        report=subprocess.check_output([sys.argv[2],'--workspace'],text=True)
        for line in report.splitlines():
            rows,grouped,marlin,decode=map(int,line.split())
            assert estimate_tp2_moe_workspace_bytes(config,rows)==grouped
            assert estimate_tp2_marlin_workspace_bytes(config,rows)==marlin,(rows,marlin)
            assert estimate_tp2_marlin_workspace_bytes(config,rows,prefill=False)==decode,(rows,decode)
    os.environ.pop('GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK')
    os.environ['GARNET_GPT_OSS_MARLIN_BOUNDED_PREFILL']='0'
budget = sharded['estimated_per_gpu_bytes'] - 1
for device in devices:
    device['free_bytes'] = budget
try:
    make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)
    raise AssertionError('planner admitted a shape exceeding available VRAM')
except ValueError:
    pass
print('TP2 expert-storage and memory admission tests passed')

os.environ.pop('GARNET_GPT_OSS_COMPACT_VOCAB_GREEDY', None)
for device in devices:
    device['free_bytes'] = device['total_bytes']
regular_vocab = make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)
os.environ['GARNET_GPT_OSS_COMPACT_VOCAB_GREEDY'] = '1'
compact_vocab = make_tensor_parallel_plan(fixture, devices, reserve_bytes=0)
assert compact_vocab['compact_vocab_greedy'] and not regular_vocab['compact_vocab_greedy']
assert compact_vocab['cache_key'] != regular_vocab['cache_key']
os.environ.pop('GARNET_GPT_OSS_COMPACT_VOCAB_GREEDY', None)
print('Compact/full vocabulary profiles have distinct cache identities')
