"""Independent memory arithmetic, rejection and shared-KV lifetime checks."""
import copy
import os
from pathlib import Path
import sys
from unittest.mock import patch

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / 'tools/gpt_oss'))
sys.path.insert(0, str(repo / 'python'))
from resident_budget import admit_resident, plan_identity, kernel_environment
from garnet_pipeline import ResidentTensorParallel

plan = dict(schema=3, mode='gpt-oss-tensor-parallel-tp2', cache_key='measured',
    hardware=[{'id': 0}, {'id': 1}], batch=256, capacity=1024, max_tokens=16,
    kv_pages=16384, config=dict(num_hidden_layers=36, head_dim=64), local_kv_heads=4,
    expert_weight_shards=False, moe_intermediate_shards=True, marlin_prepacked=True,
    compact_vocab_greedy=False, marlin_workspace_layout={'max_tokens': 4096},
    collective_workspace_layout='fixture-v10',
    weight_storage_estimate={'prepacked_marlin_constant_bytes': 30_457_036_800})
identity = dict(binaries={'core': 'a', 'plugin': 'b'}, hardware_csv='gpu UUIDs+driver',
    environment={'GARNET_GPT_OSS_MARLIN_PREPACKED': '1'}, checkpoint={'header': 'same'},
    cache=str(repo / 'unused-engine-cache'), padded_prefill=True)
profile = dict(resident_profile_schema=1, native_binaries=identity['binaries'],
    hardware_csv=identity['hardware_csv'], kernel_environment=identity['environment'],
    checkpoint=identity['checkpoint'], cache_root=str(Path(identity['cache']).resolve()),
    padded_prefill=True, plan_identity=plan_identity(plan), engine_statistics=[
        dict(device=rank, phase=phase, total_weights_bytes=35_600_143_104,
             context_device_memory_upper_bound_bytes=context)
        for rank in range(2) for phase, context in [('prefill',448_338_944), ('decode',334_106_112)]])
devices = [dict(id=rank, free_bytes=100_000_000_000, total_bytes=102_000_000_000) for rank in range(2)]
result = admit_resident(profile, plan, devices, **identity)
# Independent integer calculation of ceil(1.05*weights), both contexts, KV and reserve.
expected = (71_200_286_208 * 105 + 99) // 100 + 782_445_056 + 9_663_676_416 + 2_147_483_648
assert all(row['required_bytes'] == expected for row in result['ranks'])

def rejected(candidate=profile, placement=plan, hardware=devices, **changes):
    try:
        admit_resident(candidate, placement, hardware, **dict(identity, **changes))
    except ValueError:
        return
    raise AssertionError('Unsafe resident admission accepted')

for key in ('native_binaries', 'hardware_csv', 'kernel_environment', 'checkpoint',
            'cache_root', 'padded_prefill', 'plan_identity', 'resident_profile_schema'):
    candidate = copy.deepcopy(profile)
    candidate[key] = None
    rejected(candidate)
rejected(placement=dict(plan, marlin_prepacked=False))
rejected(placement=dict(plan, collective_workspace_layout='v9-prefill-only'))
for key,value in [('GARNET_GPT_OSS_MARLIN_PREFILL_DOWN_K','64'),
                  ('GARNET_GPT_OSS_MARLIN_PREFILL_DOWN_CTAS_PER_SM','2'),
                  ('GARNET_GPT_OSS_BF16_DECODE_ALLREDUCE','1')]:
    with patch.dict(os.environ,{key:value}):
        assert kernel_environment()[key]==value
    environment=dict(identity['environment'],**{key:value})
    rejected(environment=environment) # Old profile must not admit new kernel settings.
    candidate=copy.deepcopy(profile)
    candidate['kernel_environment']=environment
    assert admit_resident(candidate,plan,devices,**dict(identity,environment=environment))['ranks']==result['ranks']
rejected(reserve_bytes=(2 << 30) - 1)
rejected(memory_fraction=.91)
rejected(hardware=[dict(d, free_bytes=expected - 1) for d in devices])
for field, value in [('total_weights_bytes', 1), ('context_device_memory_upper_bound_bytes', 0),
                     ('device', 9), ('phase', 'wrong')]:
    candidate = copy.deepcopy(profile)
    candidate['engine_statistics'][0][field] = value
    rejected(candidate)
candidate = copy.deepcopy(profile)
candidate['engine_statistics'][0] = candidate['engine_statistics'][1]
rejected(candidate)

class Fake:
    def __init__(self, kv):
        self.stages = [dict(device_id=rank, keys=k, values=v) for rank, (k,v) in enumerate(kv)]
        self.releases = 0
    def release(self): self.releases += 1
    def forward_rank_local(self, inputs, **kwargs): return inputs

kv = [(object(), object()), (object(), object())]
first, second = Fake(kv), Fake(kv)
pair = ResidentTensorParallel.build(lambda: first, lambda shared: second)
assert pair.forward_prefill('prefill') == 'prefill'
assert pair.forward_decode('decode') == 'decode'
pair.release(); pair.release()
assert first.releases == second.releases == 1
try: pair.forward_decode('released')
except RuntimeError: pass
else: raise AssertionError('Released pair accepted execution')
first = Fake(kv)
def failure(shared): raise RuntimeError('decode failed')
try: ResidentTensorParallel.build(lambda: first, failure)
except RuntimeError: pass
else: raise AssertionError('Build failure hidden')
assert first.releases == 1
first, second = Fake(kv), Fake([(object(), object()), kv[1]])
try: ResidentTensorParallel.build(lambda: first, lambda shared: second)
except ValueError: pass
else: raise AssertionError('Different KV objects accepted')
assert first.releases == second.releases == 1
print('Resident independent admission/identity/rejection/shared-KV/cleanup checks PASS')
