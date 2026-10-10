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

# A budget that fits the original pair must reject external peer storage.
from peer_group_layout import peer_group_layout
with patch.dict(os.environ, {'GARNET_GPT_OSS_BF16_PEER_GROUP':'1'}):
    storage = peer_group_layout(256,16,2880)
assert storage['owned_bytes_per_rank'] == 23_601_408
assert storage['mapped_host_bytes'] == 32_768
assert storage['ctas'] == 64 and storage['schema'] == 2
group_plan = copy.deepcopy(plan)
group_plan['config']['hidden_size'] = 2880
group_plan['operator_execution_layout'] = storage
group_profile = copy.deepcopy(profile)
group_profile['plan_identity'] = plan_identity(group_plan)
extra = 23_601_408 + 32_768
group_result = admit_resident(group_profile,group_plan,devices,**identity)
assert all(row['required_bytes']==expected+extra and row['operator_execution_storage_bytes']==extra
           and row['runtime_graph_reserve_bytes']==2<<30 for row in group_result['ranks'])
try:
    admit_resident(group_profile,group_plan,[dict(d,free_bytes=expected+extra-1) for d in devices],**identity)
except ValueError: pass
else: raise AssertionError('External arena silently borrowed runtime reserve')
with patch.dict(os.environ, {'GARNET_GPT_OSS_BF16_PEER_GROUP':'1',
                             'GARNET_GPT_OSS_BF16_PEER_GROUP_CTAS':'128'}):
    storage128 = peer_group_layout(256,16,2880)
assert storage128['ctas'] == 128
assert storage128['owned_bytes_per_rank'] == 23_609_856
assert storage128['mapped_host_bytes'] == 65_536
group_plan128 = copy.deepcopy(plan)
group_plan128['config']['hidden_size'] = 2880
group_plan128['operator_execution_layout'] = storage128
group_profile128 = copy.deepcopy(profile)
group_profile128['plan_identity'] = plan_identity(group_plan128)
extra128 = 23_609_856 + 65_536
group_result128 = admit_resident(group_profile128, group_plan128, devices, **identity)
assert all(row['required_bytes'] == expected + extra128 and
           row['operator_execution_storage_bytes'] == extra128 and
           row['runtime_graph_reserve_bytes'] == 2 << 30 for row in group_result128['ranks'])
try:
    admit_resident(group_profile128, group_plan128,
        [dict(d, free_bytes=expected+extra128-1) for d in devices], **identity)
except ValueError: pass
else: raise AssertionError('CTA128 arena silently borrowed runtime reserve')
with patch.dict(os.environ, {'GARNET_GPT_OSS_BF16_PEER_GROUP':'1',
                             'GARNET_GPT_OSS_BF16_PEER_GROUP_CTAS':'188'}):
    storage188 = peer_group_layout(256,16,2880)
assert storage188['ctas'] == 188
assert storage188['owned_bytes_per_rank'] == 23_617_776
assert storage188['mapped_host_bytes'] == 96_256
group_plan188 = copy.deepcopy(plan)
group_plan188['config']['hidden_size'] = 2880
group_plan188['operator_execution_layout'] = storage188
group_profile188 = copy.deepcopy(profile)
group_profile188['plan_identity'] = plan_identity(group_plan188)
extra188 = 23_617_776 + 96_256
group_result188 = admit_resident(group_profile188, group_plan188, devices, **identity)
assert all(row['required_bytes'] == expected + extra188 and
           row['operator_execution_storage_bytes'] == extra188 and
           row['runtime_graph_reserve_bytes'] == 2 << 30 for row in group_result188['ranks'])
try:
    admit_resident(group_profile188, group_plan188,
        [dict(d, free_bytes=expected+extra188-1) for d in devices], **identity)
except ValueError: pass
else: raise AssertionError('CTA188 arena silently borrowed runtime reserve')
threads512_environment=dict(identity['environment'],GARNET_GPT_OSS_BF16_PEER_GROUP='1',
    GARNET_GPT_OSS_BF16_PEER_GROUP_CTAS='188',GARNET_GPT_OSS_BF16_PEER_GROUP_THREADS='512',
    GARNET_GPT_OSS_BF16_PEER_GRID_SIGNALS='1')
threads512_profile=copy.deepcopy(group_profile188)
threads512_profile['kernel_environment']=threads512_environment
threads512_result=admit_resident(threads512_profile,group_plan188,devices,**dict(identity,environment=threads512_environment))
assert all(row['required_bytes']==expected+extra188 for row in threads512_result['ranks'])
for key in ('owned_bytes_per_rank','mapped_host_bytes','phase_elements','protocol'):
    damaged=copy.deepcopy(group_plan);damaged['operator_execution_layout'][key]=None
    try: plan_identity(damaged)
    except ValueError: pass
    else: raise AssertionError('Malformed peer resource profile accepted')
for value in ('1','64x','', ' 128', '187', '189', '256'):
    with patch.dict(os.environ, {'GARNET_GPT_OSS_BF16_PEER_GROUP':'1',
                                 'GARNET_GPT_OSS_BF16_PEER_GROUP_CTAS':value}):
        try: peer_group_layout(256,16,2880)
        except ValueError: pass
        else: raise AssertionError('Malformed peer CTA policy accepted')
for value in ('128','512x',' 512','1024'):
    with patch.dict(os.environ, {'GARNET_GPT_OSS_BF16_PEER_GROUP':'1',
                                 'GARNET_GPT_OSS_BF16_PEER_GROUP_THREADS':value}):
        try: peer_group_layout(256,16,2880)
        except ValueError: pass
        else: raise AssertionError('Malformed peer threads-per-CTA policy accepted')
with patch.dict(os.environ, {'GARNET_GPT_OSS_BF16_PEER_GROUP':'0',
                             'GARNET_GPT_OSS_BF16_PEER_GROUP_CTAS':'128'}):
    try: peer_group_layout(256,16,2880)
    except ValueError: pass
    else: raise AssertionError('Non-default peer CTA policy accepted while disabled')
with patch.dict(os.environ, {'GARNET_GPT_OSS_BF16_PEER_GROUP':'0',
                             'GARNET_GPT_OSS_BF16_PEER_GROUP_CTAS':'188'}):
    try: peer_group_layout(256,16,2880)
    except ValueError: pass
    else: raise AssertionError('CTA188 policy accepted while peer group disabled')

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
                  ('GARNET_GPT_OSS_MARLIN_PREFILL_UP_CTAS_PER_SM','2'),
                  ('GARNET_GPT_OSS_MARLIN_PREFILL_DOWN_CTAS_PER_SM','1'),
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

# Phase binding precedes rank work, and the final group closure retires after
# BOTH model releases (which own captured graphs). Default pairs need no binder.
events=[]
class BoundFake(Fake):
    def __init__(self,phase):super().__init__(kv);self.phase=phase
    def forward_rank_local(self, inputs, **kwargs):events.append(('forward',self.phase));return inputs
    def release(self):events.append(('release',self.phase));super().release()
first,second=BoundFake(0),BoundFake(1);pair=ResidentTensorParallel(first,second)
resource=object()
pair.attach_operator_execution_group(resource,lambda owner,phase: events.append(('bind',phase)))
assert first.operator_execution_group is second.operator_execution_group is resource
pair.forward_prefill('prefill');pair.forward_decode('decode');pair.release()
assert events==[('bind',0),('forward',0),('bind',1),('forward',1),('release',1),('release',0)]
assert pair._operator_phase_binder is None
assert pair._operator_group_releaser is None
try: pair.forward_decode('released')
except RuntimeError: pass
else: raise AssertionError('Released pair accepted execution')
# Explicit script-resource release follows both graph-owning model releases
# and is invoked once even if the caller releases the pair again.
events=[]
first,second=BoundFake(0),BoundFake(1);pair=ResidentTensorParallel(first,second)
pair.attach_operator_execution_group(resource,lambda owner,phase: events.append(('bind',phase)),
    lambda owner: events.append(('resource-release',owner)))
pair.release();pair.release()
assert events==[('release',1),('release',0),('resource-release',resource)]
assert pair._operator_group_releaser is pair._operator_phase_binder is None
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
