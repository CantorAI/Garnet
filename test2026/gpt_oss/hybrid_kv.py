"""Actual hybrid planner/adapter/owner contracts on a CPU recording runtime.

No CUDA/compiled/numerical/throughput claim. Native and compiled gates remain
required. The address oracle uses actual production tables, with independent
full-history tagged positions through repeated requests and padded tiles.
"""
import ast
import copy
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import types
from unittest.mock import patch

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / 'tools/gpt_oss'))
sys.path.insert(0, str(repo / 'python'))
sys.path.insert(0, str(Path(__file__).parent))
from make_fixture import create
from pipeline import make_tensor_parallel_plan, build_tensor_parallel
from kv_layout import hybrid_layout, checked_layout, kv_memory, window_page_table, logical_retained_kv_bytes
from resident_budget import admit_resident, plan_identity
from garnet_pipeline import ResidentTensorParallel, TensorParallel


def rejects(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError('Unsafe contract accepted')


# Independent integer page address simulation, not attention arithmetic.
reads = 0
for window in (17, 128):
    for chunk in (1, 8, 16, 32):
        layout = hybrid_layout(dict(num_hidden_layers=6, head_dim=64,
            sliding_window=window, num_key_value_heads=8), 3, 2560, chunk, 4)
        table = window_page_table(layout)
        physical_pages = layout['window_shape'][1]
        logical_pages = (2560 + 15) // 16
        memory = {}
        def address(layer, slot, position):
            return (layer * physical_pages + table[slot * logical_pages + position // 16]) * 16 + position % 16
        for generation in (0, 1):
            for slot, length in enumerate((129, 256, 2005)):
                for start in range(0, length, chunk):
                    for layer in range(3):
                        for position in range(start, start + chunk):
                            memory[address(layer, slot, position)] = (generation, layer, slot, position)
                    for query in range(start, start + chunk):
                        for layer in range(3):
                            for position in range(max(0, query + 1 - window), query + 1):
                                assert memory[address(layer, slot, position)] == (generation, layer, slot, position)
                                reads += 1
                for query in range(length, length + 511):
                    for layer in range(3):
                        memory[address(layer, slot, query)] = (generation, layer, slot, query)
                    for layer in range(3):
                        for position in range(max(0, query + 1 - window), query + 1):
                            assert memory[address(layer, slot, position)] == (generation, layer, slot, position)
                            reads += 1
        occupied = [address(layer, slot, p) for layer in range(3) for slot in range(3)
                    for p in range(layout['window_pages_per_request'] * 16)]
        assert len(set(occupied)) == len(occupied)
assert reads == 13_124_892


class Runtime:
    def __init__(self):
        self.device = threading.local()
        self.allocations, self.models, self.updates = [], [], []
        self.fail_rank = None
    def cuda_set_device(self, device):
        old = getattr(self.device, 'id', 0)
        self.device.id = device
        return old
    def tensor_zeros(self, shape, dtype):
        return self.tensor_from_host(None, dtype, shape, 'cuda')
    def tensor_from_host(self, data, dtype, shape, device):
        assert device == 'cuda'
        value = types.SimpleNamespace(shape=list(shape), dtype=dtype,
            data=data, device=getattr(self.device, 'id', 0))
        self.allocations.append(value)
        return value
    def load_model(self, source, **options):
        model = Model(self, source, options)
        self.models.append(model)
        return model
    def cuda_synchronize(self): pass
    def tensor_to_device(self, value, device):
        assert value.device == device
        return value
    def tensor_update_int_vectors_async(self, tensors, values):
        assert all(t.device == self.device.id for t in tensors)
        self.updates.append((tensors, values))
        return True


class Model:
    def __init__(self, runtime, source, options):
        self.runtime, self.source, self.options = runtime, Path(source), options
        self.rank = getattr(runtime.device, 'id', 0)
        self.releases = 0
        self.requests = []
    def runtime_status(self): return {'ready': self.rank != self.runtime.fail_rank}
    def release_runtime(self): self.releases += 1
    def forward(self, request):
        assert all(t.device == self.rank for t in request['inputs'])
        self.requests.append(request)
        return {'status': 'ok', 'output': object()}


with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=True):
    root = Path(temporary)
    fixture = root / 'fixture'
    create(fixture)
    devices = [dict(id=i, name='synthetic', total_bytes=16 << 30, free_bytes=16 << 30,
        compute_major=12, compute_minor=0, pci_bus_id=str(i), peer_access=[True, True]) for i in range(2)]
    os.environ.update(GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS='1', GARNET_GPT_OSS_MARLIN_PREPACKED='1')
    def plan(tokens=8):
        return make_tensor_parallel_plan(fixture, devices, batch=3, capacity=128,
            tokens=tokens, reserve_bytes=0)
    default = plan()
    os.environ['GARNET_GPT_OSS_HYBRID_KV'] = '0'
    assert plan()['cache_key'] == default['cache_key'] and default['schema'] == 3
    runtime = Runtime()
    with patch.dict(sys.modules, {'garnet': runtime}):
        normal = build_tensor_parallel(fixture, root / 'cache', default, 8, True,
            last_token_logits=True, padded_prefill=True)
        assert all(len(s['model'].options['input_shapes']) == 8 for s in normal.stages)
        assert all('select_last_valid_sequence' in (s['model'].source.parent / 'gpt_oss_llm.py').read_text()
                   for s in normal.stages)
        assert all('shared_resources' not in s for s in normal.stages)
        normal.release()
        assert all(m.releases == 1 for m in runtime.models)
    os.environ['GARNET_GPT_OSS_HYBRID_KV'] = '1'
    hybrid = plan()
    assert hybrid['schema'] == 4 and hybrid['cache_key'] != default['cache_key']
    assert plan(16)['cache_key'] != hybrid['cache_key']
    layout = checked_layout(hybrid)
    kv_bytes, auxiliary = kv_memory(hybrid)
    # Fixture: two logical layers => one per bank, KV head1, dim8, full8pages,
    # ring1page/request. Four BF16 tensors plus a three-by-eight INT32 table.
    assert kv_bytes == 3 * (8 + 1) * 16 * 8 * 2 * 2 and auxiliary == 3 * 8 * 4
    assert logical_retained_kv_bytes(hybrid,65) == 3 * (65+2) * 2 * 8 * 2
    assert logical_retained_kv_bytes(default,65) == 3 * 2 * 65 * 2 * 8 * 2
    rejects(lambda: logical_retained_kv_bytes(hybrid,129))
    assert default['estimated_per_gpu_bytes'] - hybrid['estimated_per_gpu_bytes'] == kv_memory(default)[0] - kv_bytes - auxiliary
    for key in ('window_shape', 'shared_kv_bytes', 'max_prefill_tokens', 'extra_tensor_arguments'):
        bad = copy.deepcopy(hybrid)
        bad['kv_layout'][key] = None
        rejects(lambda: checked_layout(bad))
    bad = copy.deepcopy(hybrid); bad['stages'][1]['start'] = 1
    rejects(lambda: checked_layout(bad))
    for key,value in (('schema',3),('mode','pipeline')):
        bad = copy.deepcopy(hybrid); bad[key] = value
        rejects(lambda: checked_layout(bad))
    for key,value in (('device_id',0),('rank',0)):
        bad = copy.deepcopy(hybrid); bad['stages'][1][key] = value
        rejects(lambda: checked_layout(bad))
    rejects(lambda: hybrid_layout(dict(hybrid['config'], num_hidden_layers=3), 3, 128, 8, 1))
    rejects(lambda: hybrid_layout(hybrid['config'], 3, 128, 129, 1))
    rejects(lambda: hybrid_layout(hybrid['config'], True, 128, 8, 1))
    runtime = Runtime()
    with patch.dict(sys.modules, {'garnet': runtime}):
        pair = ResidentTensorParallel.build(
            lambda: build_tensor_parallel(fixture, root / 'hybrid-cache', hybrid, 8, True,
                last_token_logits=True, padded_prefill=True),
            lambda shared: build_tensor_parallel(fixture, root / 'hybrid-cache', hybrid, 1, False, shared))
        assert len(runtime.allocations) == 10 # K/V, window K/V/table per rank, allocated only once.
        for first, second in zip(pair.prefill_stages, pair.decode_stages):
            assert first['keys'] is second['keys'] and first['values'] is second['values']
            assert all(first['shared_resources'][k] is second['shared_resources'][k] for k in layout['extra_tensor_arguments'])
            assert first['shared_resources']['window_table'].data == window_page_table(layout)
            assert len(first['model'].options['input_shapes']) == 11
            assert first['model'].options['input_shapes'][2] == layout['global_shape']
            assert first['model'].options['input_shapes'][8:] == [layout['window_shape'], layout['window_shape'], [3, 8]]
            assert 'STAGE_PADDED_PREFILL = 1' in first['model'].source.read_text()
        groups = []
        for stage in pair.decode_stages:
            runtime.cuda_set_device(stage['device_id'])
            token = runtime.tensor_zeros([3, 1], 'int64')
            controls = [runtime.tensor_zeros([3, 1], 'int64')] + [runtime.tensor_zeros([3], 'int32') for _ in range(4)]
            groups.append((token, controls))
        pair.forward_decode(groups, vector_values=[[1]*3, [17]*3, [18]*3, [17]*3])
        assert len(runtime.updates) == 2
        for s in pair.decode_stages:
            assert s['model'].requests[0]['inputs'][8:] == [s['shared_resources'][n] for n in layout['extra_tensor_arguments']]
        original_layout = copy.deepcopy(pair.decode_stages[0]['shared_resource_layout'])
        pair.decode_stages[0]['shared_resource_layout']['schema'] = 0
        assert pair.prefill_stages[0]['shared_resource_layout']['schema'] == 1
        rejects(lambda: ResidentTensorParallel(pair._prefill, pair._decode))
        pair.decode_stages[0]['shared_resource_layout'] = original_layout
        # A reallocated single auxiliary tensor invalidates shared phase ownership.
        pair.decode_stages[0]['shared_resources']['window_keys'] = object()
        rejects(lambda: ResidentTensorParallel(pair._prefill, pair._decode))
        pair.release(); pair.release()
        assert all(m.releases == 1 for m in runtime.models)
        saved = len(runtime.models)
        os.environ['GARNET_GPT_OSS_HYBRID_KV'] = '0'
        rejects(lambda: build_tensor_parallel(fixture, root / 'invalid', hybrid, 1, False))
        assert len(runtime.models) == saved
        os.environ['GARNET_GPT_OSS_HYBRID_KV'] = '1'
    # A load failure on the second rank releases both loaded models exactly once.
    runtime = Runtime(); runtime.fail_rank = 1
    with patch.dict(sys.modules, {'garnet': runtime}):
        try: build_tensor_parallel(fixture, root / 'failed-cache', hybrid, 8, True)
        except RuntimeError: pass
        else: raise AssertionError('Failed builder accepted')
        assert len(runtime.models) == 2 and all(m.releases == 1 for m in runtime.models)
    runtime = Runtime()
    with patch.dict(sys.modules, {'garnet': runtime}), \
         patch.object(Model, 'runtime_status', side_effect=RuntimeError('status failed')):
        try: build_tensor_parallel(fixture, root / 'failed-status-cache', hybrid, 8, True)
        except RuntimeError: pass
        else: raise AssertionError('Status failure accepted')
        assert len(runtime.models) == 1 and runtime.models[0].releases == 1
    identity = dict(binaries={'core':'fixed','plugin':'fixed'}, hardware_csv='CPU fixture',
        environment={'GARNET_GPT_OSS_HYBRID_KV':'1'}, checkpoint={'header':'fixed'},
        cache=str(root / 'cache'), padded_prefill=True)
    profile = dict(resident_profile_schema=1, native_binaries=identity['binaries'],
        hardware_csv=identity['hardware_csv'], kernel_environment=identity['environment'],
        checkpoint=identity['checkpoint'], cache_root=str((root / 'cache').resolve()),
        padded_prefill=True, plan_identity=plan_identity(hybrid), engine_statistics=[
            dict(device=rank, phase=phase, total_weights_bytes=1 << 30,
                context_device_memory_upper_bound_bytes=8 << 20)
            for rank in (0, 1) for phase in ('prefill', 'decode')])
    result = admit_resident(profile, hybrid, devices, **identity)
    expected = ((2 << 30) * 105 + 99) // 100 + (16 << 20) + kv_bytes + auxiliary + (2 << 30)
    assert all(r['required_bytes'] == expected and r['shared_auxiliary_bytes'] == auxiliary for r in result['ranks'])
    old = copy.deepcopy(profile); old['plan_identity'] = plan_identity(default)
    rejects(lambda: admit_resident(old, hybrid, devices, **identity))
    limited = [dict(d, free_bytes=expected - 1) for d in devices]
    rejects(lambda: admit_resident(profile, hybrid, limited, **identity))

# Execute the actual new XModel with a recording tensor surface, not compilation.
operations, layers = [], []
class Expr:
    def __mul__(self, other): return self
class Surface:
    def unary_op(self, name, **attributes): operations.append(name); return Expr()
    def binary_op(self, name, **attributes): operations.append(name); return Expr()
    def fusion(self, **options): return lambda fn: fn
surface = Surface()
package = types.ModuleType('hybrid_contract_fixture')
package.__path__ = [str(repo / 'xModel/gpt_oss/120b')]
with patch.dict(sys.modules, {'garnet':types.SimpleNamespace(tensor=lambda:surface),
        'hybrid_contract_fixture':package}):
    hybrid_model = importlib.import_module('hybrid_contract_fixture.gpt_oss_hybrid_llm')
    def layer(x, pos, keys, values, table, length, slot, active, config, logical,
              prefill, physical, *args):
        layers.append((logical, physical, keys, values, table)); return x
    with patch.object(hybrid_model.llm, 'layer', side_effect=layer), \
         patch.object(hybrid_model.llm, 'rounded', side_effect=lambda x:x), \
         patch.object(hybrid_model.llm, 'norm', side_effect=lambda x,*a:x), \
         patch.object(hybrid_model.llm, 'linear', side_effect=lambda x,*a,**k:x):
        global_k,global_v,global_t,window_k,window_v,window_t = [object() for _ in range(6)]
        for padded in (0, 1):
            layers.clear(); operations.clear()
            hybrid_model.forward_stage(Expr(), Expr(), global_k, global_v, global_t,
                Expr(), Expr(), Expr(), window_k, window_v, window_t, None,
                dict(num_hidden_layers=6,hidden_size=32,vocab_size=64),
                0, 6, 1, 1, 0, 1, 1, 0, 1, padded)
            assert [r[:2] for r in layers] == [(0,0),(1,0),(2,1),(3,1),(4,2),(5,2)]
            assert all(r[2:] == (window_k,window_v,window_t) if r[0]%2==0 else
                r[2:] == (global_k,global_v,global_t) for r in layers)
            assert ('select_last_valid_sequence' in operations) == bool(padded)
            assert ('last_token' in operations) == (not padded)
print('Hybrid actual CPU planner/build/owner/admission/source-import contracts PASS;', reads, 'independent tagged reads; GPU parity/performance pending')
