"""Actual early runner, real resident bounds and file hashes; no GPU execution."""
import copy,hashlib,json,os,runpy,sys,tempfile,types
from pathlib import Path
from unittest.mock import patch
repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'tools/gpt_oss'))
import resident_budget as budget

def forbidden(*args,**kwargs): raise AssertionError('Admission allocated/loaded model or ran inference')
with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary);cache=root/'cache';cache.mkdir();weights=root/'weights';weights.mkdir()
    devices=[dict(id=i,total_bytes=16<<30,free_bytes=15<<30) for i in range(2)]
    plan=dict(schema=1,mode='tensor-parallel',cache_key='fixture',hardware=['rank0','rank1'],
        batch=2,capacity=64,max_tokens=2,kv_pages=8,local_kv_heads=1,
        config=dict(num_hidden_layers=1,head_dim=32,vocab_size=64),
        expert_weight_shards=False,moe_intermediate_shards=True,marlin_prepacked=True,
        compact_vocab_greedy=True,marlin_workspace_layout='fixture',
        weight_storage_estimate=dict(prepacked_marlin_constant_bytes=96<<20))
    env=dict(GARNET_BATCH_CONTEXT_CAPACITY='64',GARNET_BATCH_PREFILL_CHUNK='2',
        GARNET_BATCH_PLAN_ONLY='1',GARNET_RESIDENT_PROFILE=str(root/'profile.json'),
        GARNET_GPT_OSS_MARLIN_PREPACKED='1')
    binaries={'libgarnet.so':'core','libgarnet_gpt_oss.so':'plugin'}
    checkpoint={'immutable_fixture':'header'};rows=[]
    for rank in range(2):
        for phase in ('prefill','decode'):
            path=cache/f'{rank}-{phase}.engine';path.write_bytes(b'engine-fixture')
            rows.append(dict(device=rank,phase=phase,total_weights_bytes=100<<20,
                context_device_memory_upper_bound_bytes=1<<20,engine_path=str(path),
                engine_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    profile=dict(resident_profile_schema=1,padded_prefill=True,
        plan_identity=budget.plan_identity(plan),native_binaries=binaries,
        hardware_csv='actual-hardware-fixture',checkpoint=checkpoint,cache_root=str(cache),
        kernel_environment={'GARNET_GPT_OSS_MARLIN_PREPACKED':'1'},engine_statistics=rows)
    (root/'profile.json').write_text(json.dumps(profile))
    request=root/'request.json';request.write_text(json.dumps({'input_ids':[1,2,3]}))
    for mode in ('pass','free','binary','environment','checkpoint','engine'):
        current=copy.deepcopy(devices);actual=copy.deepcopy(binaries);actual_checkpoint=checkpoint
        case_env=dict(env);output=root/f'{mode}.json'
        if mode=='free': current[0]['free_bytes']=1<<30
        if mode=='binary': actual['libgarnet_gpt_oss.so']='different-plugin'
        if mode=='environment':case_env['GARNET_GPT_OSS_MARLIN_PREFILL_DOWN_K']='64'
        if mode=='checkpoint':actual_checkpoint={'immutable_fixture':'replaced'}
        if mode=='engine':Path(rows[0]['engine_path']).write_bytes(b'changed-engine')
        garnet=types.SimpleNamespace(cuda_devices_json=lambda:json.dumps(current),
            tensor_from_host=forbidden,tensor_to_device=forbidden)
        pipeline=types.SimpleNamespace(make_tensor_parallel_plan=lambda *a,**k:copy.deepcopy(plan),
            build_tensor_parallel=forbidden)
        owner=types.SimpleNamespace(ResidentTensorParallel=types.SimpleNamespace(build=forbidden))
        with patch.dict(sys.modules,{'garnet':garnet,'pipeline':pipeline,'garnet_pipeline':owner}), \
             patch.dict(os.environ,case_env,clear=True), \
             patch.object(sys,'argv',['run_resident_batch_tp2.py',str(weights),str(cache),
                str(request),str(output),'2','16']), \
             patch.object(budget,'native_identity',return_value=actual), \
             patch.object(budget,'hardware_identity',return_value='actual-hardware-fixture'), \
             patch.object(budget,'checkpoint_identity',return_value=actual_checkpoint), \
             patch('subprocess.check_output',return_value='fixture-source'):
            try:runpy.run_path(str(repo/'tools/gpt_oss/run_resident_batch_tp2.py'),run_name='__main__')
            except SystemExit as error:assert mode=='pass' and error.code==0
            except ValueError:assert mode!='pass'
            else:raise AssertionError('Early runner did not exit or reject')
        assert output.exists()==(mode=='pass')
        if mode=='pass':
            result=json.loads(output.read_text())
            assert result['admission_only'] and result['input_token_ids']==[1,2,3]
            assert all(r['required_bytes']<=r['budget_bytes'] and r['runtime_graph_reserve_bytes']==2<<30 for r in result['admission']['ranks'])
            assert not any(k in result for k in ('decode_trials','full_request_output_tokens_per_second','answers'))
        if mode=='engine':Path(rows[0]['engine_path']).write_bytes(b'engine-fixture')
print('Real resident early admission exits without build/allocate/inference; free/native/env/checkpoint/engine hazards reject PASS')
