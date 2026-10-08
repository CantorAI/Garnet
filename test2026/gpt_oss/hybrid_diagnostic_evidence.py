"""Run the actual diagnostic with CPU mocks to verify failed-matrix retention."""
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
from unittest.mock import patch

repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'tools/gpt_oss'))
import resident_budget as budget

with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary);weights=root/'weights';weights.mkdir()
    (weights/'request.json').write_text(json.dumps({'input_ids':[1]*65}))
    (weights/'expected.json').write_text(json.dumps({'generated':[1,1,1],
        'generation_logits':[[0.,0.],[0.,0.],[0.,0.]]}))
    (weights/'config.json').write_text('{}');(weights/'model.safetensors').write_bytes(b'CPU MOCK')
    plan=dict(estimated_per_gpu_bytes=1,stages=[dict(device_id=i,budget_bytes=1000) for i in (0,1)],
        config=dict(num_hidden_layers=2,head_dim=64),kv_pages=12,local_kv_heads=1)
    def execute(mode):
        output=root/(mode+'.json');cache=root/(mode+'-cache')
        calls=[];releases=[]
        class Pair:
            prefill_stages=plan['stages'];decode_stages=plan['stages']
            def reply(self,inputs):
                calls.append(inputs)
                values=[0.]*4
                if mode=='numeric':values[0]=.1
                if mode=='extent':values.pop()
                if mode=='replay' and len(calls)>11:values[0]=.001
                return {'output':values}
            forward_prefill=reply;forward_decode=reply
            def release(self):
                releases.append(True)
                if mode=='release':raise RuntimeError('Injected release failure')
        class Host:
            def __init__(self,values):self.values=values
            def tolist(self):return self.values
        garnet=types.SimpleNamespace(cuda_devices_json=lambda:'[{},{}]',cuda_set_device=lambda i:0,
            tensor_from_host=lambda values,**kw:values,tensor_to_cpu=lambda values:Host(values))
        pipeline=types.SimpleNamespace(make_tensor_parallel_plan=lambda *a,**k:plan,
            build_tensor_parallel=lambda *a,**k:None)
        owner=types.SimpleNamespace(ResidentTensorParallel=types.SimpleNamespace(build=lambda *a,**k:Pair()))
        env=dict(CANTORAI_ROOT=str(root),GARNET_TRT_SYNC_ALLOCATOR='1')
        if mode=='overwrite':output.with_suffix('.partial.json').write_text('old evidence')
        with patch.dict(sys.modules,{'garnet':garnet,'pipeline':pipeline,'garnet_pipeline':owner}), \
             patch.dict(os.environ,env,clear=True), \
             patch.object(sys,'argv',['tp_hybrid_kv.py',str(weights),str(cache),str(output),'2','8']), \
             patch('subprocess.check_output',return_value='a'*40), \
             patch.object(budget,'native_identity',return_value={'core':'cpu mock'}), \
             patch.object(budget,'kernel_environment',return_value={'GARNET_TRT_SYNC_ALLOCATOR':'1'}), \
             patch.object(budget,'hardware_identity',return_value='CPU mock, no hardware proof'):
            try:runpy.run_path(str(repo/'test2026/gpt_oss/tp_hybrid_kv.py'),run_name='__main__')
            except (ValueError,RuntimeError,FileExistsError):assert mode!='pass'
            else:assert mode=='pass'
        if mode=='overwrite':
            assert not calls and not releases and output.with_suffix('.partial.json').read_text()=='old evidence'
            return
        assert len(releases)==1
        partial=json.loads(output.with_suffix('.partial.json').read_text())
        assert partial['passed'] is False and partial['source_commit']=='a'*40
        assert partial['cache_root']==str(cache.resolve()) and len(partial['fixture_sha256'])==4
        if mode=='pass':
            complete=json.loads(output.read_text())
            assert complete['passed'] and len(complete['steps'])==6 and complete['complete_generations']==2
        else:
            assert not output.exists() and partial['failure']
            assert partial['steps'], 'Failure discarded raw numerical matrices'
            if mode in ('numeric','extent'):
                assert len(partial['steps'])==1 and not partial['steps'][0]['within_existing_tolerance']
            else:assert len(partial['steps'])==6
            if mode=='replay':assert partial['steps'][0]['tp_logits']!=partial['steps'][3]['tp_logits']
            if mode=='release':assert 'Cleanup failed' in partial['failure']
    for mode in ('pass','numeric','extent','replay','release','overwrite'):execute(mode)
print('Actual hybrid diagnostic CPU mocks: provenance, successful matrices, numeric/extent/replay/cleanup failure preservation and overwrite rejection PASS; no GPU proof')
