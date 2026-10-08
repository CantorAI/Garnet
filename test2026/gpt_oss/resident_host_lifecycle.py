"""CPU contract checks: defaults, opt-in sampling/reuse, and true ragged input length."""
import ast
import os
from pathlib import Path
import sys
import threading
import time
import types

repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'python'))
local=threading.local();calls=[];sample_calls=[]

def set_device(device):
    previous=getattr(local,'device',0);local.device=device;return previous

class Host:
    def tolist(self):
        sample_calls.append('script')
        return [3.,17.,3.,4.,2.,8.,1.,9.]

def native_merge(tensor):
    sample_calls.append('native')
    return dict(status='ok',token_ids=[4,8],token_values=[3.,2.],token_id=4)

class Model:
    def forward(self,request):
        calls.append(dict(request))
        return dict(status='ok',output='compact')

g=types.ModuleType('garnet');g.cuda_set_device=set_device;g.cuda_synchronize=lambda:None
g.tensor_to_cpu=lambda tensor:Host();g.merge_greedy_candidate_pairs=native_merge
sys.modules['garnet']=g
from garnet_pipeline import TensorParallel
stages=[dict(rank=r,device_id=r,model=Model(),keys='K',values='V') for r in range(2)]
pair=TensorParallel(stages,greedy_candidate_pairs=True)
try:
    inputs=[('tokens',['position','pages','length','slot','active']) for _ in stages]
    baseline=pair.forward_rank_local(inputs,sample=True,sample_batch=True)
    assert sample_calls==['script'] and all('reuse_output' not in r for r in calls)
    for reuse in (False,True):
        for native in (False,True):
            calls.clear();sample_calls.clear()
            reply=pair.forward_rank_local(inputs,sample=True,sample_batch=True,
                reuse_output=reuse,native_candidate_merge=native)
            assert reply==baseline and sample_calls==['native' if native else 'script']
            assert len(calls)==2 and all(r.get('reuse_output',False)==reuse for r in calls)
            calls.clear();sample_calls.clear()
            unsampled=pair.forward_rank_local(inputs,sample=False,
                reuse_output=reuse,native_candidate_merge=native)
            assert unsampled['output']=='compact' and not sample_calls
finally:
    pair.executor.shutdown(wait=True)

# Exercise the actual runner function with a tiny no-GPU engine double. This
# checks scheduling/true-length semantics, not numerical or timing performance.
tree=ast.parse((repo/'tools/gpt_oss/run_resident_batch_tp2.py').read_text())
defs=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('optional_flag','run_request')]
assert len(defs)==2
namespace=dict(os=os,time=time,chunk=4,batch=3,output_tokens=4,prefill_inputs=[],decode_inputs=[])
class Capture:
    def prefill_begin(self,*args):pass
    def prefill_end(self,*args):pass
    def decode_begin(self,*args):pass
    def decode_end(self,*args):pass
namespace['capture']=Capture()
exec(compile(ast.Module(body=defs,type_ignores=[]),'resident-runner-functions','exec'),namespace)
flag_name='GARNET_RESIDENT_REUSE_OUTPUT';saved=os.environ.get(flag_name)
try:
    os.environ.pop(flag_name,None);assert namespace['optional_flag'](flag_name) is False
    for raw in ('0','1'):
        os.environ[flag_name]=raw;assert namespace['optional_flag'](flag_name)==(raw=='1')
    for raw in ('', 'yes', '2'):
        os.environ[flag_name]=raw
        try:namespace['optional_flag'](flag_name)
        except ValueError:pass
        else:raise AssertionError('Malformed opt-in flag accepted')
finally:
    if saved is None:os.environ.pop(flag_name,None)
    else:os.environ[flag_name]=saved

for reuse in (False,True):
    for native in (False,True):
        for final_only in (False,True):
            updates=[];requests=[]
            def update(stages,prepared,tokens,positions,length,slot):
                updates.append(dict(tokens=tokens,positions=positions,length=length,slot=slot))
            class Pair:
                prefill_stages=decode_stages=[]
                def forward_prefill(self,prepared,**kwargs):
                    requests.append(dict(phase='prefill',**kwargs))
                    return dict(token_ids=[100]*3) if kwargs['sample'] else dict(output='unobserved-candidates')
                def forward_decode(self,prepared,**kwargs):
                    requests.append(dict(phase='decode',**kwargs))
                    return dict(token_ids=[100+sum(r['phase']=='decode' for r in requests)]*3)
            namespace.update(pair=Pair(),update=update,reuse_output=reuse,
                native_candidate_merge=native,final_prefill_sample_only=final_only)
            result=namespace['run_request'](0,[11,12,13,14,15])
            assert result['token_ids_by_request']==[[100,101,102,103]]*3
            assert [r['sample'] for r in requests[:2]]==[not final_only,True]
            assert all(r['reuse_output']==reuse and r['native_candidate_merge']==native for r in requests)
            assert updates[1]==dict(tokens=[15,0,0,0]*3,positions=[4,5,6,7]*3,length=5,slot=4)
            assert [(r['length'],r['slot']) for r in updates[2:]]==[(6,5),(7,6),(8,7)]
            assert len(result['prefill_step_seconds'])==2 and len(result['decode_step_seconds'])==3
            assert result['full_request_wall_seconds']>0 and not result['prefill_kv_reused_for_decode_trial']
print('Resident host-lifecycle defaults, all opt-in combinations and true ragged tail passed')
