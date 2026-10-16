"""Exercise actual admission/planning methods on CPU without inference imports."""
import ast,json,sys,types
from pathlib import Path
source=Path(sys.argv[1]);output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=False)
backend={'info':'Garnet Metal/MPS ; dense_key_split_k; qwen35_extended_prefill'}
space={'garnet':types.SimpleNamespace(backend_info=lambda:backend['info'])}
def methods(path,class_name,names):
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==class_name)
    return [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
base=ast.ClassDef(name='MetalDecoder',bases=[],keywords=[],body=methods(source/'xModel/qwen35/vl_0_8b/metal_decoder.py','MetalDecoder',{'_prefill_profiles','_prefill_chunk_size','enable_prefill'}),decorator_list=[])
dense=ast.ClassDef(name='Dense',bases=[ast.Name(id='MetalDecoder',ctx=ast.Load())],keywords=[],body=methods(source/'xModel/qwen/metal_dense/decoder.py','DenseMetalDecoder',{'_prefill_profiles','_prefill_chunk_size','enable_prefill'}),decorator_list=[])
exec(compile(ast.fix_missing_locations(ast.Module(body=[base,dense],type_ignores=[])),'actual_planner','exec'),space)
def make(hidden=1024,resident=True):
    obj=space['Dense']();obj.config={'hidden_size':hidden};obj.session=object() if resident else None;obj.batch=1;obj.slots=[0]
    obj.enable_prefill(16);return obj
cases=0
for hidden in (896,1024):
    for batch in (1,2):
        for maximum in (32,64):
            for exact in (False,True):
                obj=make(hidden);obj.batch=batch;obj.enable_prefill(maximum,finer_chunks=True,exact_tail=exact)
                pool=(2,4,8,16,22,24,26,32,48,64) if exact else (2,4,8,16,24,32,48,64)
                for remaining in range(1,129):
                    wanted=next(v for v in pool if min(remaining,maximum)<=v<=maximum)
                    assert obj._prefill_chunk_size(remaining,maximum)==wanted;cases+=1
for label,hidden,resident,kwargs in [('legacy',1024,True,{}),('wide',1025,True,{}),('explicit',1024,False,{}),('projection',1024,True,{'projection':'mps'}),('coarse',1024,True,{'finer_chunks':False})]:
    backend['info']='Garnet Metal/MPS' if label=='legacy' else 'Garnet Metal/MPS ; dense_key_split_k'
    obj=make(hidden,resident);before=(obj.chunk,obj.prefill_projection,obj.prefill_graphs,list(obj.slots));rejected=False
    try:obj.enable_prefill(16,**dict({'finer_chunks':True,'exact_tail':True},**kwargs))
    except ValueError:rejected=True
    assert rejected and before==(obj.chunk,obj.prefill_projection,obj.prefill_graphs,obj.slots)
backend['info']='Garnet Metal/MPS ; dense_key_split_k';obj=make();obj.enable_prefill(32,finer_chunks=True)
for chunk in (22,26):
    rejected=False;before=(obj.chunk,obj.prefill_graphs,list(obj.slots))
    try:obj.enable_prefill(chunk,finer_chunks=True)
    except ValueError:rejected=True
    assert rejected and before==(obj.chunk,obj.prefill_graphs,obj.slots)
    obj.enable_prefill(chunk,finer_chunks=True,exact_tail=True);assert obj.chunk==chunk
class Graph:
    closed=False
    def close(self):self.closed=True
graph=Graph();obj.prefill_graphs={'cached':graph};obj.enable_prefill(32,finer_chunks=True,exact_tail=False)
assert graph.closed and obj.prefill_graphs=={}
result=dict(planning_cases=cases,invalid_admissions_preserve_state=True,explicit_profiles_opt_in=True,policy_switch_closes_cached_graphs=True)
(output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
