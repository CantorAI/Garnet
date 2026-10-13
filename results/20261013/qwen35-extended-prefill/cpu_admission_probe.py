"""CPU-only scheduling/API checks; no model execution or speed claims."""
import ast,json,hashlib
from pathlib import Path
from types import SimpleNamespace
r=Path.cwd();namespace={'CPUDecoder':object,'garnet':SimpleNamespace(backend_info=lambda:'Garnet Metal/MPS test; dense_key_split_k')}
for path,classname in [('xModel/qwen35/vl_0_8b/metal_decoder.py','MetalDecoder'),('xModel/qwen/metal_dense/decoder.py','DenseMetalDecoder')]:
 tree=ast.parse((r/path).read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==classname)
 node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in ('_prefill_profiles','_prefill_chunk_size','enable_prefill')]
 exec(compile(ast.Module(body=[node],type_ignores=[]),path,'exec'),namespace)
Base=namespace['MetalDecoder'];Dense=namespace['DenseMetalDecoder'];d=Dense();d.config={'hidden_size':1024};d.session=object()
def plan(decoder,count):
 result=[];remaining=count
 while remaining>0:
  chunk=decoder._prefill_chunk_size(remaining,decoder.chunk)
  assert chunk in decoder._prefill_profiles() and 0<chunk<=decoder.chunk
  result.append(chunk);remaining-=chunk
 assert sum(result)>=count and sum(result[:-1])<count
 return result
d.enable_prefill(32);assert plan(d,22)==[32]
d.enable_prefill(32,finer_chunks=True);assert plan(d,22)==[24] and plan(d,26)==[32]
d.enable_prefill(64,finer_chunks=True);assert plan(d,79)==[64,16] and plan(d,40)==[48]
cases=0
for maximum in (2,4,8,16,24,32,48,64):
 for fine in (False,True):
  d.enable_prefill(maximum,finer_chunks=fine)
  for length in range(1,257):plan(d,length);cases+=1
base=Base();base.enable_prefill(64);assert plan(base,79)==[64,16]
base.enable_prefill(32);assert plan(base,79)==[32,32,16]
d.enable_prefill(16);state=(d.chunk,d.prefill_projection,d.finer_chunks,d.prefill_graphs)
for args in ((24,'mps',False),(32,'mps',True)):
 rejected=False
 try:d.enable_prefill(args[0],args[1],finer_chunks=args[2])
 except ValueError:rejected=True
 assert rejected and state==(d.chunk,d.prefill_projection,d.finer_chunks,d.prefill_graphs) and state[3] is d.prefill_graphs
namespace['garnet'].backend_info=lambda:'Garnet Metal/MPS legacy'
for chunk in (24,32,48,64):
 try:d.enable_prefill(chunk)
 except ValueError:pass
 else:raise AssertionError('Legacy backend admitted extended profile')
 assert state==(d.chunk,d.prefill_projection,d.finer_chunks,d.prefill_graphs) and state[3] is d.prefill_graphs
try:d.enable_prefill(16,finer_chunks=True)
except ValueError:pass
else:raise AssertionError('Legacy backend admitted finer scheduling')

namespace['garnet'].backend_info=lambda:'Garnet Metal/MPS test; dense_key_split_k; qwen35_extended_prefill'
base=Base();base.session=object();base.batch=1;base.config={'hidden_size':1024}
base.enable_prefill(80,linear_chunk=True);assert plan(base,79)==[80]
base.batch=2;base.enable_prefill(40,linear_chunk=True);assert plan(base,79)==[40,40]
extra=0
for batch,maximum in ((1,40),(1,80),(2,40)):
 base.batch=batch;base.enable_prefill(maximum,linear_chunk=True)
 for length in range(1,257):plan(base,length);extra+=1
state=(base.chunk,base.prefill_projection,base.linear_chunk,base.prefill_graphs)
for chunk,projection,linear,batch,width,resident in ((80,'shared',True,2,1024,True),(40,'mps',True,1,1024,True),(40,'shared',False,1,1024,True),(40,'shared',True,1,2048,True),(40,'shared',True,1,1024,False)):
 base.batch=batch;base.config={'hidden_size':width};base.session=object() if resident else None
 try:base.enable_prefill(chunk,projection,linear)
 except ValueError:pass
 else:raise AssertionError('Unvalidated extension accepted')
 assert (base.chunk,base.prefill_projection,base.linear_chunk,base.prefill_graphs)==state and base.prefill_graphs is state[3]
namespace['garnet'].backend_info=lambda:'Garnet Metal/MPS legacy'
for chunk in (40,80):
 try:base.enable_prefill(chunk,linear_chunk=True)
 except ValueError:pass
 else:raise AssertionError('Legacy build accepted extension')
 assert base.prefill_graphs is state[3]
print(json.dumps(dict(scope='CPU-only planner/admission checks; no model/speed evidence',existing_dense_cases=cases,extended_qwen35_cases=extra,legacy_defaults_preserved=True,invalid_extensions_preserve_state=True,source_sha256={p:hashlib.sha256((r/p).read_bytes()).hexdigest() for p in ['xModel/qwen35/vl_0_8b/metal_decoder.py','xModel/qwen/metal_dense/decoder.py']}),indent=2))
