import sys,json,math
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen35.vl_0_8b.prefill_layer import PrefillLayer,T
with open(sys.argv[1]) as stream:f=json.load(stream)
c=f['config'];batch=f['batch'];chunk=f['chunk'];capacity=f['capacity'];shapes=f['shapes'];kind=f['kind'];idx=f['layer']
symbols=[tensor.input('chunk_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
@T.fusion(name='metal_prefill_reference',boundary='required')
def forward(hidden,state1,state2,positions,slots,active):
 return PrefillLayer(hidden,state1,state2,positions,slots,active,c,idx,batch,chunk,capacity,kind)
graph=garnet.compiled_graph()
try:
 graph.build(forward(*symbols),symbols,f['checkpoint'])
 for files in f['invalid_inputs']:
  inputs=[garnet.tensor_io().read_f32(path,s) for path,s in zip(files,shapes)];rejected=False
  try:graph.run(inputs)
  except Exception:rejected=True
  assert rejected,'Invalid chunk controls accepted'
 for step in f['steps']:
  inputs=[garnet.tensor_io().read_f32(path,s) for path,s in zip(step['inputs'],shapes)]
  actual=graph.run(inputs).tolist();expected=garnet.tensor_io().read_f32(step['expected'],[step['count']]).tolist()
  assert len(actual)==len(expected)
  failures=[];worst=0
  for i,(x,y) in enumerate(zip(actual,expected)):
   bound=5e-6+5e-5*abs(y);ratio=abs(x-y)/bound;worst=max(worst,ratio)
   if not math.isfinite(x) or ratio>1:
    if len(failures)<4:failures.append([i,x,y,ratio])
  print('chunk reference',kind,'batch',batch,'chunk',chunk,'worst_ratio',worst,'ms',graph.last_inference_ms,flush=True)
  assert not failures,str(failures)
 print('pretrained chunk PASS',kind,batch,chunk,flush=True)
finally:graph.close()
