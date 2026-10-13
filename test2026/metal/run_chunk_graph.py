import sys,json,math
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen35.vl_0_8b.prefill_layer import PrefillLayer,T
with open(sys.argv[1]) as stream:f=json.load(stream)
c=f['config'];batch=f['batch'];chunk=f['chunk'];capacity=f['capacity'];shapes=f['shapes'];kind=f['kind'];idx=f['layer']
symbols=[tensor.input('chunk_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
@T.fusion(name='metal_prefill_reference',boundary='required')
def forward(hidden,state1,state2,positions,slots,active):
 return PrefillLayer(hidden,state1,state2,positions,slots,active,c,idx,batch,chunk,capacity,kind,parallel_attention=True,parallel_linear=len(sys.argv)>5 and sys.argv[5]=='linear-chunk')
graph=garnet.compiled_graph();session=garnet.decoder_session()
try:
 if len(sys.argv)>4 and sys.argv[4]=='bf16':graph.build_bf16(forward(*symbols),symbols,f['checkpoint'])
 else:graph.build(forward(*symbols),symbols,f['checkpoint'])
 session.build([graph],None)
 for files in f['invalid_inputs']:
  inputs=[garnet.tensor_io().read_f32(path,s) for path,s in zip(files,shapes)];rejected=False
  try:graph.run(inputs)
  except Exception:rejected=True
  assert rejected,'Invalid chunk controls accepted'
  before=[v.tolist() for v in session.snapshot(0)];rejected=False
  try:session.run([graph],inputs[0],inputs[3],inputs[4],inputs[5],-1)
  except Exception:rejected=True
  assert rejected,'Invalid resident controls accepted'
  assert before==[v.tolist() for v in session.snapshot(0)],'Rejected input changed resident state'
 for step in f['steps']:
  inputs=[garnet.tensor_io().read_f32(path,s) for path,s in zip(step['inputs'],shapes)]
  state=session.snapshot(0)
  control=list(inputs);control[1]=state[0];control[2]=state[1]
  explicit=graph.run(control).tolist()
  session.run([graph],inputs[0],inputs[3],inputs[4],inputs[5],-1)
  h=session.hidden().tolist();state=session.snapshot(0);a=state[0].tolist();b=state[1].tolist();resident=[]
  for row in range(batch):
   resident+=h[row*(len(h)//batch):(row+1)*(len(h)//batch)]
   resident+=a[row*(len(a)//batch):(row+1)*(len(a)//batch)]
   resident+=b[row*(len(b)//batch):(row+1)*(len(b)//batch)]
  assert resident==explicit,'Resident state differs from identical explicit-state execution'
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
finally:
 session.reset()
 assert all(v==0 for state in session.snapshot(0) for v in state.tolist()),'Resident reset retained state'
 graph.close();rejected=False
 try:session.run([graph],inputs[0],inputs[3],inputs[4],inputs[5],-1)
 except Exception:rejected=True
 assert rejected,'Closed resident graph accepted'
 session.close()
