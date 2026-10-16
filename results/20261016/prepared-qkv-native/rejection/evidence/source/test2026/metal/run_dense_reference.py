import sys,json,math
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen.metal_dense.layers import DensePrefillLayer as PrefillLayer,T
with open(sys.argv[1]) as stream:f=json.load(stream)
c=f['config'];batch=f['batch'];chunk=f['chunk'];capacity=f['capacity'];shapes=f['shapes'];kind='full_attention';idx=f['layer']
symbols=[tensor.input('chunk_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
@T.fusion(name='metal_prefill_reference',boundary='required')
def forward(hidden,state1,state2,positions,slots,active):
 return PrefillLayer(hidden,state1,state2,positions,slots,active,c,idx,batch,chunk,capacity,kind)
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
  if 'joined_mps_prefill' in garnet.backend_info():
   expected_pair=int((batch*chunk==44 and c['hidden_size']==1024 and c['intermediate_size']==3072) or (batch*chunk==26 and c['hidden_size']==896 and c['intermediate_size']==4864)) if len(sys.argv)>4 and sys.argv[4]=='bf16' else 0
   assert session.joined_prefill_pairs==expected_pair,'Joined prefill dispatch selection mismatch'
   print('joined_prefill_dispatch PASS',batch,chunk,session.joined_prefill_pairs)
  if 'prepared_qkv_text' in garnet.backend_info():
   expected_qkv=int(len(sys.argv)>4 and sys.argv[4]=='bf16' and c['hidden_size']==1024 and c['num_attention_heads']*c['head_dim']==2048 and c['num_key_value_heads']*c['head_dim']==1024 and batch*chunk in (24,44))
   assert session.prepared_qkv_dispatches==expected_qkv,'Resident QKV route mismatch'
   assert graph.prepared_qkv_dispatches==expected_qkv,'Explicit graph QKV route mismatch'
   print('prepared_qkv_trained_route '+json.dumps(dict(batch=batch,chunk=chunk,expected=expected_qkv,session=session.prepared_qkv_dispatches,graph=graph.prepared_qkv_dispatches)),flush=True)
  h=session.hidden().tolist();state=session.snapshot(0);a=state[0].tolist();b=state[1].tolist();resident=[]
  for row in range(batch):
   resident+=h[row*(len(h)//batch):(row+1)*(len(h)//batch)]
   resident+=a[row*(len(a)//batch):(row+1)*(len(a)//batch)]
   resident+=b[row*(len(b)//batch):(row+1)*(len(b)//batch)]
  assert resident==explicit,'Resident state differs from identical explicit-state execution'
  actual=graph.run(inputs).tolist();expected=garnet.tensor_io().read_f32(step['expected'],[step['count']]).tolist()
  assert len(actual)==len(expected)
  failures=[];worst=0;hidden_bounds={};hidden_l2_worst=0;value_l2_worst=0;value_bounds={};row_width=len(actual)//batch
  # Test hidden vectors in both infinity and Euclidean norms. Cancellation
  # can put individual components near zero in the pretrained final MLP.
  for b in range(batch):
   for token in range(chunk):
    start=b*row_width+token*c['hidden_size'];end=start+c['hidden_size']
    wanted=expected[start:end];obtained=actual[start:end]
    hidden_bounds[start]=5e-6+5e-5*max([abs(y) for y in wanted])
    error=math.sqrt(sum([(x-y)**2 for x,y in zip(obtained,wanted)]))
    scale=math.sqrt(sum([y*y for y in wanted]));ratio=error/(5e-6+5e-5*scale)
    hidden_l2_worst=max(hidden_l2_worst,ratio)
    assert math.isfinite(error) and ratio<=1,'Hidden vector norm mismatch'
  key_start=chunk*c['hidden_size'];cache=c['num_key_value_heads']*capacity*c['head_dim'];dim=c['head_dim']
  for b in range(batch):
   for vector in range(c['num_key_value_heads']*capacity):
    start=b*row_width+key_start+cache+vector*dim;end=start+dim
    wanted=expected[start:end];obtained=actual[start:end]
    value_bounds[start]=5e-6+5e-5*max([abs(y) for y in wanted])
    error=math.sqrt(sum([(x-y)**2 for x,y in zip(obtained,wanted)]));scale=math.sqrt(sum([y*y for y in wanted]))
    ratio=error/(5e-6+5e-5*scale);value_l2_worst=max(value_l2_worst,ratio)
    assert math.isfinite(error) and ratio<=1,'Value vector norm mismatch'
  for i,(x,y) in enumerate(zip(actual,expected)):
   bound=5e-6+5e-5*abs(y);difference=abs(x-y)
   row_width=len(actual)//batch;local=i%row_width;key_start=chunk*c['hidden_size'];cache=c['num_key_value_heads']*capacity*c['head_dim']
   if local<key_start:bound=hidden_bounds[(i//row_width)*row_width+(local//c['hidden_size'])*c['hidden_size']]
   elif local>=key_start+cache:bound=value_bounds[(i//row_width)*row_width+key_start+cache+((local-key_start-cache)//dim)*dim]
   if key_start<=local<key_start+cache:
    # RoPE is a two-dimensional rotation: test vector error against vector
    # magnitude, avoiding a component-relative bound near axis crossings.
    dim=c['head_dim'];col=(local-key_start)%dim;partner=i+dim//2 if col<dim//2 else i-dim//2
    bound=5e-6+5e-5*math.sqrt(y*y+expected[partner]*expected[partner])
    difference=math.sqrt((x-y)**2+(actual[partner]-expected[partner])**2)
   ratio=difference/bound;worst=max(worst,ratio)
   if not math.isfinite(x) or ratio>1:
    if len(failures)<4:failures.append([i,x,y,ratio])
  if failures:
   for entry in failures:
    index=entry[0];row_width=len(actual)//batch;local=index%row_width;key_start=chunk*c['hidden_size'];cache=c['num_key_value_heads']*capacity*c['head_dim']
    if key_start<=local<key_start+cache:
     dim=c['head_dim'];col=(local-key_start)%dim;partner=index+dim//2 if col<dim//2 else index-dim//2
     print('key_rotation_diagnostic',dict(index=index,actual=actual[index],reference=expected[index],partner_actual=actual[partner],partner_reference=expected[partner],pair_magnitude=math.sqrt(expected[index]**2+expected[partner]**2)),flush=True)
  print('dense reference',kind,'batch',batch,'chunk',chunk,'max_abs_error',max([abs(x-y) for x,y in zip(actual,expected)]),'hidden_l2_worst_ratio',hidden_l2_worst,'value_l2_worst_ratio',value_l2_worst,'worst_ratio',worst,'ms',graph.last_inference_ms,flush=True)
  assert not failures,str(failures)
 print('pretrained dense PASS',kind,batch,chunk,flush=True)
finally:
 session.reset()
 assert all(v==0 for state in session.snapshot(0) for v in state.tolist()),'Resident reset retained state'
 graph.close();rejected=False
 try:session.run([graph],inputs[0],inputs[3],inputs[4],inputs[5],-1)
 except Exception:rejected=True
 assert rejected,'Closed resident graph accepted'
 session.close()
