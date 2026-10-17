"""Two real vision projections: materialized-weight liveness and immutable sharing."""
import sys,json,math,pathlib,hashlib,struct
mode=sys.argv[1];w=pathlib.Path(__file__).resolve().parent;s='/Users/shawn/CantorAI/out/research/worldsense-decode-step-profile/source';sys.path.insert(0,s);sys.path.insert(0,'/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-decode-step-profile-'+mode)
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T
checkpoint='/Users/shawn/CantorAI/out/models/Qwen3.5-4B/model.safetensors.index.json';enabled=mode=='on'
@T.fusion(name='streamed_two_vision_projection_guard',boundary='required')
def forward(x):
 a=x*T.unary_op('qwen_text_head',weight_name='model.visual.blocks.0.attn.qkv.weight',bf16_weight=0)
 b=x*T.unary_op('qwen_text_head',weight_name='model.visual.blocks.1.attn.qkv.weight',bf16_weight=0)
 return a*T.binary_op('add')*b
records=[]
def unbound(g):
 inv=json.loads(garnet.memory_inventory_json([dict(object=g,tag='current')],None));ids={r['buffer_id']for r in inv['references']if r['role']=='graph_workspace'}
 assert not any(r['buffer_id']in ids for r in inv['references']if r['role']=='slot')
 rawrefs=[r for r in inv['references']if 'VisionRawBF16'in r['name']]
 assert all(r['representation']=='bf16'for r in rawrefs)
 if enabled:
  assert rawrefs
  assert len([r for r in inv['references']if r['role']=='graph_workspace'and r['shape']==[3072,1024]])==1
 return inv
for batch in [1,3]:
 symbol=tensor.input('projection_input_'+str(batch),shape=[batch,1024],dtype=tensor.float32);g=garnet.compiled_graph();h=garnet.compiled_graph()
 try:
  g.build(forward(symbol),[symbol],checkpoint);assert g.vision_materialized_weight_slots==(2 if enabled else 0)
  plan=json.loads(g.workspace_plan_json());weights=[e for e in plan['bindings']if e['rows']==3072 and e['columns']==1024]
  assert len(weights)==(2 if enabled else 0)
  if enabled:assert weights[0]['color']==weights[1]['color']and weights[0]['last']<weights[1]['first'];assert not any(e['slot']in plan['protected_slots']for e in weights)
  old=None;old_values=None
  for iteration in range(3):
   values=[float((i*17+iteration*3)%31-15)*.00390625 for i in range(batch*1024)];x=tensor.tensor(values,shape=[batch,1024],dtype=tensor.float32);result=g.run([x]);actual=result.tolist();assert all(math.isfinite(v)for v in actual)and x.tolist()==values
   if old is not None:assert old.tolist()==old_values
   old=result;old_values=actual;bits=struct.pack('<'+'f'*len(actual),*actual)
   records.append(dict(batch=batch,iteration=iteration,values=len(actual),sha256=hashlib.sha256(bits).hexdigest()))
   assert g.vision_weight_expansion_dispatches==(2 if enabled else 0)and g.vision_pooled_weight_slots==(2 if enabled else 0)and g.vision_raw_weight_payload_bytes==(2*3072*1024*2 if enabled else 0);unbound(g)
  saved=g.graph_workspace_allocated_bytes;bad=False
  try:g.run([tensor.tensor([0.],shape=[1],dtype=tensor.float32)])
  except Exception:bad=True
  assert bad and saved==g.graph_workspace_allocated_bytes
  h.build_shared(forward(symbol),[symbol],checkpoint,g)
  union=json.loads(garnet.memory_inventory_json([dict(object=g,tag='parent'),dict(object=h,tag='child')],None))
  masters={tag:{r['buffer_id']for r in union['references']if r['tag']==tag and 'VisionRawBF16'in r['name']}for tag in ['parent','child']}
  if enabled:assert len(masters['parent'])==2 and masters['parent']==masters['child']
  g.close();assert g.vision_materialized_weight_slots==0 and h.run([x]).tolist()==actual;unbound(h)
  g.build(forward(symbol),[symbol],checkpoint);assert g.run([x]).tolist()==actual;unbound(g)
  broken=garnet.compiled_graph();bad=False
  try:broken.build_shared(forward(symbol),[symbol],checkpoint+'.missing',g)
  except Exception:bad=True
  assert bad and not broken.ready and g.run([x]).tolist()==actual;broken.close()
  print('Graph materialization/private scratch/master protection/shared-close/output/input/invalid/rebuild PASS',mode,batch,flush=True)
 finally:g.close();h.close()
(w/('graph-'+mode+'.json')).write_text(json.dumps(records,indent=2))
if enabled:assert records==json.loads((w/'graph-off.json').read_text())
print('Complete projection output bits match OFF control '+mode,flush=True)

# Eligible names with F32 checkpoint dtype retain the original path.
symbol=tensor.input('f32_fallback_input',shape=[1,3],dtype=tensor.float32);g=garnet.compiled_graph()
try:
 g.build(forward(symbol),[symbol],str(w/'f32-fallback.safetensors'));x=tensor.tensor([.25,-.5,.75],shape=[1,3],dtype=tensor.float32);actual=g.run([x]).tolist();wanted=[]
 for row in range(7):
  total=0.
  for layer in [0,1]:
   weight=[float(((row*3+j)%(9 if layer==0 else 7))-(4 if layer==0 else 3))*(.03125 if layer==0 else .0625)for j in range(3)]
   total+=sum(a*b for a,b in zip(weight,[.25,-.5,.75]))
  wanted.append(total)
 assert actual==wanted and g.vision_materialized_weight_slots==0 and g.vision_raw_weight_payload_bytes==0 and g.vision_weight_expansion_dispatches==0
 print('F32 named vision projection fallback exact PASS '+mode,flush=True)
finally:g.close()
