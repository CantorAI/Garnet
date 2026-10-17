"""Two real vision projections: materialized-weight liveness and immutable sharing."""
import sys,json,math,pathlib,hashlib,struct
mode='fault';w=pathlib.Path(__file__).resolve().parent;s='/Users/shawn/CantorAI/out/research/worldsense-private-execution/fault-source';sys.path.insert(0,s);sys.path.insert(0,'/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-private-execution-'+mode)
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T
checkpoint='/Users/shawn/CantorAI/out/models/Qwen3.5-4B/model.safetensors.index.json';enabled=True
@T.fusion(name='streamed_two_vision_projection_guard',boundary='required')
def forward(x):
 a=x*T.unary_op('qwen_text_head',weight_name='model.visual.blocks.0.attn.qkv.weight',bf16_weight=0)
 b=x*T.unary_op('qwen_text_head',weight_name='model.visual.blocks.1.attn.qkv.weight',bf16_weight=0)
 return a*T.binary_op('add')*b

def digest(values):return hashlib.sha256(struct.pack('<'+'f'*len(values),*values)).hexdigest()
def unbound(g):
 inv=json.loads(garnet.memory_inventory_json([dict(object=g,tag='current')],None));ids={r['buffer_id']for r in inv['references']if r['role']=='graph_workspace'}
 assert not any(r['buffer_id']in ids for r in inv['references']if r['role']=='slot')
 masters={r['buffer_id']for r in inv['references']if 'VisionRawBF16'in r['name']}
 assert len(masters)==2 and not masters.intersection(ids)
 return masters
records=[]
for batch in [1,3]:
 symbol=tensor.input('weight_fault_'+str(batch),shape=[batch,1024],dtype=tensor.float32);g=garnet.compiled_graph()
 try:
  g.build(forward(symbol),[symbol],checkpoint)
  values=[float((i*17)%31-15)*.00390625 for i in range(batch*1024)];x=tensor.tensor(values,shape=[batch,1024],dtype=tensor.float32)
  old=g.run([x]);expected=old.tolist();masters=unbound(g)
  reference=[r for r in json.loads((w/'graph-off.json').read_text())if r['batch']==batch and r['iteration']==0][0];assert digest(expected)==reference['sha256']
  g._test_graph_workspace_fail_after(1);rejected=False
  try:g.run([x])
  except Exception as e:rejected='Injected ordinary workspace allocation failure'in str(e)
  assert rejected and g.graph_workspace_buffer_count==1 and old.tolist()==expected and x.tolist()==values and unbound(g)==masters
  assert g.run([x]).tolist()==expected and unbound(g)==masters
  # Op0 expansion, op1 MPS projection, op2 next expansion, op3 next MPS, op4 sum.
  for op in [1,2,3,4]:
   g._test_graph_workspace_encode_op(op);rejected=False
   try:g.run([x])
   except Exception as e:rejected='Injected ordinary workspace encoder failure'in str(e)
   assert rejected and g.vision_weight_expansion_dispatches==(1 if op<3 else 2)
   assert old.tolist()==expected and x.tolist()==values and unbound(g)==masters
   assert g.run([x]).tolist()==expected and unbound(g)==masters
   records.append(dict(batch=batch,abort_before_op=op,exact_recovery=True))
  g.close();g.build(forward(symbol),[symbol],checkpoint);assert g.run([x]).tolist()==expected;unbound(g)
 finally:g.close()
(w/'weight-fault-result.json').write_text(json.dumps(records,indent=2));print('Weight expansion partial pool growth and four queued expansion/MPS abort boundaries exact recovery PASS',flush=True)
