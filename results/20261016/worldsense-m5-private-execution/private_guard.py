"""Explicit private graph policy, shared inheritance, failures and rebuild reset."""
import sys,json
mode=sys.argv[1];source='/Users/shawn/CantorAI/out/research/worldsense-private-execution/source';build='/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-private-execution-'+mode;sys.path.insert(0,source);sys.path.insert(0,build)
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T
checkpoint='/Users/shawn/CantorAI/out/models/Qwen3-1.7B/model.safetensors.index.json'
@T.fusion(name='private_graph_chain',boundary='required')
def forward(x,y):
 a=x*T.binary_op('add')*y;b=a*T.binary_op('add')*x;c=b*T.binary_op('add')*y;d=c*T.binary_op('add')*x;return d*T.binary_op('add')*y
for rows,cols in [(1,7),(2,33),(3,129)]:
 shape=[rows,cols];sy=[tensor.input('private_'+str(rows)+'_'+str(cols)+'_'+str(i),shape=shape,dtype=tensor.float32)for i in range(2)];g=garnet.compiled_graph();h=garnet.compiled_graph()
 try:
  g.build_bf16_private(forward(*sy),sy,checkpoint);assert g.private_storage
  assert not json.loads(g.workspace_plan_json())['bindings']
  old=None;oldvalues=None
  for n in range(3):
   xs=[float((j+n)%8-3)*.125 for j in range(rows*cols)];ys=[float((j+n)%4-1)*.25 for j in range(rows*cols)];inputs=[tensor.tensor(v,shape=shape,dtype=tensor.float32)for v in [xs,ys]];values=[3*x+3*y for x,y in zip(xs,ys)];result=g.run(inputs);assert result.tolist()==values
   assert [v.tolist()for v in inputs]==[xs,ys]
   if old is not None:assert old.tolist()==oldvalues
   old=result;oldvalues=values
  assert g.graph_workspace_allocated_bytes==0 and g.graph_workspace_buffer_count==0 and g.graph_workspace_bound_slots==0
  h.build_shared(forward(*sy),sy,checkpoint,g);assert h.private_storage;g.close();assert not g.private_storage
  assert h.run(inputs).tolist()==values and not json.loads(h.workspace_plan_json())['bindings']
  g.build_bf16_private(forward(*sy),sy,checkpoint);rejected=False
  try:g.build_bf16_private(forward(*sy),sy,checkpoint+'.missing')
  except Exception:rejected=True
  assert rejected and not g.ready and not g.private_storage
  g.build_bf16(forward(*sy),sy,checkpoint);assert not g.private_storage
  assert bool(json.loads(g.workspace_plan_json())['bindings'])==(mode=='on');assert g.run(inputs).tolist()==values
  h.close();h.build_shared(forward(*sy),sy,checkpoint,g);assert not h.private_storage;assert h.run(inputs).tolist()==values
  print('Private buffer ownership/shared inheritance/source close/failure/rebuild reset PASS',mode,rows,cols,flush=True)
 finally:g.close();h.close()
