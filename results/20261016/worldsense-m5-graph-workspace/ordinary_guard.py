"""Ordinary graph full-op lifetime, output ownership and shared rebuild correctness."""
import sys,json,math
mode=sys.argv[1];source='/Users/shawn/CantorAI/out/research/worldsense-graph-workspace/source';build='/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-graph-workspace-'+mode;sys.path.insert(0,source);sys.path.insert(0,build)
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T
checkpoint='/Users/shawn/CantorAI/out/models/Qwen3-1.7B/model.safetensors.index.json';enabled=mode=='on'
@T.fusion(name='ordinary_workspace_chain',boundary='required')
def forward(x,y):
 a=x*T.binary_op('add')*y;b=a*T.binary_op('add')*x;c=b*T.binary_op('add')*y;d=c*T.binary_op('add')*x;return d*T.binary_op('add')*y
def unbound(g):
 inv=json.loads(garnet.memory_inventory_json([dict(object=g,tag='ordinary')],None));ids={r['buffer_id']for r in inv['references']if r['role']=='graph_workspace'};assert len(ids)==g.graph_workspace_buffer_count;assert not any(r['buffer_id']in ids for r in inv['references']if r['role']=='slot')
for rows,cols in [(1,7),(2,33),(3,129)]:
 shape=[rows,cols];sy=[tensor.input('ow_'+str(rows)+'_'+str(cols)+'_'+str(i),shape=shape,dtype=tensor.float32)for i in range(2)];g=garnet.compiled_graph();shared=garnet.compiled_graph()
 try:
  g.build(forward(*sy),sy,checkpoint);plan=json.loads(g.workspace_plan_json());assert bool(plan['bindings'])==enabled
  protected=set(plan['protected_slots']);assert not any(e['slot']in protected for e in plan['bindings'])
  for x in plan['bindings']:
   for y in plan['bindings']:
    if x['slot']!=y['slot']and (x['rows'],x['columns'],x['color'])==(y['rows'],y['columns'],y['color']):assert x['last']<y['first']or y['last']<x['first']
  previous=None;previous_values=None
  for i in range(3):
   xv=[float((j+i)%8-3)*.125 for j in range(rows*cols)];yv=[float((j+i)%4-1)*.25 for j in range(rows*cols)];inputs=[tensor.tensor(v,shape=shape,dtype=tensor.float32)for v in [xv,yv]];out=g.run(inputs);values=out.tolist();assert values==[3*x+3*y for x,y in zip(xv,yv)]and all(math.isfinite(v)for v in values);assert [v.tolist()for v in inputs]==[xv,yv]
   if previous is not None:assert previous.tolist()==previous_values,'Returned tensor aliases overwritten root/workspace'
   previous=out;previous_values=values;assert (g.graph_workspace_buffer_count>0)==enabled and (g.graph_workspace_bound_slots>g.graph_workspace_buffer_count)==enabled;unbound(g)
  before=g.graph_workspace_allocated_bytes;rejected=False
  try:g.run([inputs[0],tensor.tensor([0.],shape=[1],dtype=tensor.float32)])
  except Exception:rejected=True
  assert rejected and before==g.graph_workspace_allocated_bytes
  shared.build_shared(forward(*sy),sy,checkpoint,g);g.close();assert g.graph_workspace_allocated_bytes==0 and shared.run(inputs).tolist()==values;unbound(shared)
  g.build(forward(*sy),sy,checkpoint);assert g.graph_workspace_allocated_bytes==0 and g.run(inputs).tolist()==values;unbound(g)
  print('Ordinary full-op graph reuse/output snapshot/input protection/shared close/rebuild PASS',mode,rows,cols,flush=True)
 finally:g.close();shared.close()
