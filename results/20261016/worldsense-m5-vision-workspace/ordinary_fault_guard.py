"""Separate fault build: generic pool partial growth/encoder abort and root/input lifetime."""
import sys,json
sys.path.insert(0,'/Users/shawn/CantorAI/out/research/worldsense-vision-workspace/fault-source');sys.path.insert(0,'/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-vision-workspace-fault')
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T
checkpoint='/Users/shawn/CantorAI/out/models/Qwen3-1.7B/model.safetensors.index.json';shape=[3,33];sy=[tensor.input('of_'+str(i),shape=shape,dtype=tensor.float32)for i in range(2)]
@T.fusion(name='ordinary_fault_chain',boundary='required')
def forward(x,y):
 a=x*T.binary_op('add')*y;b=a*T.binary_op('add')*x;c=b*T.binary_op('add')*y;d=c*T.binary_op('add')*x;return d*T.binary_op('add')*y
def unbound(g):
 inv=json.loads(garnet.memory_inventory_json([dict(object=g,tag='graph')],None));ids={r['buffer_id']for r in inv['references']if r['role']=='graph_workspace'};assert len(ids)==g.graph_workspace_buffer_count and not any(r['buffer_id']in ids for r in inv['references']if r['role']=='slot')
g=garnet.compiled_graph()
try:
 g.build(forward(*sy),sy,checkpoint);inputs=[tensor.tensor([float(j%8-3)*.125 for j in range(99)],shape=shape,dtype=tensor.float32),tensor.tensor([.25]*99,shape=shape,dtype=tensor.float32)];host=[v.tolist()for v in inputs];old=g.run(inputs);old_values=old.tolist();unbound(g)
 changed=[tensor.tensor([x+.25 for x in host[0]],shape=shape,dtype=tensor.float32),inputs[1]];want=[3*(x+.25)+3*y for x,y in zip(*host)];g._test_graph_workspace_fail_after(1);rejected=False
 try:g.run(changed)
 except Exception as e:rejected='Injected ordinary workspace allocation failure'in str(e)
 assert rejected and g.graph_workspace_buffer_count==1 and old.tolist()==old_values and [v.tolist()for v in inputs]==host;unbound(g);result=g.run(changed);assert result.tolist()==want;unbound(g)
 g._test_graph_workspace_encode_op(1);rejected=False
 try:g.run(inputs)
 except Exception as e:rejected='Injected ordinary workspace encoder failure'in str(e)
 assert rejected and result.tolist()==want;unbound(g);assert g.run(inputs).tolist()==old_values;unbound(g)
 g.close();assert g.graph_workspace_allocated_bytes==0;g.build(forward(*sy),sy,checkpoint);assert g.run(inputs).tolist()==old_values;unbound(g)
 print('Ordinary pool partial grow/root+input preservation/encoder abort/unbinding/recovery/rebuild PASS',flush=True)
finally:g.close()
