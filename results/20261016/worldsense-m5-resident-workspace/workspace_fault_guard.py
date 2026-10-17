"""Fault-only two-layer trained graphs: resource failure isolation and abort recovery."""
import sys,json,math
from pathlib import Path
family=int(sys.argv[1]);source='/Users/shawn/CantorAI/out/research/worldsense-resident-workspace/fault-source';build='/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-resident-workspace-fault';sys.path.insert(0,source);sys.path.insert(0,build)
import garnet,tensor
from xModel.qwen.metal_dense.layers import DenseLayer,T
from xModel.qwen35.vl_0_8b.decoder_layer import Qwen35LinearDecoderLayer
model='/Users/shawn/CantorAI/out/models/'+('Qwen3.5-4B'if family==4 else'Qwen3-1.7B');raw=json.loads(open(model+'/config.json').read());c=raw['text_config']if family==4 else raw;c['head_dim']=c.get('head_dim',c['hidden_size']//c['num_attention_heads']);width=c['hidden_size'];checkpoint=model+'/model.safetensors.index.json'
if family==4:
 channels=2*c['linear_num_key_heads']*c['linear_key_head_dim']+c['linear_num_value_heads']*c['linear_value_head_dim'];shapes=[[1,width],[1,channels,c['linear_conv_kernel_dim']],[1,c['linear_num_value_heads'],c['linear_key_head_dim'],c['linear_value_head_dim']],[1]]
else:
 cache=[1,c['num_key_value_heads'],8,c['head_dim']];shapes=[[1,width],cache,cache,[1,3],[1],[1]]
graphs=[];symbols=[];functions=[];a=garnet.decoder_session();b=garnet.decoder_session();position=tensor.tensor([0.,0.,0.],shape=[1,3],dtype=tensor.float32);slot=tensor.tensor([0.],shape=[1],dtype=tensor.float32);active=tensor.tensor([1.],shape=[1],dtype=tensor.float32)
def function(idx):
 if family==4:
  @T.fusion(name='workspace_fault_linear_'+str(idx),boundary='required')
  def forward(hidden,conv,state,flags):return Qwen35LinearDecoderLayer(hidden,conv,state,flags,c,idx)
 else:
  @T.fusion(name='workspace_fault_dense_'+str(idx),boundary='required')
  def forward(hidden,key,value,positions,slots,flags):return DenseLayer(hidden,key,value,positions,slots,flags,c,idx,8)
 return forward
def values(s):return [[v.tolist()for v in s.snapshot(i)]for i in range(2)]
def controls(hidden,cache):return [hidden,*cache,active]if family==4 else[hidden,*cache,position,slot,active]
def wanted(s,hidden):
 expected=[];current=hidden
 for i,g in enumerate(graphs):
  cache=s.snapshot(i);n1=len(cache[0].tolist());out=g.run(controls(current,cache)).tolist();assert all(math.isfinite(v)for v in out);expected.append([out[width:width+n1],out[width+n1:]]);current=tensor.tensor(out[:width],shape=[1,width],dtype=tensor.float32)
 return expected,current.tolist()
def run(s,hidden):s.run(graphs,hidden,position,slot,active,-1)
def exact(s,expect):assert values(s)==expect[0]and s.hidden().tolist()==expect[1]
def unbound(s):
 inv=json.loads(garnet.memory_inventory_json([dict(object=g,tag='g'+str(i))for i,g in enumerate(graphs)],s));ids={r['buffer_id']for r in inv['references']if r['tag']=='session'and r['role']=='session_workspace'};assert len(ids)==s.workspace_buffer_count,'Workspace inventory/count mismatch';assert not any(r['buffer_id']in ids for r in inv['references']if r['tag'].startswith('g')),'Graph retained session workspace binding'
try:
 for i in range(2):
  sy=[tensor.input('wf_'+str(family)+'_'+str(i)+'_'+str(j),shape=s,dtype=tensor.float32)for j,s in enumerate(shapes)];f=function(i);g=garnet.compiled_graph()
  if i:g.build_shared_resident(f(*sy),sy,checkpoint,graphs[0])
  else:g.build_bf16_resident(f(*sy),sy,checkpoint)
  graphs.append(g);symbols.append(sy);functions.append(f)
 a.build(graphs,None);b.build(graphs,None);hidden=tensor.tensor([float(i%19-9)*.02 for i in range(width)],shape=[1,width],dtype=tensor.float32);expect=wanted(a,hidden);run(a,hidden);exact(a,expect);unbound(a)
 # Force a partial grow after a nonzero completed state, not merely after initialization.
 next_hidden=tensor.tensor([float(i%17-8)*.03 for i in range(width)],shape=[1,width],dtype=tensor.float32);expect=wanted(a,next_hidden);saved=values(a);saved_hidden=a.hidden().tolist();saved_b=values(b);a._test_workspace_fail_after(1);rejected=False
 try:run(a,next_hidden)
 except Exception as e:rejected='Injected workspace allocation failure'in str(e)
 assert rejected and a.workspace_buffer_count==1 and saved==values(a)and saved_hidden==a.hidden().tolist()and saved_b==values(b);unbound(a);run(a,next_hidden);exact(a,expect);unbound(a)
 # State/root allocations succeed, then the second generic scratch allocation fails.
 saved=values(a);saved_hidden=a.hidden().tolist();cache=a.snapshot(0);control=controls(hidden,cache);host=[v.tolist()for v in control];graphs[0]._test_lazy_fail_after(4);rejected=False
 try:graphs[0].run(control)
 except Exception as e:rejected='Injected deferred storage allocation failure'in str(e)
 assert rejected and saved==values(a)and saved_hidden==a.hidden().tolist()and host==[v.tolist()for v in control]and saved_b==values(b)
 expect=wanted(a,hidden);run(a,hidden);exact(a,expect);unbound(a)
 # Fail after layer0 has enqueued work and exchanged state ownership.
 a._test_workspace_encode_layer(1);rejected=False
 try:run(a,next_hidden)
 except Exception as e:rejected='Injected workspace encoder failure'in str(e)
 assert rejected;blocked=False
 try:a.snapshot(0)
 except Exception:blocked=True
 assert blocked,'GPU failure did not poison session';unbound(a);a.reset();assert all(v==0 for layer in values(a)for cache in layer for v in cache)
 expect=wanted(a,next_hidden);run(a,next_hidden);exact(a,expect);unbound(a);assert saved_b==values(b)
 expect_b=wanted(b,hidden);saved=values(a);saved_hidden=a.hidden().tolist();run(b,hidden);exact(b,expect_b);assert saved==values(a)and saved_hidden==a.hidden().tolist();unbound(b)
 print('Workspace '+str(family)+'-input partial-pool-grow/private-nonzero-isolation/generic-scratch-failure/layer1-abort-binding-clear/reset-recovery/two-session exact PASS',flush=True)
finally:
 a.close();b.close()
 for g in graphs:g.close()
