"""Trained four-input Qwen3.5 linear layer, correctness only; no speed claim."""
import sys,json,math
mode=sys.argv[1];source='/Users/shawn/CantorAI/out/research/worldsense-private-resident-defaults/source';sys.path.insert(0,source);sys.path.insert(0,'/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-private-resident-defaults-'+mode)
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T,Qwen35LinearDecoderLayer
model='/Users/shawn/CantorAI/out/models/Qwen3.5-4B';c=json.loads(open(model+'/config.json').read())['text_config'];checkpoint=model+'/model.safetensors.index.json';channels=2*c['linear_num_key_heads']*c['linear_key_head_dim']+c['linear_num_value_heads']*c['linear_value_head_dim'];shapes=[[1,c['hidden_size']],[1,channels,c['linear_conv_kernel_dim']],[1,c['linear_num_value_heads'],c['linear_key_head_dim'],c['linear_value_head_dim']],[1]]
symbols=[tensor.input('linear_admission_'+str(i),shape=s,dtype=tensor.float32)for i,s in enumerate(shapes)]
@T.fusion(name='four_input_linear_admission',boundary='required')
def forward(hidden,conv,state,active):return Qwen35LinearDecoderLayer(hidden,conv,state,active,c,0)
def state(s):return [v.tolist()for v in s.snapshot(0)]
def packed(s):
 caches=state(s);return s.hidden().tolist()+caches[0]+caches[1]
graph=garnet.compiled_graph();a=garnet.decoder_session();b=garnet.decoder_session();shared=garnet.compiled_graph();other=garnet.decoder_session();enabled='private_resident_defaults' in garnet.backend_info();position=tensor.tensor([0.,0.,0.],shape=[1,3],dtype=tensor.float32);slot=tensor.tensor([0.],shape=[1],dtype=tensor.float32)
try:
 graph.build_bf16_private_resident(forward(*symbols),symbols,checkpoint);assert (graph.resident_avoided_storage_bytes>0)==enabled
 inv=json.loads(garnet.memory_inventory_json([dict(object=graph,tag='linear')],None));assert any(r['legacy_output']for r in inv['references'])==(not enabled)
 for i in [1,2]:assert any(r['role']=='slot'and r['slot']==i for r in inv['references'])==(not enabled)
 a.build([graph],None)
 for i,flag in enumerate([1.,1.,0.,1.,0.]):
  hidden=tensor.tensor([float((j+i)%23-11)*.01 for j in range(c['hidden_size'])],shape=shapes[0],dtype=tensor.float32);active=tensor.tensor([flag],shape=[1],dtype=tensor.float32);control=[hidden,*a.snapshot(0),active];host=[v.tolist()for v in control]
  if mode=='fault'and i==0:
   saved=state(a);before=graph.lazy_default_rehydrated_bytes;graph._test_lazy_fail_after(1);rejected=False
   try:graph.run(control)
   except Exception as e:rejected='Injected deferred storage allocation failure'in str(e)
   assert rejected and graph.lazy_default_rehydrated_bytes>before and saved==state(a) and host==[v.tolist()for v in control]
  want=graph.run(control).tolist();assert all(math.isfinite(v)for v in want)
  a.run([graph],hidden,position,slot,active,-1);assert (a.workspace_bound_slots>0)==False;assert (a.workspace_allocated_bytes>0)==False;assert packed(a)==want and host==[v.tolist()for v in control]
  if i==0:b.build([graph],None)
  saved=state(a);h=a.hidden().tolist();b.run([graph],hidden,position,slot,active,-1);assert saved==state(a)and h==a.hidden().tolist()
  assert graph.run(control).tolist()==want
 # Restore deferred slots before invalid-shape check so accidental allocation is detectable.
 a.run([graph],hidden,position,slot,active,-1);before=graph.lazy_default_rehydrated_bytes;saved=state(a);rejected=False
 try:graph.run([hidden,*a.snapshot(0),tensor.tensor([1.,1.],shape=[2],dtype=tensor.float32)])
 except Exception:rejected=True
 assert rejected and graph.lazy_default_rehydrated_bytes==before and saved==state(a)
 rejected=False
 try:a.run([graph],hidden,position,slot,tensor.tensor([2.],shape=[1],dtype=tensor.float32),-1)
 except Exception:rejected=True
 assert rejected and saved==state(a)
 active=tensor.tensor([1.],shape=[1],dtype=tensor.float32)
 # Shared-weight construction retains its own owners after closing source graph.
 shared.build_shared_resident(forward(*symbols),symbols,checkpoint,graph);assert shared.private_storage;assert not json.loads(shared.workspace_plan_json())['bindings'];assert (shared.resident_avoided_storage_bytes>0)==enabled;other.build([shared],None);control=[hidden,*other.snapshot(0),active];want=shared.run(control).tolist();graph.close();other.run([shared],hidden,position,slot,active,-1);assert packed(other)==want
 graph.build_bf16_private_resident(forward(*symbols),symbols,checkpoint);assert graph.lazy_default_rehydrated_bytes==0 and graph.lazy_default_retired_bytes==0;a.build([graph],None);control=[hidden,*a.snapshot(0),active];want=graph.run(control).tolist();a.run([graph],hidden,position,slot,active,-1);assert (a.workspace_bound_slots>0)==False;assert (a.workspace_allocated_bytes>0)==False;assert packed(a)==want
 head=garnet.compiled_graph();with_head=garnet.decoder_session()
 try:
  hs=tensor.input('ordinary_pooled_head',shape=[1,c['hidden_size']],dtype=tensor.float32)
  @T.fusion(name='ordinary_pooled_head',boundary='required')
  def headfn(h):
   x=h*T.binary_op('add')*h;y=x*T.binary_op('add')*h;z=y*T.binary_op('add')*h;return z*T.binary_op('add')*h
  head.build(headfn(hs),[hs],checkpoint);head.run([hidden]);assert (head.graph_workspace_buffer_count>0)==(mode!='off')
  with_head.build([graph],head);control=[hidden,*with_head.snapshot(0),active];want=graph.run(control).tolist();wh=tensor.tensor(want[:c['hidden_size']],shape=[1,c['hidden_size']],dtype=tensor.float32);logits=head.run([wh]).tolist();prediction=with_head.run([graph],hidden,position,slot,active,0)
  assert prediction==[logits.index(max(logits))]and packed(with_head)==want,'Ordinary pooled head to direct encoding fallback failed'
 finally:with_head.close();head.close()
 print('Four-input trained linear initial omission/generic-resident parity/host-private isolation/inactive/malformed/shared-after-source-close/rebuild PASS '+mode,flush=True)
finally:
 a.close();b.close();other.close();graph.close();shared.close()
