"""Verify partial clearing after real chunk-prefill cache reshape copies."""
import json,sys
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen.metal_dense.layers import DensePrefillLayer,T
from xModel.qwen35.vl_0_8b.prefill_layer import PrefillLayer
with open(sys.argv[1]) as stream:f=json.load(stream)
c=f['config'];batch=f['batch'];chunk=2;capacity=8;shapes=f['shapes'];layer=3 if f['vl'] else 0
symbols=[tensor.input('prefix_chunk_reference_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
@T.fusion(name='prefix_chunk_cache_copy_reference',boundary='required')
def forward(hidden,key,value,positions,slots,active):
 if f['vl']:return PrefillLayer(hidden,key,value,positions,slots,active,c,layer,batch,chunk,capacity,'full_attention',parallel_attention=True)
 return DensePrefillLayer(hidden,key,value,positions,slots,active,c,layer,batch,chunk,capacity,'full_attention')
graph=garnet.compiled_graph();session=garnet.decoder_session();other=garnet.decoder_session()
def state(s):return [v.tolist() for v in s.snapshot(0)]
try:
 assert 'dirty_prefix_reset' in garnet.backend_info()
 graph.build_bf16(forward(*symbols),symbols,f['checkpoint']);session.build([graph],None);other.build([graph],None)
 values=[garnet.tensor_io().read_f32(p,s) for p,s in zip(f['inputs'],shapes)]
 for _ in range(2):
  session.run([graph],values[0],values[3],values[4],values[5],-1);other.run([graph],values[0],values[3],values[4],values[5],-1);saved=state(other)
  session.reset();assert all(v==0 for values in state(session) for v in values) and saved==state(other)
  assert session.dirty_prefix_reset_states==2 and session.dirty_prefix_reset_fallback_states==0
  assert session.dirty_prefix_reset_bytes==2*batch*c['num_key_value_heads']*4*c['head_dim']*4
 print('prefix_reset_prefill PASS',json.dumps(dict(batch=batch,chunk=chunk,partial_extent=4,prefill_copy_admitted=True,complete_zero_verified=True,inactive_high_slot_ignored=True,other_session_unchanged=True)),flush=True)
finally:session.close();other.close();graph.close()
