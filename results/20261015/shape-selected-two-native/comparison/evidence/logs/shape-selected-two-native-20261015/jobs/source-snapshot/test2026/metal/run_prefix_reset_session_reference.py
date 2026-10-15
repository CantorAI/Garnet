"""Native partial reset, unknown-writer fallback and poisoned-head recovery."""
import json,sys,math
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen.metal_dense.layers import DenseLayer,T,reshape
from xModel.qwen35.vl_0_8b.decoder_layer import Qwen35FullDecoderLayer,add
with open(sys.argv[1]) as stream:f=json.load(stream)
c=f['config'];batch=f['batch'];capacity=f['capacity'];shapes=f['shapes'];layer=3 if f['vl'] else 0
symbols=[tensor.input('prefix_recovery_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
@T.fusion(name='prefix_real_attention_reference',boundary='required')
def forward(hidden,key,value,positions,slots,active):
    if f['vl']:return Qwen35FullDecoderLayer(hidden,key,value,positions,slots,active,c,layer,capacity)
    return DenseLayer(hidden,key,value,positions,slots,active,c,layer,capacity)
@T.fusion(name='prefix_unknown_cache_writer',boundary='required')
def generic(hidden,key,value,positions,slots,active):
    result=hidden * T.binary_op('qwen35_concat_sequence') * reshape(add(key,key),[batch,-1])
    return result * T.binary_op('qwen35_concat_sequence') * reshape(add(value,value),[batch,-1])
head_symbol=tensor.input('prefix_failure_head',shape=shapes[0],dtype=tensor.float32)
@T.fusion(name='prefix_nonfinite_failure_head',boundary='required')
def head_forward(hidden):return add(hidden,hidden)
graph=garnet.compiled_graph();unknown=garnet.compiled_graph();head=garnet.compiled_graph();session=garnet.decoder_session();other=garnet.decoder_session()
def states(s):return [v.tolist() for v in s.snapshot(0)]
def zero(s):assert all(v==0 for values in states(s) for v in values)
def inputs(name):return [garnet.tensor_io().read_f32(p,s) for p,s in zip(f[name],shapes)]
def run(s,g,name,predict=-1):
    v=inputs(name);return s.run([g],v[0],v[3],v[4],v[5],predict)
try:
    assert 'dirty_prefix_reset' in garnet.backend_info()
    graph.build_bf16(forward(*symbols),symbols,f['checkpoint']);unknown.build(generic(*symbols),symbols,f['checkpoint']);head.build(head_forward(head_symbol),[head_symbol],f['checkpoint'])
    session.build([graph],head);other.build([graph],head);zero(session);zero(other)
    run(session,graph,'partial');run(other,graph,'partial');saved=states(other)
    session.reset();zero(session);assert saved==states(other)
    assert session.dirty_prefix_reset_states==2 and session.dirty_prefix_reset_fallback_states==0
    assert session.dirty_prefix_reset_bytes==2*batch*c['num_key_value_heads']*4*c['head_dim']*4
    run(session,graph,'partial');run(session,unknown,'partial');session.reset();zero(session)
    assert session.dirty_prefix_reset_states==0 and session.dirty_prefix_reset_fallback_states==2
    full_bytes=2*batch*c['num_key_value_heads']*capacity*c['head_dim']*4
    assert session.dirty_prefix_reset_bytes==full_bytes and saved==states(other)
    rejected=False
    try:run(session,graph,'nonfinite',0)
    except Exception:rejected=True
    assert rejected,'Nonfinite prediction did not fail the request'
    rejected=False
    try:run(session,graph,'partial')
    except Exception:rejected=True
    assert rejected,'Poisoned session ran without reset'
    session.reset();zero(session)
    assert session.dirty_prefix_reset_states==0 and session.dirty_prefix_reset_fallback_states==2 and session.dirty_prefix_reset_bytes==full_bytes
    run(session,graph,'partial');session.reset();zero(session)
    assert session.dirty_prefix_reset_states==2 and session.dirty_prefix_reset_bytes==2*batch*c['num_key_value_heads']*4*c['head_dim']*4
    assert saved==states(other)
    print('prefix_reset_recovery PASS',json.dumps(dict(batch=batch,partial_extent=4,capacity=capacity,partial_clear_verified=True,unknown_writer_full_fallback=True,nonfinite_request_poisoned=True,full_recovery_verified=True,private_sessions_isolated=True)),flush=True)
finally:
    session.close();other.close();graph.close();unknown.close();head.close()
