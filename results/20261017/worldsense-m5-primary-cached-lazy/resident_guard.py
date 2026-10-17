"""Actual decode graphs: exact explicit/resident parity and private cache isolation."""
import sys,json,math
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen.metal_dense.layers import DenseLayer,T
from xModel.qwen35.vl_0_8b.decoder_layer import Qwen35FullDecoderLayer
with open(sys.argv[1]) as stream:f=json.load(stream)
c=f['config'];batch=f['batch'];capacity=f['capacity'];layer=f['layer'];shapes=f['shapes']
symbols=[tensor.input('kv_private_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
@T.fusion(name='private_kv_decode_reference',boundary='required')
def forward(hidden,key,value,positions,slots,active):
    if f['vl']:return Qwen35FullDecoderLayer(hidden,key,value,positions,slots,active,c,layer,capacity)
    return DenseLayer(hidden,key,value,positions,slots,active,c,layer,capacity)
graph=garnet.compiled_graph();session=garnet.decoder_session();other=garnet.decoder_session();built=False
enabled='resident_kv_inplace' in garnet.backend_info();expected_updates=2 if enabled else 0
prefix_reset='dirty_prefix_reset' in garnet.backend_info();dirty_extent=0
lazy_enabled='lazy_resident_defaults'in garnet.backend_info();assert ('cached_matrix_metadata'in garnet.backend_info())==lazy_enabled
def snapshot(s):return [v.tolist() for v in s.snapshot(0)]
def collect(s):
    hidden=s.hidden().tolist();key,value=snapshot(s);packed=[]
    for row in range(batch):
        for values in (hidden,key,value):packed+=values[row*(len(values)//batch):(row+1)*(len(values)//batch)]
    return packed
try:
    graph.build_bf16(forward(*symbols),symbols,f['checkpoint']);session.build([graph],None);built=True
    for index,files in enumerate(f['steps']):
        values=[garnet.tensor_io().read_f32(p,s) for p,s in zip(files,shapes)]
        initial=session.snapshot(0);control=[values[0],initial[0],initial[1],values[3],values[4],values[5]]
        host_before=[v.tolist() for v in control]
        wanted=graph.run(control).tolist();assert all(math.isfinite(v) for v in wanted)
        session.run([graph],values[0],values[3],values[4],values[5],-1)
        if prefix_reset:
            for slot,flag in zip(values[4].tolist(),values[5].tolist()):
                if flag:dirty_extent=max(dirty_extent,int(slot)+1)
        assert collect(session)==wanted,'Private update differs from explicit graph output/state'
        if enabled:assert session.kv_inplace_updates==2,'Private update was not executed'
        if index==0:
            other.build([graph],None)
        assert (graph.lazy_default_retired_bytes>0)==lazy_enabled
        # Neither the caller-owned tensors nor a second session may be changed.
        assert host_before==[v.tolist() for v in control],'Explicit host inputs were mutated'
        saved=snapshot(session);hidden=session.hidden().tolist()
        other.run([graph],values[0],values[3],values[4],values[5],-1)
        assert saved==snapshot(session) and hidden==session.hidden().tolist(),'Shared graph changed another session'
        assert graph.run(control).tolist()==wanted,'Temporary graph output bindings leaked'
        assert (graph.lazy_default_rehydrated_bytes>0)==lazy_enabled
        print('private_kv_step',json.dumps(dict(index=index,batch=batch,capacity=capacity,layer=layer,inplace_updates=expected_updates,exact_output_and_state=True)),flush=True)
    for files in f['invalid']:
        values=[garnet.tensor_io().read_f32(p,s) for p,s in zip(files,shapes)];saved=snapshot(session);rejected=False
        try:session.run([graph],values[0],values[3],values[4],values[5],-1)
        except Exception:rejected=True
        assert rejected and saved==snapshot(session),'Invalid controls changed private caches'
        before=graph.lazy_default_rehydrated_bytes;rejected=False
        bad_control=[values[0],*session.snapshot(0),values[3],values[4],values[5]]
        try:graph.run(bad_control)
        except Exception:rejected=True
        assert rejected and graph.lazy_default_rehydrated_bytes==before and saved==snapshot(session),'Invalid generic input allocated or changed state'
    saved=snapshot(other);session.reset()
    assert all(v==0 for state in snapshot(session) for v in state) and saved==snapshot(other)
    if prefix_reset:
        assert session.dirty_prefix_reset_states==2 and session.dirty_prefix_reset_fallback_states==0
        assert session.dirty_prefix_reset_bytes==2*batch*c['num_key_value_heads']*dirty_extent*c['head_dim']*4
        session.reset();assert session.dirty_prefix_reset_bytes==0 and all(v==0 for state in snapshot(session) for v in state)
    values=[garnet.tensor_io().read_f32(p,s) for p,s in zip(f['steps'][0],shapes)]
    graph.close();rejected=False
    try:session.run([graph],values[0],values[3],values[4],values[5],-1)
    except Exception:rejected=True
    assert rejected,'Closed graph accepted resident call'
    graph.build_bf16(forward(*symbols),symbols,f['checkpoint']);assert graph.lazy_default_retired_bytes==0 and graph.lazy_default_rehydrated_bytes==0 and graph.lazy_default_planned_slots==0
    session.build([graph],None)
    control=[values[0],*session.snapshot(0),values[3],values[4],values[5]];wanted=graph.run(control).tolist()
    session.run([graph],values[0],values[3],values[4],values[5],-1);assert collect(session)==wanted,'Rebuilt graph parity failed'
    print('private_kv_reference PASS',json.dumps(dict(batch=batch,capacity=capacity,layer=layer,enabled=enabled,steps=len(f['steps']),invalid_controls=len(f['invalid']),shared_graph_isolated=True)),flush=True)
finally:
    if built:session.reset();other.reset()
    graph.close();session.close();other.close()
