"""Actual trained attention/head graphs: greedy parity and private-session recovery."""
import json,math,sys
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen.metal_dense.layers import DenseLayer,T,norm as dense_norm
from xModel.qwen35.vl_0_8b.decoder_layer import Qwen35FullDecoderLayer,norm as vl_norm
with open(sys.argv[1]) as stream:f=json.load(stream)
c=f['config'];batch=f['batch'];capacity=f['capacity'];shapes=f['shapes'];layer=3 if f['vl'] else 0
symbols=[tensor.input('fused_session_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
@T.fusion(name='fused_trained_attention_reference',boundary='required')
def forward(hidden,key,value,positions,slots,active):
    if f['vl']:return Qwen35FullDecoderLayer(hidden,key,value,positions,slots,active,c,layer,capacity)
    return DenseLayer(hidden,key,value,positions,slots,active,c,layer,capacity)
head_symbol=tensor.input('fused_session_head',shape=shapes[0],dtype=tensor.float32)
def project(hidden,bf16):
    if f['vl']:return vl_norm(hidden,'model.language_model.norm',c)*T.unary_op('qwen35_tied_head',bf16_weight=bf16)
    name='model.embed_tokens.weight' if c.get('tie_word_embeddings',False) else 'lm_head.weight'
    return dense_norm(hidden,'model.norm',c)*T.unary_op('qwen_text_head',weight_name=name,bf16_weight=bf16)
@T.fusion(name='fused_trained_bf16_head',boundary='required')
def bf16_head(hidden):return project(hidden,1)
@T.fusion(name='fused_trained_float_head_fallback',boundary='required')
def float_head(hidden):return project(hidden,0)
graph=garnet.compiled_graph();head=garnet.compiled_graph();fallback=garnet.compiled_graph()
session=garnet.decoder_session();other=garnet.decoder_session();plain=garnet.decoder_session()
def states(s):return [v.tolist() for v in s.snapshot(0)]
def zero(s):assert all(v==0 for values in states(s) for v in values)
def inputs(name):return [garnet.tensor_io().read_f32(p,s) for p,s in zip(f[name],shapes)]
def run(s,name,predict=0):
    v=inputs(name);return s.run([graph],v[0],v[3],v[4],v[5],predict)
def expected(s,h):
    scores=h.run([s.hidden()]).tolist();width=len(scores)//batch
    assert all(math.isfinite(v) for v in scores)
    result=[]
    for row in range(batch):
        best=0
        for index in range(1,width):
            if scores[row*width+index]>scores[row*width+best]:best=index
        result.append(best)
    return result
try:
    assert 'fused_head_argmax' in garnet.backend_info()
    graph.build_bf16(forward(*symbols),symbols,f['checkpoint']);head.build(bf16_head(head_symbol),[head_symbol],f['checkpoint']);fallback.build(float_head(head_symbol),[head_symbol],f['checkpoint'])
    session.build([graph],head);other.build([graph],head);plain.build([graph],fallback)
    zero(session);zero(other);zero(plain)
    result=run(session,'partial');assert session.fused_head_argmax_dispatches==1 and result==expected(session,head)
    saved=states(session);saved_hidden=session.hidden().tolist()
    run(other,'partial');assert saved==states(session) and saved_hidden==session.hidden().tolist()
    # Generic compiled-graph logits remain complete and reproducible after fusion.
    assert result==expected(session,head)
    result=run(plain,'partial');assert plain.fused_head_argmax_dispatches==0 and expected(plain,fallback)==result
    session.reset();zero(session);assert session.fused_head_argmax_dispatches==0 and states(other)==saved
    run(session,'partial',-1);assert session.fused_head_argmax_dispatches==0
    rejected=False
    try:run(session,'nonfinite')
    except Exception:rejected=True
    assert rejected and session.fused_head_argmax_dispatches==1,'Nonfinite prediction did not execute/detect the fused head'
    rejected=False
    try:run(session,'partial')
    except Exception:rejected=True
    assert rejected,'Poisoned session ran without reset'
    session.reset();zero(session);assert session.fused_head_argmax_dispatches==0
    result=run(session,'partial');assert session.fused_head_argmax_dispatches==1 and result==expected(session,head)
    assert states(other)==saved
    print('fused_head_session_reference PASS',json.dumps(dict(batch=batch,actual_fused_head=True,generic_full_logits_preserved=True,float_head_fallback=True,nonfinite_poison_recovery=True,private_sessions_isolated=True)),flush=True)
finally:
    session.close();other.close();plain.close();graph.close();head.close();fallback.close()
