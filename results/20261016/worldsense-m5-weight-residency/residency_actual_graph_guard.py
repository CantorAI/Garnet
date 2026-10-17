"""Correctness-only actual Run graph selection and independent explicit graph oracle."""
import sys,json,math
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen.metal_dense.layers import DenseLayer,T
f=json.loads(open(sys.argv[1]).read());c=f['config'];shapes=f['shapes'];batch=f['batch'];width=c['hidden_size']
symbols=[tensor.input('residency_actual_'+str(i),shape=s,dtype=tensor.float32)for i,s in enumerate(shapes)]
@T.fusion(name='residency_actual_layer',boundary='required')
def forward(h,k,v,p,s,a):return DenseLayer(h,k,v,p,s,a,c,f['layer'],f['capacity'])
private=garnet.compiled_graph();ordinary=garnet.compiled_graph();heads=[garnet.compiled_graph(),garnet.compiled_graph()];session=garnet.decoder_session();records=[];enabled='weight_residency_4gib'in garnet.backend_info();fault=len(sys.argv)>4 and sys.argv[4]=='fault'
hs=tensor.input('residency_actual_head',shape=[batch,width],dtype=tensor.float32)
@T.fusion(name='residency_actual_head_graph',boundary='required')
def head(h):return h*T.binary_op('add')*h
def packed():
 arrays=[session.hidden().tolist(),*[v.tolist()for v in session.snapshot(0)]];out=[]
 for row in range(batch):
  for a in arrays:out+=a[row*(len(a)//batch):(row+1)*(len(a)//batch)]
 return out
try:
 private.build_bf16_private(forward(*symbols),symbols,f['checkpoint']);ordinary.build(forward(*symbols),symbols,f['checkpoint'])
 heads[0].build(head(hs),[hs],f['checkpoint']);heads[1].build_private(head(hs),[hs],f['checkpoint'])
 values=[garnet.tensor_io().read_f32(p,s)for p,s in zip(f['steps'][0],shapes)];host=[v.tolist()for v in values]
 for h in heads:
  session.build([private],h)
  for g in [private,ordinary,private]:
   session.reset();control=[values[0],*session.snapshot(0),values[3],values[4],values[5]];wanted=g.run(control).tolist()
   prediction=session.run([g],values[0],values[3],values[4],values[5],0);assert packed()==wanted and host==[v.tolist()for v in values]
   logits=h.run([tensor.tensor(session.hidden().tolist(),shape=[batch,width],dtype=tensor.float32)]).tolist();assert list(prediction)==[max(range(width),key=lambda i:logits[row*width+i])for row in range(batch)]
   r=json.loads(session.weight_residency_stats_json);assert r['active_bytes']==0
   if not enabled:assert r['admitted']==0 and r['fallback']==0
   elif g is ordinary:assert r['attempted_count']==0 and r['reason']==2
   else:
    assert r['attempted_count']>0 and r['attempted_bytes']>0
    assert r['reason']==(5 if fault else 1)
   if fault:assert all(r[k]==0 for k in ['test_owners','test_set_count','test_attached','test_requested','test_global_used'])
   records.append(dict(private_layer=g is private,head_private=h.private_storage,residency=r,explicit_oracle_exact=True))
  session.close()
 print('Actual Run BF16/F32 alternate graph and head residency/explicit state+prediction parity PASS',json.dumps(records),flush=True)
finally:
 session.close();private.close();ordinary.close()
 for h in heads:h.close()
