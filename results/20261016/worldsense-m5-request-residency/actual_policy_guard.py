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

private=garnet.compiled_graph();ordinary=garnet.compiled_graph();heads=[garnet.compiled_graph(),garnet.compiled_graph()];session=garnet.decoder_session();enabled='private_execution'in garnet.backend_info();records=[]
headsymbol=tensor.input('actual_head_hidden',shape=shapes[0],dtype=tensor.float32)
@T.fusion(name='actual_private_execution_head',boundary='required')
def headforward(x):return x*T.binary_op('add')*x
try:
 private.build_bf16_private(forward(*symbols),symbols,f['checkpoint']);ordinary.build_bf16_resident(forward(*symbols),symbols,f['checkpoint'])
 heads[0].build(headforward(headsymbol),[headsymbol],f['checkpoint']);heads[1].build_private(headforward(headsymbol),[headsymbol],f['checkpoint']);assert not heads[0].private_storage and heads[1].private_storage
 values=[garnet.tensor_io().read_f32(p,s)for p,s in zip(f['steps'][0],shapes)];saved=[v.tolist()for v in values]
 baseline=None
 for headindex in [1,0,1]:
  session.build([private],heads[headindex])
  for graph in [private,ordinary,private]:
   session.reset();before=[session.private_execution_runs,session.scoped_execution_runs]
   prediction=session.run([graph],values[0],values[3],values[4],values[5],0)
   result=[session.hidden().tolist(),*[v.tolist()for v in session.snapshot(0)],list(prediction)]
   if baseline is None:baseline=result
   assert result==baseline and saved==[v.tolist()for v in values]
   wanted_private=enabled and graph.private_storage and heads[headindex].private_storage
   assert session.private_execution_runs-before[0]==int(wanted_private)
   assert session.scoped_execution_runs-before[1]==int(enabled and not wanted_private)
   records.append(dict(private_layer=graph.private_storage,private_head=headindex==1,private_selected=wanted_private))
  before=[session.private_execution_runs,session.scoped_execution_runs];rejected=False
  try:session.run([],values[0],values[3],values[4],values[5],0)
  except Exception:rejected=True
  assert rejected and before==[session.private_execution_runs,session.scoped_execution_runs]
  if len(sys.argv)>4 and sys.argv[4]=='fault':
   before=[session.private_execution_runs,session.scoped_execution_runs];private._test_graph_workspace_encode_op(1);rejected=False
   try:session.run([private],values[0],values[3],values[4],values[5],0)
   except Exception as e:rejected='Injected ordinary workspace encoder failure'in str(e)
   assert rejected and before==[session.private_execution_runs,session.scoped_execution_runs]and saved==[v.tolist()for v in values]
   rejected=False
   try:session.run([private],values[0],values[3],values[4],values[5],0)
   except Exception as e:rejected='failed Metal session'in str(e)
   assert rejected
   session.reset();prediction=session.run([private],values[0],values[3],values[4],values[5],0)
   assert [session.hidden().tolist(),*[v.tolist()for v in session.snapshot(0)],list(prediction)]==baseline
  heads[headindex].close();heads[headindex].build_private(headforward(headsymbol),[headsymbol],f['checkpoint']);rejected=False
  try:session.run([private],values[0],values[3],values[4],values[5],0)
  except Exception as e:rejected='head was closed/rebuilt'in str(e)
  assert rejected
  session.close();assert session.private_execution_runs==0 and session.scoped_execution_runs==0
  # Restore this head's original construction policy for next matrix row.
  if headindex==0:heads[0].build(headforward(headsymbol),[headsymbol],f['checkpoint'])
 print('Actual alternate layer list/private and ordinary heads/route counters/exact state+prediction/input/head generation recovery PASS',json.dumps(records),flush=True)
finally:
 session.close();private.close();ordinary.close()
 for head in heads:head.close()
