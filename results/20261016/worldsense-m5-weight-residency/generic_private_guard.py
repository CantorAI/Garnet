"""Two real vision projections: materialized-weight liveness and immutable sharing."""
import sys,json,math,pathlib,hashlib,struct
mode=sys.argv[1];w=pathlib.Path(__file__).resolve().parent;s='/Users/shawn/CantorAI/out/research/worldsense-weight-residency/source';sys.path.insert(0,s);sys.path.insert(0,'/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-weight-residency-'+mode)
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T
checkpoint='/Users/shawn/CantorAI/out/models/Qwen3.5-4B/model.safetensors.index.json';enabled=mode in ['on','fault']
@T.fusion(name='streamed_two_vision_projection_guard',boundary='required')
def forward(x):
 a=x*T.unary_op('qwen_text_head',weight_name='model.visual.blocks.0.attn.qkv.weight',bf16_weight=0)
 b=x*T.unary_op('qwen_text_head',weight_name='model.visual.blocks.1.attn.qkv.weight',bf16_weight=0)
 return a*T.binary_op('add')*b

symbol=tensor.input('private_generic',shape=[1,3],dtype=tensor.float32);x=tensor.tensor([.25,-.5,.75],shape=[1,3],dtype=tensor.float32);ordinary=garnet.compiled_graph();private=garnet.compiled_graph();shared=garnet.compiled_graph();cp=str(w/'f32-fallback.safetensors')
try:
 ordinary.build(forward(symbol),[symbol],cp);private.build_private(forward(symbol),[symbol],cp)
 assert not ordinary.private_storage and private.private_storage
 assert not json.loads(private.workspace_plan_json())['bindings']
 a=ordinary.run([x]);b=private.run([x]);assert a.tolist()==b.tolist()
 assert private.private_execution_runs==(1 if enabled else 0)and private.scoped_execution_runs==0
 assert ordinary.private_execution_runs==0 and ordinary.scoped_execution_runs==(1 if enabled else 0)
 if mode=='fault':
  before=private.private_execution_runs;old=b.tolist();private._test_graph_workspace_encode_op(1);rejected=False
  try:private.run([x])
  except Exception as e:rejected='Injected ordinary workspace encoder failure'in str(e)
  assert rejected and private.private_execution_runs==before and b.tolist()==old and x.tolist()==[.25,-.5,.75]
  assert private.run([x]).tolist()==old and private.private_execution_runs==before+1
 shared.build_shared(forward(symbol),[symbol],cp,private);assert shared.private_storage;private.close()
 assert shared.run([x]).tolist()==a.tolist() and shared.private_execution_runs==(1 if enabled else 0)
 before=shared.private_execution_runs;rejected=False
 try:shared.run([tensor.tensor([0.],shape=[1],dtype=tensor.float32)])
 except Exception:rejected=True
 assert rejected and shared.private_execution_runs==before and shared.run([x]).tolist()==a.tolist()
 shared.close();assert not shared.private_storage and shared.private_execution_runs==0
 shared.build(forward(symbol),[symbol],cp);assert not shared.private_storage and shared.run([x]).tolist()==a.tolist()
 rejected=False
 try:shared.build_private(forward(symbol),[symbol],cp+'.missing')
 except Exception:rejected=True
 assert rejected and not shared.ready and not shared.private_storage and shared.private_execution_runs==0
 print('Generic private FP32 parity/actual scope route/shared close/invalid/rebuild/failure reset PASS '+mode,flush=True)
finally:ordinary.close();private.close();shared.close()
