import sys,json,math
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T,norm,mlp,add
with open(sys.argv[1]) as stream:f=json.load(stream)
symbol=tensor.input('hidden',shape=f['shape'],dtype=tensor.float32)
active=tensor.input('active',shape=[f['shape'][0]],dtype=tensor.float32)
@T.fusion(name='metal_qwen_mlp_head_reference',boundary='required')
def forward(hidden,active):
    hidden=hidden * T.binary_op('qwen35_mask_rows') * active
    value=norm(hidden,'model.language_model.norm',dict(rms_norm_eps=1e-6))
    value=add(hidden,mlp(value,'model.language_model.layers.0.mlp'))
    return value * T.unary_op('qwen35_tied_head')
graph=garnet.compiled_graph()
try:
    graph.build(forward(symbol,active),[symbol,active],f['checkpoint'])
    for _ in range(2):
        result=graph.run([tensor.tensor(f['hidden'],shape=f['shape'],dtype=tensor.float32),
            tensor.tensor(f['active'],shape=[f['shape'][0]],dtype=tensor.float32)])
        actual=result.tolist()
        assert len(actual)==len(f['expected'])
        for x,y in zip(actual,f['expected']):
            assert math.isfinite(x) and abs(x-y)<=2e-5+5e-5*abs(y),(x,y)
    print(garnet.backend_info(),'actual captured Qwen mask/norm/MLP/residual/head batch',f['shape'][0],
          'PASS compile_ms',graph.compile_ms,'run_ms',graph.last_inference_ms,flush=True)
finally:graph.close()
