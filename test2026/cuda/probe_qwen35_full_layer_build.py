"""Isolate one official full-attention layer and an explicit workspace cap."""
import json,os,sys,time
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T
from xModel.qwen35.vl_0_8b.cuda_chunk import Qwen35CUDAChunk
checkpoint,output=sys.argv[1:3];workspace=int(sys.argv[3]);batch=8;chunk=4;capacity=384
if len(sys.argv)>4:batch=int(sys.argv[4])
if len(sys.argv)>5:capacity=int(sys.argv[5])
config=json.load(open(os.path.join(checkpoint,'config.json')))['text_config']
shapes=[[batch,chunk,config['hidden_size']]]+[[batch,config['num_key_value_heads'],capacity,config['head_dim']]]*2+[[batch,chunk,3],[batch,chunk],[batch,chunk]]
symbols=[tensor.input('probe_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
@T.fusion(name='qwen35_cuda_decoder_layer',boundary='required')
def forward(*a):return Qwen35CUDAChunk(a[0],a[1],a[2],a[5],a[3],a[4],config,3,'full_attention')
graph=garnet.compiled_graph();graph.set_builder_workspace_bytes(workspace<<20)
start=time.monotonic()
try:
    graph.build(forward(*symbols),symbols,checkpoint)
    result={'workspace_mib':workspace,'batch':batch,'chunk':chunk,'capacity':capacity,
        'compile_seconds':time.monotonic()-start,'stats':json.loads(graph.stats_json()),
        'limitation':'One layer build only; no full model residency, inference correctness or process peak measurement'}
    with open(output,'w') as stream:json.dump(result,stream,indent=2)
    print('QWEN35_FULL_LAYER_BUILD_COMPLETE',json.dumps(result),flush=True)
finally:graph.close()
