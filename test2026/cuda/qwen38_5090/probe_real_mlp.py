"""Actual pinned MLP correctness via Garnet; not a benchmark."""
import os,sys,json,math
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T,mlp
checkpoint,oracle_path,output=sys.argv[1:4]
with open(oracle_path) as f:oracle=json.load(f)
width=len(oracle['input']);symbol=tensor.input('hidden',shape=[1,width],dtype=tensor.float32)
@T.fusion(name='actual_nvfp4_mlp',boundary='required')
def forward(hidden):return mlp(hidden,'model.language_model.layers.'+str(oracle['layer'])+'.mlp')
graph=garnet.compiled_graph();io=garnet.cuda_tensor_io()
try:
    graph.build(forward(symbol),[symbol],checkpoint)
    inputs=io.from_host(oracle['input'],[1,width]);checks=[]
    for enabled in [False,True,True,True,True]:
        graph.enable_cuda_graph(enabled)
        actual=garnet.tensor_to_cpu(graph.run_device([inputs])).tolist()
        assert len(actual)==len(oracle['expected']) and all(math.isfinite(v) for v in actual)
        errors=[abs(a-b) for a,b in zip(actual,oracle['expected'])]
        relative_l2=(sum(e*e for e in errors)/max(1e-20,sum(v*v for v in oracle['expected'])))**.5
        maximum=max(errors);checks.append({'max_absolute_error':maximum,'relative_l2_error':relative_l2})
        assert relative_l2<=.01 and maximum<=.05,checks[-1]
    stats=json.loads(graph.stats_json());assert stats['graph_replays']>=2 and stats['graph_capture_failures']==0
    with open(output,'w') as f:json.dump({'checks':checks,'stats':stats,'scope':oracle['scope']},f,indent=2)
    print('ACTUAL_NVFP4_MLP_CPU_ORACLE_AND_GRAPH_PASS')
finally:graph.close();io.close()
