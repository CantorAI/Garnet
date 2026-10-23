"""XLang3/Garnet synthetic MLP: exercises capture, packed engines and CUDA replay."""
import os,sys,json
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.getcwd())
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T,mlp
checkpoint,output=sys.argv[1:3]
bf16=len(sys.argv)>3 and sys.argv[3]=='bf16'
fp8=len(sys.argv)>3 and sys.argv[3]=='fp8'
expected=3.375 if fp8 else (3.3359375 if bf16 else 3.4296875)
symbol=tensor.input('hidden',shape=[1,256],dtype=tensor.float32)
@T.fusion(name='nvfp4_identity_mlp',boundary='required')
def forward(hidden):
    return mlp(hidden,'model.language_model.layers.0.mlp') * T.binary_op('add') * hidden
graph=garnet.compiled_graph();io=garnet.cuda_tensor_io()
try:
    graph.build(forward(symbol),[symbol],checkpoint)
    info=json.loads(graph.stats_json())
    assert info['checkpoint_weight_precision']==('bf16' if bf16 else 'modelopt_mixed')
    assert info['native_nvfp4_projection_weights']==(0 if bf16 or fp8 else 3)
    assert info['native_fp8_projection_weights']==(3 if fp8 else 0)
    # The SDK may embed BF16 plugin constants despite requested stripping.
    # Record its actual refit capability; numerical checks remain mandatory.
    if not bf16: assert not info['engine_refittable']
    inputs=io.from_host([1.5]*256,[1,256])
    # Disabled run, graph warmup, capture, then two cached graph launches.
    for replay in [False,True,True,True,True]:
        graph.enable_cuda_graph(replay)
        actual=garnet.tensor_to_cpu(graph.run_device([inputs])).tolist()
        assert actual==[expected]*256,actual[:8]
    final=json.loads(graph.stats_json())
    assert final['graph_captures']>=1 and final['graph_replays']>=2 and final['graph_capture_failures']==0,final
    with open(output,'w') as f: json.dump({'stats':final,
        'precision':'bf16' if bf16 else ('fp8_w8a8' if fp8 else 'nvfp4_w4a4'),'exact_output':expected,'outputs':256,'replay_runs':2,
        'scope':'Synthetic residual identity MLP capture/engine/graph correctness; no actual-model or performance claim'},f,indent=2)
    print('GARNET_NATIVE_MLP_CAPTURE_GRAPH_PASS', 'bf16' if bf16 else ('fp8_w8a8' if fp8 else 'nvfp4_w4a4'))
finally: graph.close();io.close()
