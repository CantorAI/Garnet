"""Bounded full vision FP32 arithmetic diagnostic; never the serving/quantized arm."""
import json,math,os,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet,tensor
from xModel.qwen38.vl_27b.model import validate_config
from xModel.qwen35.vl_0_8b.tensor_compat import tensor as compat
checkpoint,fixture,output=sys.argv[1:4]
with open(os.path.join(checkpoint,'config.json')) as f:config=json.load(f)
with open(os.path.join(fixture,'oracle.json')) as f:oracle=json.load(f)
validate_config(config);v=config['vision_config'];T=compat()
assert oracle['grid']==[1,16,16] and oracle['reference_dtype'].startswith('CPU float32')
symbol=tensor.input('diagnostic_vision_fp32_patches',shape=oracle['patch_shape'],dtype=tensor.float32)
@T.fusion(name='diagnostic_qwen38_fp32_vision',boundary='required')
def forward(patches):
    return patches*T.unary_op('qwen35_vision_encoder',grid_h=16,grid_w=16,
        hidden_size=v['hidden_size'],num_heads=v['num_heads'],depth=v['depth'],
        patch_size=v['patch_size'],temporal_patch_size=v['temporal_patch_size'],in_channels=v['in_channels'],
        spatial_merge_size=v['spatial_merge_size'],num_position_embeddings=v['num_position_embeddings'],
        out_hidden_size=v['out_hidden_size'],rope_theta=10000,diagnostic_fp32_projections=1)
graph=garnet.compiled_graph();io=garnet.cuda_tensor_io()
try:
    graph.build(forward(symbol),[symbol],checkpoint)
    patches=io.read_f32(os.path.join(fixture,'patches.f32'),oracle['patch_shape'])
    expected=garnet.tensor_to_cpu(io.read_f32(os.path.join(fixture,'expected.f32'),oracle['output_shape'])).tolist();checks=[]
    for enabled in [False,True,True,True,True]:
        graph.enable_cuda_graph(enabled);features=graph.run_device([patches])
        if not checks:
            with open(output+'.f32','wb') as f:f.write(io.finite_f32_bytes(features))
        actual=garnet.tensor_to_cpu(features).tolist()
        assert len(actual)==len(expected) and all(math.isfinite(x) for x in actual)
        errors=[abs(a-b) for a,b in zip(actual,expected)]
        checks.append({'max_absolute_error':max(errors),'relative_l2_error':(sum(e*e for e in errors)/sum(x*x for x in expected))**.5})
    result={'checks':checks,'stats':json.loads(graph.stats_json()),'oracle':oracle,
        'projection_precision':'Diagnostic FP32 constants/projections; original BF16 profile unchanged',
        'scope':'Structural full vision arithmetic diagnostic only; not BF16 vision parity, VLM quality or performance'}
    with open(output,'w') as f:json.dump(result,f,indent=2)
    for row in checks:
        assert row['max_absolute_error']<=.05 and row['relative_l2_error']<=.001,row
    assert result['stats']['graph_replays']>=2 and result['stats']['graph_capture_failures']==0
    print('NATIVE_FP32_VISION_STRUCTURE_PASS',checks,flush=True)
finally:graph.close();io.close()
