"""Actual full vision tower compared with a fixed CPU oracle; not a benchmark."""
import json,math,os,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
from xModel.qwen38.vl_27b.model import create_cuda_vision
checkpoint,fixture,output=sys.argv[1:4]
with open(os.path.join(checkpoint,'config.json')) as f:config=json.load(f)
with open(os.path.join(fixture,'oracle.json')) as f:oracle=json.load(f)
io=garnet.cuda_tensor_io();encoder=None
try:
    patches=io.read_f32(os.path.join(fixture,'patches.f32'),oracle['patch_shape'])
    expected=garnet.tensor_to_cpu(io.read_f32(os.path.join(fixture,'expected.f32'),oracle['output_shape'])).tolist()
    encoder=create_cuda_vision(checkpoint,config,oracle['grid'][1],oracle['grid'][2]);reports=[]
    for enabled in [False,True,True,True,True]:
        encoder.graph.enable_cuda_graph(enabled)
        features=encoder.encode_patches(patches)
        if not reports:
            with open(output+'.f32','wb') as f:f.write(io.finite_f32_bytes(features))
        actual=garnet.tensor_to_cpu(features).tolist()
        assert len(actual)==len(expected) and all(math.isfinite(v) for v in actual)
        errors=[abs(a-b) for a,b in zip(actual,expected)]
        rel=(sum(v*v for v in errors)/max(1e-20,sum(v*v for v in expected)))**.5
        reports.append({'max_absolute_error':max(errors),'relative_l2_error':rel})
    result={'checks':reports,'stats':json.loads(encoder.graph.stats_json()),'oracle':oracle,
        'scope':'Full vision tower only on CPU-produced patches; no native JPEG path, full VLM or performance claim'}
    with open(output,'w') as f:json.dump(result,f,indent=2)
    for report in reports:
        assert report['relative_l2_error']<=oracle['acceptance']['relative_l2_max'],report
        assert report['max_absolute_error']<=oracle['acceptance']['max_absolute_error_max'],report
    assert result['stats']['graph_replays']>=2 and result['stats']['graph_capture_failures']==0
    print('NATIVE_FULL_VISION_ORACLE_PASS',reports,flush=True)
finally:
    if encoder is not None:encoder.close()
    io.close()
