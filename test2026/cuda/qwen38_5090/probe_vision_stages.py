"""Locate numerical divergence with diagnostic prefix towers; not serving admission."""
import json,math,os,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
from xModel.qwen38.vl_27b.model import validate_config
from xModel.qwen35.vl_0_8b.cuda_decoder import CUDAImageEncoder
checkpoint,fixture,output=sys.argv[1:4]
with open(os.path.join(checkpoint,'config.json')) as f:config=json.load(f)
with open(os.path.join(fixture,'oracle.json')) as f:oracle=json.load(f)
validate_config(config);reports=[];io=garnet.cuda_tensor_io()
try:
    patches=io.read_f32(os.path.join(fixture,'patches.f32'),oracle['patch_shape'])
    for depth in [1,3,9]:
        spec=oracle['diagnostic_stages'][str(depth)]
        expected=garnet.tensor_to_cpu(io.read_f32(os.path.join(fixture,spec['file']),oracle['output_shape'])).tolist()
        vision=dict(config['vision_config']);vision['depth']=depth
        encoder=CUDAImageEncoder(checkpoint,vision,16,16,1)
        try:
            features=encoder.encode_patches(patches)
            with open(output+'.depth'+str(depth)+'.f32','wb') as f:f.write(io.finite_f32_bytes(features))
            actual=garnet.tensor_to_cpu(features).tolist()
            assert len(actual)==len(expected) and all(math.isfinite(v) for v in actual)
            errors=[abs(a-b) for a,b in zip(actual,expected)]
            rel=(sum(e*e for e in errors)/max(1e-20,sum(v*v for v in expected)))**.5
            row={'depth':depth,'max_absolute_error':max(errors),'relative_l2_error':rel}
            reports.append(row);print('VISION_PREFIX_DIAGNOSTIC',row,flush=True)
            with open(output,'w') as f:json.dump({'checks':reports,'oracle':oracle,'scope':'Diagnostic prefix towers plus real merger; does not establish full vision/VLM admission'},f,indent=2)
        finally:encoder.close()
    for row in reports:
        assert row['relative_l2_error']<=oracle['acceptance']['relative_l2_max'] and row['max_absolute_error']<=oracle['acceptance']['max_absolute_error_max'],row
finally:io.close()
