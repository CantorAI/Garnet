import sys,json,math
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import tensor,garnet
from xModel.qwen35.vl_0_8b.metal_vision import MetalImageEncoder
with open(sys.argv[1]) as stream:f=json.load(stream)
encoder=MetalImageEncoder(f['checkpoint'],f['config'],f['grid_h'],f['grid_w'],f['shape'][0])
model=encoder.graph
try:
    times=[]
    for step in f['steps']:
        if 'patch_file' in step:result=encoder.encode_file(step['patch_file'])
        else:result=model.run([tensor.tensor(step['input'],shape=f['shape'],dtype=tensor.float32)])
        expected=(garnet.tensor_io().read_f32(step['expected_file'],[step['expected_count']]).tolist()
            if 'expected_file' in step else step['expected'])
        actual=result.tolist();assert len(actual)==len(expected)
        max_error=0;worst_ratio=0;error_squared=0;reference_squared=0;failures=[]
        for index,(observed,wanted) in enumerate(zip(actual,expected)):
            max_error=max(max_error,abs(observed-wanted))
            difference=abs(observed-wanted);bound=5e-6+5e-5*abs(wanted)
            worst_ratio=max(worst_ratio,difference/bound);error_squared+=difference*difference;reference_squared+=wanted*wanted
            if not math.isfinite(observed) or difference>bound:
                if len(failures)<5:failures.append([index,observed,wanted,difference])
        print('vision reference max_abs_error',max_error,'relative_l2',math.sqrt(error_squared/reference_squared),'worst_tolerance_ratio',worst_ratio,flush=True)
        assert not failures,str(failures)
        tokens=f['grid_h']*f['grid_w']//(f['config']['spatial_merge_size']**2);width=f['config']['out_hidden_size']
        selected=garnet.tensor_io().visual_token(result,tokens-1).tolist()
        wanted=[value for b in range(f['shape'][0]) for value in actual[(b*tokens+tokens-1)*width:(b*tokens+tokens)*width]]
        assert selected==wanted,'visual token batch extraction differs from encoder output'
        times.append(model.last_inference_ms)
    print('actual Garnet vision graph batch',f['shape'][0],'grid',f['grid_h'],f['grid_w'],'PASS compile_ms',model.compile_ms,'inference_ms',times,flush=True)
finally:
    encoder.close()
    garnet.shutdown()
