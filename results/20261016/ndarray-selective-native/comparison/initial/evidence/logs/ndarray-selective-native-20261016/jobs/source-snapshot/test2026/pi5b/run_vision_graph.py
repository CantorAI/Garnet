import sys,json,math
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import tensor,garnet
from xModel.qwen35.vl_0_8b.vision_encoder import Qwen35VisionEncoder,T
with open(sys.argv[1]) as stream:f=json.load(stream)
symbol=tensor.input('image_patches',shape=f['shape'],dtype=tensor.float32)
@T.fusion(name='qwen35_vision_reference',boundary='required')
def forward(patches):return Qwen35VisionEncoder(patches,f['config'],f['grid_h'],f['grid_w'])
model=garnet.compiled_graph()
try:
    model.build(forward(symbol),[symbol],f['checkpoint']);times=[]
    for step in f['steps']:
        result=model.run([tensor.tensor(step['input'],shape=f['shape'],dtype=tensor.float32)])
        actual=result.tolist();assert len(actual)==len(step['expected'])
        max_error=0;worst_ratio=0;error_squared=0;reference_squared=0;failures=[]
        for index,(observed,wanted) in enumerate(zip(actual,step['expected'])):
            max_error=max(max_error,abs(observed-wanted))
            difference=abs(observed-wanted);bound=5e-6+5e-5*abs(wanted)
            worst_ratio=max(worst_ratio,difference/bound);error_squared+=difference*difference;reference_squared+=wanted*wanted
            if not math.isfinite(observed) or difference>bound:
                if len(failures)<5:failures.append([index,observed,wanted,difference])
        print('vision reference max_abs_error',max_error,'relative_l2',math.sqrt(error_squared/reference_squared),'worst_tolerance_ratio',worst_ratio,flush=True)
        assert not failures,str(failures)
        times.append(model.last_inference_ms)
    print('actual Garnet vision graph batch',f['shape'][0],'grid',f['grid_h'],f['grid_w'],'PASS compile_ms',model.compile_ms,'inference_ms',times,flush=True)
finally:
    model.close()
    garnet.shutdown()
