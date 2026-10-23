"""Native resident decoder trajectory, reset, ragged rows and capacity admission."""
import os,sys,json
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet
from xModel.qwen35.vl_0_8b.cuda_decoder import CUDADecoder
with open(sys.argv[1]) as f:fixture=json.loads(f.read())
model=CUDADecoder(fixture['checkpoint'],fixture['config'],fixture['batch'],fixture['capacity'])
def flat(v):
    if isinstance(v,list):
        result=[]
        for x in v:result.extend(flat(x))
        return result
    return [float(v)]
try:
    for repeat in range(2):
        if repeat:model.reset()
        for step in fixture['steps']:
            prediction=model.step(step['tokens'],step['active'])
            assert list(prediction)==step['predictions'],(prediction,step['predictions'])
            actual=flat(garnet.tensor_to_cpu(model.last_hidden).tolist())
            assert all(abs(a-e)<=2e-5+2e-4*abs(e) for a,e in zip(actual,step['hidden'])),(actual,step['hidden'])
            assert model.slots==step['slots'],(model.slots,step['slots'])
    print('QWEN35_CUDA_DECODER_COMPLETE',flush=True)
finally:model.close()
