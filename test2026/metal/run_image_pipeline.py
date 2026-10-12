import json,math,sys
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet
from xModel.qwen35.vl_0_8b.metal_vision import MetalImageEncoder
from xModel.qwen35.vl_0_8b.metal_decoder import MetalDecoder
with open(sys.argv[1]) as stream:f=json.load(stream)
vision=None;decoder=None
def compare(actual,wanted):
    assert len(actual)==len(wanted)
    for a,b in zip(actual,wanted):assert math.isfinite(a) and abs(a-b)<=5e-6+5e-5*abs(b),(a,b)
try:
    vision=MetalImageEncoder(f['checkpoint'],f['vision_config'],4,6,f['batch'])
    decoder=MetalDecoder(f['checkpoint'],f['config'],f['batch'],f['capacity'])
    for request in f['requests']:
        decoder.reset();visual=vision.encode_file(request['patch_file']);compare(visual.tolist(),request['visual'])
        for step in request['steps']:
            if step['visual_token'] is None:
                prediction=decoder.step(step['tokens'],active=step['active'],positions=step['positions'])
            else:
                hidden=decoder.io.visual_token(visual,step['visual_token'])
                prediction=decoder.step_hidden(hidden,active=step['active'],positions=step['positions'])
            assert prediction==step['prediction'],(prediction,step['prediction'])
            states=[decoder.layers[0]['state1'],decoder.layers[0]['state2'],decoder.layers[1]['state1'],decoder.layers[1]['state2']]
            for state,wanted in zip(states,step['states']):compare(state.tolist(),wanted)
    print(garnet.backend_info(),'image-to-mixed-decoder/head batch',f['batch'],'PASS two requests, 14 steps each',flush=True)
finally:
    if vision is not None:vision.close()
    if decoder is not None:decoder.close()
    garnet.shutdown()
