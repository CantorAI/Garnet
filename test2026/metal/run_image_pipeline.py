import json,math,sys
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet
from xModel.qwen35.vl_0_8b.metal_vision import MetalImageEncoder
from xModel.qwen35.vl_0_8b.metal_decoder import MetalDecoder
with open(sys.argv[1]) as stream:f=json.load(stream)
vision=None;decoder=None
io=garnet.tensor_io()
def compare(actual,wanted):
    assert len(actual)==len(wanted)
    for a,b in zip(actual,wanted):assert math.isfinite(a) and abs(a-b)<=5e-6+5e-5*abs(b),(a,b)
try:
    vision=MetalImageEncoder(f['checkpoint'],f['vision_config'],4,6,f['batch'])
    decoder=MetalDecoder(f['checkpoint'],f['config'],f['batch'],f['capacity'])
    for request in f['requests']:
        decoder.reset();visual=vision.encode_file(request['patch_file']);compare(visual.tolist(),io.read_f32(request['visual_file'],[request['visual_count']]).tolist())
        for step in request['steps']:
            if step['visual_token'] is None:
                prediction=decoder.step(step['tokens'],active=step['active'],positions=step['positions'])
            else:
                hidden=decoder.io.visual_token(visual,step['visual_token'])
                prediction=decoder.step_hidden(hidden,active=step['active'],positions=step['positions'])
            assert prediction==step['prediction'],(prediction,step['prediction'])
            first=decoder.session.snapshot(0);second=decoder.session.snapshot(1)
            states=[first[0],first[1],second[0],second[1]]
            for state,wanted in zip(states,step['state_files']):compare(state.tolist(),io.read_f32(wanted['path'],[wanted['count']]).tolist())
    print(garnet.backend_info(),'image-to-mixed-decoder/head batch',f['batch'],'PASS two requests, 14 steps each',flush=True)
finally:
    if vision is not None:vision.close()
    if decoder is not None:decoder.close()
    garnet.shutdown()
