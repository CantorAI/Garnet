import sys,json,time,math
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen35.vl_0_8b.metal_decoder import MetalDecoder
with open(sys.argv[1]) as stream:f=json.load(stream)
model=MetalDecoder(f['checkpoint'],f['config'],f['config']['batch'],16)
try:
    assert model.io.zeros([2,3]).tolist()==[0,0,0,0,0,0]
    source=tensor.tensor([1,7,3,5,2,5],shape=[2,3],dtype=tensor.float32)
    assert model.io.argmax(source)==[1,0]
    assert model.io.slice(source,1,[2,2]).tolist()==[7,3,2,5]
    for step in f['steps']:
        embedding=model.io.embedding(step['tokens']).tolist()
        assert embedding==step['embedding']
        prediction=model.step(step['tokens'],active=step['active'])
        assert prediction==step['prediction'],(prediction,step['prediction'])
        for actual,wanted in zip(model.layers[0]['state1'].tolist(),step['conv']):
            assert abs(actual-wanted)<=5e-6+5e-5*abs(wanted),(actual,wanted)
        for actual,wanted in zip(model.layers[0]['state2'].tolist(),step['state']):
            assert abs(actual-wanted)<=5e-6+5e-5*abs(wanted),(actual,wanted)
    # Compare serving-level chunking with sequential execution, then switch
    # profiles twice to check cache replacement, retained state and partial tails.
    batch=f['config']['batch'];sequence=[[3+i%5+b for b in range(batch)] for i in range(7)]
    model.reset()
    for i,tokens in enumerate(sequence):expected_prediction=model.step(tokens,predict=i==6)
    expected_states=[model.layers[0]['state1'].tolist(),model.layers[0]['state2'].tolist()]
    for chunk in (2,4,8,16,2):
        model.reset();model.enable_prefill(chunk)
        hidden=[model.io.embedding(tokens) for tokens in sequence]
        prediction=model.prefill_sequence(hidden,[[[i]*3 for i in range(7)] for b in range(batch)])
        assert prediction==expected_prediction,'Chunked serving prediction differs'
        for actual,wanted in zip([model.layers[0]['state1'].tolist(),model.layers[0]['state2'].tolist()],expected_states):
            for x,y in zip(actual,wanted):assert math.isfinite(x) and abs(x-y)<=5e-6+5e-5*abs(y),(chunk,x,y)
        assert model.slots==[7]*batch,'Chunked serving slots differ'
    print('Metal chunk serving lifecycle batch',batch,'PASS chunks 2/4/8/16/2, partial tails and profile replacement',flush=True)
    # The adapter accepts already GPU-projected vocabulary logits only.
    rejected=False
    try:model.io.logits(model.io.zeros([f['config']['batch'],f['config']['hidden_size']]))
    except Exception:rejected=True
    assert rejected,'Unprojected hidden rows were accepted as logits'
    print(garnet.backend_info(),'closed-loop MetalDecoder batch',f['config']['batch'],
        'PASS steps',len(f['steps']),'last_layer_ms',model.last_layer_ms,'last_head_ms',model.last_head_ms,flush=True)
finally:model.close()
