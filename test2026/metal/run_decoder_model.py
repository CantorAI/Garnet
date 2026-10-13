import sys,json,time,math
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen35.vl_0_8b.metal_decoder import MetalDecoder
with open(sys.argv[1]) as stream:f=json.load(stream)
model=MetalDecoder(f['checkpoint'],f['config'],f['config']['batch'],16,resident=False)
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
    model.reset();model.enable_resident()
    for chunk in (2,4,8,16,2):
        model.reset();model.enable_prefill(chunk)
        prediction=model.prefill_sequence([model.io.embedding(tokens) for tokens in sequence],[[[i]*3 for i in range(7)] for b in range(batch)])
        assert prediction==expected_prediction,'Resident chunk serving prediction differs'
        for state,wanted in zip(model.session.snapshot(0),expected_states):
            for x,y in zip(state.tolist(),wanted):assert math.isfinite(x) and abs(x-y)<=5e-6+5e-5*abs(y),(chunk,x,y)
        assert model.slots==[7]*batch
    model.reset()
    for step in f['steps']:
        assert model.step(step['tokens'],active=step['active'])==step['prediction'],'Resident paused-row prediction differs'
        for state,wanted in zip(model.session.snapshot(0),[step['conv'],step['state']]):
            for x,y in zip(state.tolist(),wanted):assert math.isfinite(x) and abs(x-y)<=5e-6+5e-5*abs(y),(x,y)
    saved=[state.tolist() for state in model.session.snapshot(0)];saved_hidden=model.session.hidden().tolist()
    other=garnet.decoder_session()
    try:
        other.build([model.layers[0]['graph']],model.head)
        tokens=f['steps'][0]['tokens'];position=tensor.tensor([0]*batch*3,shape=[batch,3],dtype=tensor.float32)
        slots=tensor.tensor([0]*batch,shape=[batch],dtype=tensor.float32);flags=tensor.tensor([1]*batch,shape=[batch],dtype=tensor.float32)
        other.run([model.layers[0]['graph']],model.io.embedding(tokens),position,slots,flags,0)
        assert saved==[state.tolist() for state in model.session.snapshot(0)],'Sessions shared recurrent state'
        assert saved_hidden==model.session.hidden().tolist(),'Another session overwrote hidden snapshot'
        other_saved=[state.tolist() for state in other.snapshot(0)];model.reset()
        assert all(v==0 for state in model.session.snapshot(0) for v in state.tolist()),'Resident reset retained state'
        assert other_saved==[state.tolist() for state in other.snapshot(0)],'Reset changed another session'
    finally:other.close()
    model.reset();rejected=False
    try:model.step_hidden(tensor.tensor([float('nan')]*batch*f['config']['hidden_size'],shape=[batch,f['config']['hidden_size']],dtype=tensor.float32))
    except Exception:rejected=True
    assert rejected,'Nonfinite resident logits accepted'
    rejected=False
    try:model.step(f['steps'][0]['tokens'])
    except Exception:rejected=True
    assert rejected,'Failed resident request resumed without reset'
    model.reset()
    assert model.step(f['steps'][0]['tokens'])==f['steps'][0]['prediction'],'Resident reset did not recover failed request'
    default_model=MetalDecoder(f['checkpoint'],f['config'],batch,16)
    try:
        assert default_model.session is not None,'Default Metal adapter omitted resident optimization'
        default_model.enable_prefill();assert default_model.chunk==16,'Default Metal prefill profile differs'
        assert default_model.step(f['steps'][0]['tokens'])==f['steps'][0]['prediction'],'Default Metal adapter prediction differs'
    finally:default_model.close()
    print('Metal resident serving PASS batch',batch,'paused rows, reset, profile replacement and shared-graph session isolation',flush=True)
    # The adapter accepts already GPU-projected vocabulary logits only.
    rejected=False
    try:model.io.logits(model.io.zeros([f['config']['batch'],f['config']['hidden_size']]))
    except Exception:rejected=True
    assert rejected,'Unprojected hidden rows were accepted as logits'
    print(garnet.backend_info(),'closed-loop MetalDecoder batch',f['config']['batch'],
        'PASS steps',len(f['steps']),'last_layer_ms',model.last_layer_ms,'last_head_ms',model.last_head_ms,flush=True)
finally:model.close()
