"""Independent small-model chunk vs serial trajectory, states and decode padding.

Reference fixture is produced by PyTorch/NumPy, not the CUDA implementation.
Keep the same strict tolerances as the serial decoder tests.
"""
import os,sys,json
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet
from xModel.qwen35.vl_0_8b.cuda_decoder import CUDADecoder
with open(sys.argv[1]) as f:fixture=json.loads(f.read())
chunk=int(sys.argv[2]) if len(sys.argv)>2 else 4
capture_expected='no-graph' not in sys.argv[3:]
fused_delta='fused' in sys.argv[3:]
fused_attention='attention' in sys.argv[3:]
projection_workspace_bytes=16384 if 'workspace16k' in sys.argv[3:] else 0
def flat(value):
    if isinstance(value,list):
        result=[]
        for x in value:result.extend(flat(x))
        return result
    return [float(value)]
def check(tensor,expected):
    actual=flat(garnet.tensor_to_cpu(tensor).tolist())
    assert len(actual)==len(expected)
    error=max(abs(a-e) for a,e in zip(actual,expected))
    assert all(abs(a-e)<=2e-5+2e-4*abs(e) for a,e in zip(actual,expected)),('max_abs_error',error)
    return error
model=CUDADecoder(fixture['checkpoint'],fixture['config'],fixture['batch'],fixture['capacity'],chunk,fused_delta,fused_attention,projection_workspace_bytes)
try:
    hidden=[model.io.embedding(s['tokens']) for s in fixture['steps']]
    positions=[[] for b in range(fixture['batch'])];slots=[0]*fixture['batch']
    for s in fixture['steps']:
        for b in range(fixture['batch']):positions[b].append([slots[b]]*3);slots[b]+=s['active'][b]
    prediction=model.chunk_hidden(hidden,[s['active'] for s in fixture['steps']],positions)
    assert list(prediction)==fixture['steps'][-1]['predictions']
    assert model.slots==slots
    errors=[check(model.last_hidden,fixture['steps'][-1]['hidden'])]
    for layer,expected in zip(model.layers,fixture['final_states']):
        errors.extend([check(layer['state1'],expected['state1']),check(layer['state2'],expected['state2'])])
    model.reset()
    model.enable_cuda_graph(capture_expected)
    for step in fixture['steps']:
        prediction=model.step(step['tokens'],step['active'])
        assert list(prediction)==step['predictions']
        errors.append(check(model.last_hidden,step['hidden']))
    for layer,expected in zip(model.layers,fixture['final_states']):
        errors.extend([check(layer['state1'],expected['state1']),check(layer['state2'],expected['state2'])])
    memory=model.memory_info()
    if capture_expected:assert all(g['graph_captures']>0 and g['graph_replays']>0 and g['graph_capture_failures']==0 for g in memory['graphs']),memory
    if os.environ.get('GARNET_TRT_SYNC_ALLOCATOR')=='1':
        assert memory['owned_decoder_bytes']>0 and memory['trt_runtime_allocator_live_bytes']>0,memory
    print(json.dumps({'event':'chunk_memory','data':memory}),flush=True)
    # Oversized chunks and exhausted active rows must reject before changing state.
    invalid=False
    try:model.chunk_hidden(hidden*(chunk+1),[s['active'] for s in fixture['steps']]*(chunk+1),positions)
    except BaseException:invalid=True
    assert invalid
    # Fill the actual attention cache through its final valid slot, then verify
    # rejection is transactional (no state/hidden mutation or poisoned request).
    assert len(set(model.slots))==1
    for step in range(model.capacity-model.slots[0]):model.step(fixture['steps'][-1]['tokens'])
    assert model.slots==[model.capacity]*fixture['batch']
    tensors=[model.last_hidden]+[layer[key] for layer in model.layers for key in ['state1','state2']]
    before=[flat(garnet.tensor_to_cpu(t).tolist()) for t in tensors]
    invalid=False
    try:model.step(fixture['steps'][-1]['tokens'])
    except BaseException:invalid=True
    assert invalid and not model.poisoned and model.slots==[model.capacity]*fixture['batch']
    assert before==[flat(garnet.tensor_to_cpu(t).tolist()) for t in tensors]
    print('QWEN35_CUDA_CHUNK_COMPLETE','chunk',chunk,'fused_delta',fused_delta,'fused_attention',fused_attention,'cuda_graph',capture_expected,'checks',len(errors),'max_abs_error',max(errors),flush=True)
finally:model.close()
