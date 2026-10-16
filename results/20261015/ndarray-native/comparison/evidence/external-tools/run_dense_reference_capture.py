"""Native execution/state checks; numeric comparison runs separately in NumPy."""
import sys,json
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen.metal_dense.layers import DensePrefillLayer,T
with open(sys.argv[1]) as stream:f=json.load(stream)
c=f['config'];batch=f['batch'];chunk=f['chunk'];capacity=f['capacity'];shapes=f['shapes'];idx=f['layer']
symbols=[tensor.input('capture_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
@T.fusion(name='metal_dense_reference_capture',boundary='required')
def forward(hidden,key,value,positions,slots,active):
    return DensePrefillLayer(hidden,key,value,positions,slots,active,c,idx,batch,chunk,capacity,'full_attention')
graph=garnet.compiled_graph();session=garnet.decoder_session();captured=[]
try:
    if len(sys.argv)>4 and sys.argv[4]=='bf16':graph.build_bf16(forward(*symbols),symbols,f['checkpoint'])
    else:graph.build(forward(*symbols),symbols,f['checkpoint'])
    session.build([graph],None)
    for files in f['invalid_inputs']:
        inputs=[garnet.tensor_io().read_f32(path,s) for path,s in zip(files,shapes)]
        rejected=False
        try:graph.run(inputs)
        except Exception:rejected=True
        assert rejected,'Invalid graph controls accepted'
        before=[v.tolist() for v in session.snapshot(0)];rejected=False
        try:session.run([graph],inputs[0],inputs[3],inputs[4],inputs[5],-1)
        except Exception:rejected=True
        assert rejected,'Invalid resident controls accepted'
        assert before==[v.tolist() for v in session.snapshot(0)],'Rejected input changed state'
    for step in f['steps']:
        inputs=[garnet.tensor_io().read_f32(path,s) for path,s in zip(step['inputs'],shapes)]
        state=session.snapshot(0);control=list(inputs);control[1]=state[0];control[2]=state[1]
        explicit=graph.run(control).tolist()
        session.run([graph],inputs[0],inputs[3],inputs[4],inputs[5],-1)
        states=session.snapshot(0)
        captured.append(dict(explicit=explicit,hidden=session.hidden().tolist(),key=states[0].tolist(),value=states[1].tolist(),actual=graph.run(inputs).tolist()))
    session.reset();reset=[v.tolist() for v in session.snapshot(0)]
    graph.close();rejected=False
    try:session.run([graph],inputs[0],inputs[3],inputs[4],inputs[5],-1)
    except Exception:rejected=True
    assert rejected,'Closed graph accepted'
    with open(sys.argv[1]+'.capture.json','w') as stream:stream.write(json.dumps(dict(steps=captured,reset=reset,invalid_controls_rejected=True,closed_graph_rejected=True)))
finally:
    session.close();graph.close();garnet.shutdown()
