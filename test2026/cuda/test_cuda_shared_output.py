"""Exact-output CUDA storage ownership, binding and lifecycle boundaries."""
import json,os,struct,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T
checkpoint,output=sys.argv[1:3]
io=garnet.cuda_tensor_io();io.open(checkpoint)
graphs=[]
def bits(value):
    values=garnet.tensor_to_cpu(value).tolist();return struct.pack('<'+str(len(values))+'f',*values)
def reject(action):
    rejected=False
    try:action()
    except BaseException:rejected=True
    assert rejected,'Invalid storage action was accepted'
def graph(shape):
    x=tensor.input('shared_storage_input',shape=shape,dtype=tensor.float32)
    @T.fusion(name='shared_storage_exact_add',boundary='required')
    def forward(value):return value+value
    g=garnet.compiled_graph();graphs.append(g);g.build(forward(x),[x],checkpoint)
    g.enable_cuda_graph(True);g.enable_serial_graph_reuse(True);return g
try:
    pool=io.zeros([2,8]);io.update(pool,[float(i) for i in range(16)])
    wide=io.storage_view(pool,[2,8]);small=io.storage_view(pool,[2,4]);saved=bits(pool)
    for shape in ([2,0],[-1,4],[3,8]):reject(lambda:io.storage_view(pool,shape))
    reject(lambda:io.storage_view(small,[2,8]))
    assert bits(pool)==saved
    unbuilt=garnet.compiled_graph();reject(lambda:unbuilt.set_output_storage(wide));unbuilt.close()
    a=graph([2,8]);b=graph([2,4]);control=graph([2,8])
    reject(lambda:a.set_output_storage(small))
    reject(lambda:a.set_output_storage(garnet.tensor_to_cpu(wide)))
    reject(lambda:a.set_output_storage(tensor.input('not_materialized',shape=[2,8],dtype=tensor.float32)))
    assert not json.loads(a.stats_json())['external_output_storage']
    a.set_output_storage(wide);b.set_output_storage(small)
    reject(lambda:a.set_output_storage(wide))
    reject(lambda:b.run_device([small]));assert bits(pool)==saved
    reject(lambda:b.run([io.zeros([2,4])]))
    assert json.loads(b.stats_json())['graph_captures']==0
    x=io.from_host([float(i) for i in range(16)],[2,8]);y=io.from_host([float(20+i) for i in range(8)],[2,4])
    expected_wide=struct.pack('<16f',*[2.*i for i in range(16)])
    expected_small=struct.pack('<8f',*[2.*(20+i) for i in range(8)])
    # Drop all caller pool/view owners; graph views retain the actual storage.
    del pool,wide,small
    for trial in range(5):
        out=a.run_device([x]);assert bits(out)==expected_wide
        old=control.run_device([x]);assert bits(old)==expected_wide
        out=b.run_device([y]);assert bits(out)==expected_small
    sa=json.loads(a.stats_json());sb=json.loads(b.stats_json());sc=json.loads(control.stats_json())
    assert sa['device_output_bytes']==64 and sb['device_output_bytes']==32
    assert sa['owned_device_output_bytes']==sb['owned_device_output_bytes']==0
    assert sa['external_output_storage_span_bytes']==sb['external_output_storage_span_bytes']==64
    assert sc['owned_device_output_bytes']==sc['device_output_bytes']==64
    assert all(s['contexts']==1 and s['graph_captures']>0 and s['graph_replays']>0 and not s['graph_capture_failures'] for s in (sa,sb,sc))
    a.close();io.close();out=b.run_device([y]);assert bits(out)==expected_small
    b.close();assert bits(out)==expected_small
    result={'all_passed':True,'interleaved_trials':5,'wide':sa,'small':sb,'control':sc,
        'invalid_shape_device_symbolic_rebinding_overlap_and_host_run_rejected':True,
        'caller_owner_drop_and_graph_io_close_preserve_remaining_alias_owner':True,
        'scope':'Native exact-output/storage lifetime test; not model timing, cross-thread serving or physical peak memory'}
    with open(output,'w') as f:json.dump(result,f,indent=2)
    print('CUDA_SHARED_OUTPUT_COMPLETE',flush=True)
finally:
    for g in graphs:g.close()
    io.close()
