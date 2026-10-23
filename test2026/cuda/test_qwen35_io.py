"""Native CUDA embedding/head, row copies, reuse and invalid-ID checks."""
import json,sys,os
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
import garnet
with open(sys.argv[1]) as f:fixture=json.loads(f.read())
io=garnet.cuda_tensor_io()
def flat(v):
    if isinstance(v,list):
        result=[]
        for x in v:result.extend(flat(x))
        return result
    return [float(v)]
def values(t):return flat(garnet.tensor_to_cpu(t).tolist())
def check(actual,expected):
    assert len(actual)==len(expected)
    assert all(abs(a-e)<1e-6 for a,e in zip(actual,expected)),(actual,expected)
try:
    assert io.open(fixture['checkpoint'])
    embedding=io.embedding(fixture['tokens']);check(values(embedding),fixture['embedding'])
    hidden=io.from_host(fixture['hidden'],[2,8])
    check(values(io.logits(hidden)),fixture['logits'])
    assert list(io.predict(hidden))==fixture['predictions']
    assert list(io.predict(hidden))==fixture['predictions']
    matrix=io.from_host([float(i) for i in range(28)],[2,14])
    row0=io.row(matrix,0);row1=io.row(matrix,1)
    check(values(row1),[float(i) for i in range(14,28)])
    check(values(io.stack_rows([row1,row0])),[float(i) for i in range(14,28)]+[float(i) for i in range(14)])
    check(values(io.stack_rows([row0])),[float(i) for i in range(14)])
    chunk=io.zeros([2,4,14]);io.write_chunk_token(matrix,chunk,2)
    check(values(chunk),[0]*28+[float(i) for i in range(14)]+[0]*14+[0]*28+[float(i) for i in range(14,28)]+[0]*14)
    flags_chunk=io.from_host([0,0,1,0,0,0,0,0],[2,4]);last=io.from_host([99]*28,[2,14])
    io.copy_last_chunk(chunk,flags_chunk,last);check(values(last),[float(i) for i in range(14)]+[99]*14)
    view=io.zeros([2,2,3]);io.slice_into(matrix,2,view)
    check(values(view),[2,3,4,5,6,7,16,17,18,19,20,21])
    check(values(io.slice(matrix,2,[2,2,3])),values(view))
    io.zero(view);check(values(view),[0]*12)
    visual=io.from_host([float(i) for i in range(24)],[2,3,4])
    check(values(io.visual_token(visual,1)),[4,5,6,7,16,17,18,19])
    src=io.from_host([1,2,3,4,5,6],[2,3]);dst=io.from_host([9]*6,[2,3]);flags=io.from_host([1,0],[2])
    io.copy_active(src,flags,dst);check(values(dst),[1,2,3,9,9,9])
    io.update(flags,[0,1]);io.copy_active(src,flags,dst);check(values(dst),[1,2,3,4,5,6])
    invalid=False
    try:io.embedding([-1,32])
    except BaseException:invalid=True
    assert invalid
    memory=json.loads(io.memory_json());assert memory['embedding_head_shared_bytes']==32*8*2
    print(json.dumps({'event':'io_memory','data':memory}),flush=True)
    print('QWEN35_CUDA_IO_COMPLETE',flush=True)
finally:io.close()
