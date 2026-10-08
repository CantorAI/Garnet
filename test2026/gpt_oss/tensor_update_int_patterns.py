"""XLang3: exact integer patterns, ownership and no-partial-write negatives."""
import json
from pathlib import Path
import sys
import garnet as G

output=Path(sys.argv[1])
if output.exists():raise FileExistsError(output)
devices=json.loads(G.cuda_devices_json())
if not devices:raise RuntimeError('CUDA GPU required')
records=[]
def read(tensors):return [G.tensor_to_cpu(tensor).tolist() for tensor in tensors]
def rejected(tensors,patterns):
    try:result=G.tensor_update_int_patterns_async(tensors,patterns)
    except Exception:return
    assert result is False
for device in devices[:2]:
    previous=G.cuda_set_device(device['id'])
    try:
        for batch in (1,7,224,512):
            for sequence in (1,3,8,16):
                lengths=[batch*sequence,batch*sequence,batch,batch]
                types=['int64','int64','int32','int32']
                tensors=[G.tensor_from_host([-12345]*length,dtype=dtype,shape=[length],device='cuda')
                         for length,dtype in zip(lengths,types)]
                for changed in (False,True):
                    patterns=[[13+3*column+(100 if changed else 0) for column in range(sequence)],
                              [511+column+(100 if changed else 0) for column in range(sequence)],
                              [sequence-1 if changed else sequence],[-91 if changed else 511]]
                    frozen=[row[:] for row in patterns]
                    expected=[frozen[0]*batch,frozen[1]*batch,frozen[2]*batch,frozen[3]*batch]
                    assert G.tensor_update_int_patterns_async(tensors,patterns)
                    # Mutate the caller's lists immediately, before any readback.
                    # CUDA must consume by-value arguments, never these lists.
                    patterns[0][0]=-999
                    actual=read(tensors)
                    assert actual==expected
                    records.append(dict(device=device['id'],batch=batch,sequence=sequence,changed=changed,
                        types=types,lengths=lengths,patterns=frozen,readbacks=actual))
                    for invalid in ([[],frozen[1],frozen[2],frozen[3]],
                                    [frozen[0],frozen[1],frozen[2],[2**31]],
                                    [frozen[0],frozen[1],frozen[2],[-2**31-1]],
                                    [frozen[0],frozen[1],frozen[2],[1.5]]):
                        rejected(tensors,invalid);assert read(tensors)==expected
                    rejected(tensors,frozen[:3]);assert read(tensors)==expected
                    rejected([tensors[0],tensors[0]],[[1],[2]]);assert read(tensors)==expected
                    try:result=G.tensor_update_int_patterns_async(tensors,frozen,unexpected=1)
                    except Exception:pass
                    else:assert result is False
                    assert read(tensors)==expected
        # Full payload boundary and exact signed INT32 endpoints in INT64/INT32.
        values=[-2**31 if i%2 else 2**31-1 for i in range(768)]
        boundary=[G.tensor_from_host([0]*1536,dtype=dtype,shape=[1536],device='cuda') for dtype in ['int64','int32']]
        for dtype,tensor in zip(['int64','int32'],boundary):
            assert G.tensor_update_int_patterns_async([tensor],[values])
            assert read([tensor])==[values*2]
            records.append(dict(device=device['id'],boundary=True,types=[dtype],
                lengths=[1536],patterns=[values],readbacks=read([tensor])))
            rejected([tensor],[values+[1]]);assert read([tensor])==[values*2]
            rejected([tensor],[[1]*5]);assert read([tensor])==[values*2]
        cpu=G.tensor_from_host([0],dtype='int32',shape=[1],device='cpu')
        rejected([cpu],[[1]]);assert read([cpu])==[[0]]
        floating=G.tensor_from_host([0.],dtype='float32',shape=[1],device='cuda')
        rejected([floating],[[1]]);assert read([floating])==[[0.]]
        rejected([],[]);rejected([1],[[1]])
        # Each pattern fits individually; only their combined payload overflows.
        rejected([boundary[0],boundary[1],tensors[2]],[[1]*384,[2]*384,[3]])
        assert read(boundary)==[values*2,values*2]
        if len(devices)>1:
            target=next(row['id'] for row in devices[:2] if row['id']!=device['id'])
            G.cuda_set_device(target)
            try:rejected([boundary[0]],[[1]])
            finally:G.cuda_set_device(device['id'])
            assert read([boundary[0]])==[values*2]
        print('integer pattern update passed GPU',device['id'],flush=True)
    finally:G.cuda_set_device(previous)
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(dict(protocol='integer-pattern-readbacks-v1',devices=devices[:2],
    records=records,scope='Exact persisted integer outputs and source-bound alias/range/type/no-borrow/no-partial-write assertions'),indent=2))
print('INTEGER_PATTERN_READBACKS',str(output),len(records),flush=True)
