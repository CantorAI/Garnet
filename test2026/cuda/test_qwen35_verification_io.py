"""Multi-token head bounds/ownership and bitwise GPU snapshot/copy checks."""
import json,os,struct,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
import garnet
fixture=json.load(open(sys.argv[1]));io=garnet.cuda_tensor_io();io.open(fixture['checkpoint'])
def cpu(t):return garnet.tensor_to_cpu(t).tolist()
def bits(t):
    values=cpu(t);return struct.pack('<'+str(len(values))+'f',*values)
def caches():
    m=json.loads(io.memory_json());return {k:m[k] for k in ('head_input_bytes','head_output_bytes','token_head_input_bytes','token_head_output_bytes')}
try:
    values=[-0.,1.,-2.,3.,-4.,5.]*4;source=io.from_host(values,[2,3,4]);snapshot=io.clone(source)
    assert bits(source)==bits(snapshot)
    io.zero(snapshot);assert bits(source)==struct.pack('<24f',*values) and all(x==0 for x in cpu(snapshot))
    io.copy_into(source,snapshot);assert bits(source)==bits(snapshot)
    io.copy_into(source,source);assert bits(source)==bits(snapshot)
    wrong=io.zeros([3,8]);before=bits(wrong);rejected=False
    try:io.copy_into(source,wrong)
    except BaseException:rejected=True
    assert rejected and bits(wrong)==before,'Invalid copy changed destination'
    for rows in fixture['rows']:
        hidden=io.from_host(fixture['values'][:rows*fixture['width']],[rows,fixture['width']])
        actual=list(io.predict_tokens(hidden));assert actual==fixture['expected'][:rows],('Multi-token head',rows)
        if rows<=8:assert list(io.predict(hidden))==actual
        else:
            before=caches();rejected=False
            try:io.predict(hidden)
            except BaseException:rejected=True
            assert rejected and caches()==before,'Legacy head limit/cache changed'
    cube=io.from_host(fixture['values'],[8,16,fixture['width']]);assert list(io.predict_tokens(cube))==fixture['expected']
    before=caches();invalid=io.zeros([129,fixture['width']]);rejected=False
    try:io.predict_tokens(invalid)
    except BaseException:rejected=True
    assert rejected and caches()==before,'Expanded head limit/cache changed'
    expected_token_input=128*fixture['width']*2;expected_token_output=128*64*2
    m=json.loads(io.memory_json());assert m['token_head_input_bytes']==expected_token_input and m['token_head_output_bytes']==expected_token_output,m
    io.close();m=json.loads(io.memory_json());assert m['embedding_head_shared_bytes']==m['head_input_bytes']==m['head_output_bytes']==0,m
    print('QWEN35_VERIFICATION_IO_COMPLETE','head_rows',fixture['rows'],'copy_snapshot_bounds',True,flush=True)
finally:io.close()
