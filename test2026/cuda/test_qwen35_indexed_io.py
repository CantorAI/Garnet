"""Indexed proposal API: CPU oracle, cache separation, bounds and lifetime."""
import json,os,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
import garnet
fixture=json.load(open(sys.argv[1]));io=garnet.cuda_tensor_io();io.open(fixture['checkpoint'])
def cpu(t):return garnet.tensor_to_cpu(t).tolist()
def memory():return json.loads(io.memory_json())
try:
    assert memory()['candidate_workspace_bytes']==memory()['candidate_index_bytes']==memory()['indexed_head_buffers_bytes']==0
    width=fixture['width'];cube=io.from_host(fixture['values'],[8,16,width]);io.predict_tokens(cube)
    records=[]
    for case in fixture['cases']:
        rows=case['rows'];query=io.from_host(fixture['values'][:rows*width],[rows,width])
        # Overwrite ordinary-head cache; token-head shortlist must remain unchanged.
        io.predict(query)
        indices=io.head_candidates(case['selected'],case['count'],True)
        assert cpu(indices)==case['indices'],('CPU stable sort',case)
        for repeat in range(3):assert list(io.predict_indexed(query,indices))==case['expected'],('Indexed head',case)
        before=memory();saved=cpu(indices)
        for selected,count,token in [([-1],16,True),([128],16,True),([0],65,True),([0],0,True),([],16,True)]:
            rejected=False
            try:io.head_candidates(selected,count,token)
            except BaseException:rejected=True
            assert rejected and cpu(indices)==saved and memory()['candidate_index_bytes']==before['candidate_index_bytes']
        bad=io.zeros([rows+1,width]);rejected=False
        try:io.predict_indexed(bad,indices)
        except BaseException:rejected=True
        assert rejected and memory()['indexed_head_buffers_bytes']==before['indexed_head_buffers_bytes']
        records.append({'rows':rows,'count':case['count'],'cpu_oracle':True})
    after=memory();assert after['embedding_head_shared_bytes']==fixture['vocab']*width*2
    io.close();closed=memory()
    assert all(closed[k]==0 for k in ('embedding_head_shared_bytes','head_input_bytes','head_output_bytes','candidate_workspace_bytes','candidate_index_bytes','indexed_head_buffers_bytes'))
    rejected=False
    try:io.predict_indexed(query,indices)
    except BaseException:rejected=True
    assert rejected
    print('QWEN35_INDEXED_IO_COMPLETE',json.dumps({'cases':records,'owned_io':after,'closed_fields_zero':True}),flush=True)
finally:io.close()
