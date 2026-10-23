"""One LM Studio model; synchronized request groups, explicit batch latency.

Preload --parallel B with actual context512, no speculative draft. Runtime logs
must still establish slot/cache execution; overlapping HTTP calls alone do not.
"""
import argparse,base64,json,threading,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests,lmstudio as lms
from qwen35_benchmark_workload import batch_inputs,input_hashes
p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('output');p.add_argument('--identifier',required=True)
p.add_argument('--model',choices=['Qwen3.5-2B','Qwen3.5-4B'],required=True)
p.add_argument('--batch',type=int,choices=[1,2,4,8],required=True);p.add_argument('--rounds',type=int,default=7)
p.add_argument('--load-audit',help='Required at batch>1: actual per-sequence context and slot evidence from pre-load runtime capture')
a=p.parse_args();output=Path(a.output);assert not output.exists(),'Preserve prior evidence'
config={'model':a.identifier,'reasoning':'off','store':False,'temperature':0,'max_output_tokens':32,'top_k':1,'repeat_penalty':1.0}
rows=[];error_lock=threading.Lock()
with lms.Client() as client:
    assert len(client.llm.list_loaded())==1
    model=client.llm.model(a.identifier);load=model.get_load_config().to_dict()
    audit=json.loads(Path(a.load_audit).read_text()) if a.load_audit else None
    if audit is not None:
        assert audit['identifier']==a.identifier and audit['requested_batch']==a.batch
        assert audit['context_per_sequence_verified'] and audit['configured_parallel_slots_verified'],audit
        assert audit.get('independent_slot_capacity_verified') or a.batch==1,'Shared unified pool is not matched per-row capacity'
    else:
        assert a.batch==1,'Batch>1 requires actual runtime slot/context audit before timing'
        assert load['contextLength']==512,load
    output.with_suffix('.config.json').write_text(json.dumps({'load_config':load,'request_config':config,
        'requested_batch':a.batch,'parallel_slots_verified':bool(audit),'actual_load_audit':audit,
        'limitation':'Must audit actual runtime parallel slots and image/text cache logs; client concurrency is not proof of GPU batching.'},indent=2))
    try:
        with ThreadPoolExecutor(max_workers=a.batch) as pool:
            for trial in range(a.rounds):
                for group in range(4):
                    names,paths=batch_inputs(a.dataset,trial,group,a.batch,True);hashes=input_hashes(paths)
                    barrier=threading.Barrier(a.batch+1)
                    def invoke(row):
                        barrier.wait(timeout=30);began=time.perf_counter();unix_begin=time.time()
                        data=base64.b64encode(Path(paths[row]).read_bytes()).decode('ascii')
                        body=dict(config,input=[{'type':'image','data_url':'data:image/jpeg;base64,'+data},
                            {'type':'text','content':'Describe the main objects in this image in one sentence.'}])
                        response=requests.post('http://127.0.0.1:1234/api/v1/chat',json=body,timeout=300)
                        if not response.ok:
                            with error_lock:
                                with output.with_suffix('.errors.jsonl').open('a') as stream:
                                    stream.write(json.dumps({'trial':trial,'group':group,'row':row,
                                        'http_status':response.status_code,'body':response.text,'input_sha256':hashes[row]})+'\n')
                        response.raise_for_status();result=response.json()
                        assert result['stats']['reasoning_output_tokens']==0
                        assert result['stats']['input_tokens']==85
                        return {'row':row,'result':result,'wall_seconds':time.perf_counter()-began,
                            'begin_unix_seconds':unix_begin,'end_unix_seconds':time.time()}
                    futures=[pool.submit(invoke,row) for row in range(a.batch)]
                    started=time.perf_counter();barrier.wait(timeout=30)
                    records=[f.result() for f in futures];elapsed=time.perf_counter()-started
                    rows.append({'engine':'lmstudio-llama-rest','model':a.model,'batch':a.batch,'trial':trial,
                        'warmup':trial<2,'images':names,'workload_group':group,'input_sha256':hashes,
                        'matched_batch_inputs':True,'dataset':a.dataset,'wall_seconds':elapsed,'records':records,
                        'timing_scope':'Group completion latency; each row records HTTP latency including JPEG read/base64. No throughput-to-latency substitution.'})
                    output.write_text(json.dumps(rows,indent=2))
                    print('LM_BATCH_GROUP_COMPLETE',trial,group,a.batch,elapsed,flush=True)
        print('QWEN35_LM_BATCH_COMPLETE',len(rows),flush=True)
    finally:model.unload()
