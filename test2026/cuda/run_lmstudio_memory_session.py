"""Controlled LM Studio memory holds; timing is diagnostic only.

Preload exactly one model, identify its GPU worker PID, then start this script
and the separate adapter-resolved WDDM collector. Loading is not measured here.
"""
import base64,json,os,sys,time
from pathlib import Path
import lmstudio as lms
import requests
from concurrent.futures import ThreadPoolExecutor
from qwen35_benchmark_workload import batch_inputs,input_hashes
dataset,output,identifier,name,worker=sys.argv[1:6];worker=int(worker)
assert name in ('Qwen3.5-2B','Qwen3.5-4B')
batch=int(sys.argv[6]) if len(sys.argv)>6 else 1
matched=len(sys.argv)>7 and sys.argv[7]=='matched'
assert batch in (1,2,4,8) and (matched or batch==1)
events=[];profile={}
def emit(event,**values):
    record=dict(event=event,unix_seconds=time.time(),process_id=worker,frontend_process_id=os.getpid(),
        engine='lmstudio-llama-cuda',model=name,profile=dict(profile),**values)
    events.append(record);Path(output).write_text(json.dumps(events,indent=2))
    print(json.dumps({'event':event,'unix_seconds':record['unix_seconds'],'memory_process_id':worker}),flush=True)
config={'model':identifier,'reasoning':'off','store':False,'temperature':0,'max_output_tokens':32,
    'top_k':1,'repeat_penalty':1.0}
images=['frame_0.jpg','frame_10050.jpg','frame_10200.jpg','frame_10350.jpg']
def infer(trial):
    if matched:names,paths=batch_inputs(dataset,trial//4,trial%4,batch,True)
    else:
        names=[images[trial%4]];paths=[Path(dataset)/str(trial)/names[0]]
    def invoke(path):
        data=base64.b64encode(Path(path).read_bytes()).decode('ascii')
        body=dict(config,input=[{'type':'image','data_url':'data:image/jpeg;base64,'+data},
            {'type':'text','content':'Describe the main objects in this image in one sentence.'}])
        response=requests.post('http://127.0.0.1:1234/api/v1/chat',json=body,timeout=300)
        response.raise_for_status();result=response.json()
        assert result['stats']['input_tokens']==85 and result['stats']['reasoning_output_tokens']==0
        return result
    with ThreadPoolExecutor(max_workers=batch) as pool:records=list(pool.map(invoke,paths))
    return {'records':records,'images':names,'input_sha256':input_hashes(paths),'batch':batch} if matched else records[0]
emit('load_begin',loading_observed=False,scope='Attach already-loaded model; memory diagnostic, not latency evidence')
with lms.Client() as client:
    assert len(client.llm.list_loaded())==1
    model=client.llm.model(identifier);profile.update(model.get_load_config().to_dict())
    profile.update({'requested_batch':batch,'matched_batch_inputs':matched})
    emit('loaded',request_config=config)
    try:
        for trial in range(2):emit('warmup_complete',trial=trial,result=infer(trial))
        emit('steady_before_begin');time.sleep(8);emit('steady_before_end')
        emit('inference_begin')
        for trial in range(2,6):emit('request_complete',trial=trial,result=infer(trial))
        emit('inference_end')
        emit('steady_after_begin');time.sleep(8);emit('steady_after_end')
    finally:
        model.unload();emit('closed_begin');time.sleep(8);emit('closed_end')
emit('complete')
print('QWEN35_LMSTUDIO_MEMORY_SESSION_COMPLETE',name,flush=True)
