"""LM Studio native REST cohort with explicit reasoning off and no stored chat.

API schema: https://lmstudio.ai/docs/developer/rest/chat . Preload one model,
one parallel slot, no Draft MTP; use byte-identical common JPEG variants.
"""
import json,time,sys,base64
from pathlib import Path
import requests,lmstudio as lms
dataset,output=sys.argv[1:3]
rounds=int(sys.argv[3]) if len(sys.argv)>3 else 7
model_name=sys.argv[4] if len(sys.argv)>4 else 'Qwen3.5-2B'
identifier=sys.argv[5] if len(sys.argv)>5 else 'qwen35-2b-b1'
assert model_name in ('Qwen3.5-2B','Qwen3.5-4B')
images=['frame_0.jpg','frame_10050.jpg','frame_10200.jpg','frame_10350.jpg'];rows=[]
config={'model':identifier,'reasoning':'off','store':False,'temperature':0,'max_output_tokens':32,
    'top_k':1,'repeat_penalty':1.0}
with lms.Client() as client:
    assert len(client.llm.list_loaded())==1
    model=client.llm.model(identifier)
    Path(output).with_suffix('.config.json').write_text(json.dumps({'load_config':model.get_load_config().to_dict(),
        'request_config':config,'batch':1,'parallel_slots':1,'api_schema':'https://lmstudio.ai/docs/developer/rest/chat',
        'cache_limitation':'Fresh images force image re-evaluation; short text prefix reuse requires runtime-log audit.'},indent=2))
    try:
        for trial in range(rounds):
            for name in images:
                started=time.perf_counter()
                data=base64.b64encode((Path(dataset)/str(trial)/name).read_bytes()).decode('ascii')
                body=dict(config,input=[{'type':'image','data_url':'data:image/jpeg;base64,'+data},
                    {'type':'text','content':'Describe the main objects in this image in one sentence.'}])
                response=requests.post('http://127.0.0.1:1234/api/v1/chat',json=body,timeout=120)
                response.raise_for_status();result=response.json();elapsed=time.perf_counter()-started
                row={'engine':'lmstudio-llama-rest','model':model_name,'batch':1,'trial':trial,'warmup':trial<2,
                    'images':[name],'dataset':dataset,'variant_round':trial,'wall_seconds':elapsed,'result':result,
                    'timing_scope':'JPEG read/base64 plus local REST image upload, preprocess, prefill, decode, detokenization; load excluded'}
                rows.append(row);Path(output).write_text(json.dumps(rows,indent=2))
                print(json.dumps(row),flush=True)
                assert result['stats']['reasoning_output_tokens']==0,'LM Studio ignored reasoning off'
                assert result['stats']['input_tokens']==85,'LM Studio prompt work differs from the85-token Garnet/vLLM request'
        print('QWEN35_LMSTUDIO_REST_BENCHMARK_COMPLETE','requests',len(rows),flush=True)
    finally:model.unload()
