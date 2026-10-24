"""Exploratory synchronized native-call timing; deliberately not HTTP comparison."""
import json,os,time

def run_native_text_pilot(decoder,tokenizer,request_path,output):
    with open(request_path) as f:inputs=json.load(f)
    reports=[]
    for index,item in enumerate([inputs['requests'][0]]+inputs['requests']):
        submitted=time.monotonic();decoder.reset();body=item['body']
        assert body['temperature']==0 and body['max_tokens']==32 and isinstance(body['messages'][0]['content'],str)
        prompt='<|im_start|>user\n'+body['messages'][0]['content']+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        ids=tokenizer.encode(prompt);assert len(ids)==41 and len(ids)+32<=4096
        for i,token in enumerate(ids):prediction=decoder.step([token],predict=i+1==len(ids))
        events=[{'token_id':prediction[0],'monotonic_seconds':time.monotonic()}]
        stops=[tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
        while len(events)<32 and events[-1]['token_id'] not in stops:
            prediction=decoder.step([events[-1]['token_id']])
            events.append({'token_id':prediction[0],'monotonic_seconds':time.monotonic()})
        completed=time.monotonic();generated=[e['token_id'] for e in events]
        reports.append({'key':item['key'],'measured':index>0,'request_sha256':item['request_sha256'],
            'submitted':submitted,'first_token':events[0]['monotonic_seconds'],'last_token':events[-1]['monotonic_seconds'],
            'completed':completed,'generated_tokens':len(events),'prompt_tokens':len(ids),'status':'completed',
            'token_events':events,'generated_ids':generated,'text':tokenizer.decode([t for t in generated if t not in stops]),
            'finish_reason':'stop' if generated[-1] in stops else 'length'})
        result={'run_id':os.environ.get('GARNET_STUDY_RUN_ID'),'rows':reports,'status':'completed' if index==len(inputs['requests']) else 'running','context_capacity':4096,'batch':1,'concurrency':1,'warmups':1,
            'source_request_manifest':inputs,'io_memory_snapshot':json.loads(decoder.io.memory_json()),
            'scope':'Exploratory synchronized direct native calls; includes tokenization/reset/prefill/sampling, excludes HTTP; not a matched TTFT/throughput or2x claim'}
        with open(output,'w') as f:json.dump(result,f,indent=2)
        print('NATIVE_TEXT_PILOT_REQUEST',index,item['key'],len(events),flush=True)

