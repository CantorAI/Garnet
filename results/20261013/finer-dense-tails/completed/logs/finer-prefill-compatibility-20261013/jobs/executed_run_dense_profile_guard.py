"""Native matched text/coder requests, reporting first prediction and decode steps.

Arguments after xlang3: build, source, model, case JSON.
"""
import sys,json,time
started=time.monotonic()
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet
from xModel.qwen.metal_dense.decoder import DenseMetalDecoder
folder=sys.argv[3]
with open(sys.argv[4]) as stream:workload=json.load(stream)
with open(folder+'/config.json') as stream:config=json.load(stream)
tokenizer=garnet.tokenizer();tokenizer.open(folder)
chat='<|im_start|>user\n'+workload['prompt']+'<|im_end|>\n<|im_start|>assistant\n'
if config['model_type']=='qwen3':chat+='<think>\n\n</think>\n\n'
ids=tokenizer.encode(chat)
stops=[config['eos_token_id'],tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
print('matched_prompt',json.dumps(dict(chat=chat,token_ids=ids,stop_token_ids=stops)),flush=True)
decoder=None;batch=workload['batch'];limit=workload['max_tokens'];repeats=workload.get('requests',3)
try:
    decoder=DenseMetalDecoder(folder+'/model.safetensors',config,batch,workload['capacity'])
    decoder.enable_prefill(workload.get('chunk',16),workload.get('prefill_projection','shared'))
    for unsupported in (24,32,48,64):
        before=decoder.chunk;before_projection=decoder.prefill_projection;before_graphs=decoder.prefill_graphs;before_slots=list(decoder.slots);rejected=False
        try:decoder.enable_prefill(unsupported)
        except ValueError:rejected=True
        assert rejected,'Unvalidated dense profile accepted'
        assert decoder.chunk==before and decoder.prefill_projection==before_projection and decoder.prefill_graphs is before_graphs and decoder.slots==before_slots,'Rejected profile changed decoder state'
    before=decoder.chunk;before_projection=decoder.prefill_projection;before_graphs=decoder.prefill_graphs;before_slots=list(decoder.slots);rejected=False
    try:decoder.enable_prefill(16,finer_chunks=True)
    except ValueError:rejected=True
    assert rejected,'Legacy backend accepted finer scheduling'
    assert decoder.chunk==before and decoder.prefill_projection==before_projection and decoder.prefill_graphs is before_graphs and decoder.slots==before_slots,'Rejected scheduling changed decoder state'
    print('dense_profile_guard PASS',flush=True)
    initialization=time.monotonic()-started
    for request in range(repeats):
        begin=time.monotonic();decoder.reset()
        hidden=[decoder.io.embedding([token]*batch) for token in ids]
        positions=[[[i]*3 for i in range(len(ids))] for row in range(batch)]
        prediction=decoder.prefill_sequence(hidden,positions);ttft=time.monotonic()-begin
        output=[[] for row in range(batch)];active=[1]*batch;durations=[];native_ms=[];phases=[]
        for token in range(limit):
            for row in range(batch):
                if active[row]:output[row].append(prediction[row])
                if prediction[row] in stops:active[row]=0
            if not any(active) or token==limit-1:break
            step=time.monotonic();prediction=decoder.step(prediction,active=active)
            durations.append(time.monotonic()-step);native_ms.append(decoder.last_session_ms)
            phases.append(dict(prepare_ms=decoder.session.prepare_ms,encode_ms=decoder.session.encode_ms,
                gpu_ms=decoder.session.gpu_ms,wait_ms=decoder.session.wait_ms))
        total=time.monotonic()-begin
        decode_window=total-ttft
        assert all(d>0 for d in durations)
        assert all(d*1000+0.1>=n for d,n in zip(durations,native_ms))
        result=dict(runtime='garnet',model=folder,request=request,
            clock='time.monotonic',initialization_seconds=initialization,ttft_seconds=ttft,total_seconds=total,
            token_ids=output,text=[tokenizer.decode(o) for o in output],finished=[not a for a in active],
            prompt_token_ids=ids,decode_step_seconds=durations,decode_steps_after_first=len(durations),
            decode_steps_per_second=len(durations)/sum(durations) if durations else None,
            decode_window_seconds=decode_window,
            decode_wall_steps_per_second=len(durations)/decode_window if durations else None,
            native_decode_ms=native_ms,decode_phases=phases,prefill_profile=decoder.prefill_profile,
            weights='checkpoint BF16 decode projections/head; exact FP32 expansion for MPS prefill if selected; FP32 accumulation/activations/KV')
        result['case']=workload
        print('matched_result',json.dumps(result),flush=True)
finally:
    if decoder is not None:decoder.close()
    garnet.shutdown()
