"""Actual Qwen text generation through Garnet's native Metal package.

Run with xlang3, followed by native-package directory, source directory,
model directory, output-token limit, batch size, and optional user prompt.
"""
import sys,json,time
started=time.time()
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet
from xModel.qwen.metal_dense.decoder import DenseMetalDecoder as MetalDecoder

folder=sys.argv[3]
limit=int(sys.argv[4]) if len(sys.argv)>4 else 16
batch=int(sys.argv[5]) if len(sys.argv)>5 else 1
prompt=sys.argv[6] if len(sys.argv)>6 else 'What is the capital of France? Answer briefly.'
chunk=int(sys.argv[7]) if len(sys.argv)>7 else 0
if limit<1 or limit>128:raise ValueError('output-token limit must be 1 through 128')
with open(folder+'/config.json') as stream:config=json.load(stream)
tokenizer=garnet.tokenizer();tokenizer.open(folder)
chat='<|im_start|>user\n'+prompt+'<|im_end|>\n<|im_start|>assistant\n'
if config['model_type']=='qwen3':chat+='<think>\n\n</think>\n\n'
tokens=tokenizer.encode(chat)
stops=[config['eos_token_id'],tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
decoder=None
try:
    decoder=MetalDecoder(folder+'/model.safetensors',config,batch,len(tokens)+limit)
    if chunk:decoder.enable_prefill(chunk)
    initialization=time.time()-started
    print('metal_text_initialization',dict(seconds=initialization,batch=batch,prompt_tokens=len(tokens)),flush=True)
    # A reset makes both requests begin with empty recurrent/KV state.
    # The second run measures a loaded model, without weight loading/compilation.
    for request in range(2):
        decoder.reset();request_start=time.time()
        if chunk:
            hidden_tokens=[decoder.io.embedding([token]*batch) for token in tokens]
            positions=[[[i]*3 for i in range(len(tokens))] for b in range(batch)]
            next_ids=decoder.prefill_sequence(hidden_tokens,positions)
        else:
            for i,token in enumerate(tokens):next_ids=decoder.step([token]*batch,predict=i==len(tokens)-1)
        ttft=time.time()-request_start
        outputs=[[] for b in range(batch)];active=[1 for b in range(batch)];durations=[]
        layer_durations=[];head_durations=[]
        for i in range(limit):
            for b in range(batch):
                if active[b]:outputs[b].append(next_ids[b])
                if next_ids[b] in stops:active[b]=0
            if not any(active) or i==limit-1:break
            step_start=time.time();next_ids=decoder.step(next_ids,active=active)
            durations.append(time.time()-step_start)
            layer_durations.append(decoder.last_layer_ms);head_durations.append(decoder.last_head_ms)
        elapsed=time.time()-request_start
        decoded_steps=len(durations)
        decode_seconds=sum(durations)
        print('metal_text_result',json.dumps(dict(request=request,batch=batch,prompt=prompt,prefill_chunk=chunk,
            prefill_compile_ms=decoder.prefill_compile_ms if chunk else 0,
            text=[tokenizer.decode(ids) for ids in outputs],token_ids=outputs,
            prompt_tokens=len(tokens),output_tokens=[len(ids) for ids in outputs],finished=[a==0 for a in active],
            initialization_seconds=initialization,ttft_loaded_seconds=ttft,total_loaded_seconds=elapsed,
            ttft_first_request_seconds=initialization+ttft if request==0 else None,
            decode_step_seconds=durations,decode_layer_ms=layer_durations,decode_head_ms=head_durations,
            decode_steps_per_second=decoded_steps/decode_seconds if decode_seconds>0 else None)),flush=True)
finally:
    if decoder is not None:decoder.close()
    garnet.shutdown()
