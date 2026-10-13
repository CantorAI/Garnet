"""Keep all three real models loaded and run interleaved native GPU requests."""
import sys,json,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet
from xModel.qwen35.vl_0_8b.metal_decoder import MetalDecoder
from xModel.qwen35.vl_0_8b.metal_vision import MetalImageEncoder
from xModel.qwen.metal_dense.decoder import DenseMetalDecoder
root=sys.argv[3];items=[];vision=None;started=time.monotonic()
with open(root+'/images/RealWorld-04.batch1.json') as stream:metadata=json.load(stream)
try:
    for name in ('Qwen3.5-0.8B','Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):
        folder=root+'/models/'+name
        with open(folder+'/config.json') as stream:config=json.load(stream)
        tokenizer=garnet.tokenizer();tokenizer.open(folder)
        if name=='Qwen3.5-0.8B':
            gh,gw=metadata['grid_thw'][1:];merge=config['vision_config']['spatial_merge_size'];mh,mw=gh//merge,gw//merge
            vision=MetalImageEncoder(folder+'/model.safetensors',config['vision_config'],gh,gw,1)
            prefix=tokenizer.encode('<|im_start|>user\n<|vision_start|>')
            suffix=tokenizer.encode('<|vision_end|>Describe the main objects in this image in one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
            c=config['text_config'];limit=32;decoder=MetalDecoder(folder+'/model.safetensors',c,1,len(prefix)+mh*mw+len(suffix)+limit)
            tokens=None
        else:
            prompt='What is the capital of France? Answer briefly.' if name=='Qwen3-0.6B' else 'Write a Python function add(a, b) that returns their sum. Output only code.'
            chat='<|im_start|>user\n'+prompt+'<|im_end|>\n<|im_start|>assistant\n'
            if config['model_type']=='qwen3':chat+='<think>\n\n</think>\n\n'
            tokens=tokenizer.encode(chat);c=config;limit=64
            decoder=DenseMetalDecoder(folder+'/model.safetensors',c,1,len(tokens)+limit)
        decoder.enable_prefill()
        items.append(dict(name=name,decoder=decoder,tokenizer=tokenizer,tokens=tokens,limit=limit,
            stops=[c['eos_token_id'],tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]))
    print('metal_three_resident_initialization',json.dumps(dict(seconds=time.monotonic()-started,models_loaded=len(items))),flush=True)
    for request in range(2):
        for item in items:
            decoder=item['decoder'];decoder.reset();begin=time.monotonic();hidden=[];positions=[]
            if item['tokens'] is None:
                embeddings=vision.encode_file(metadata['patch_file']);current=0
                for token in prefix:hidden.append(decoder.io.embedding([token]));positions.append([current]*3);current+=1
                base=current
                for i in range(mh*mw):hidden.append(decoder.io.visual_token(embeddings,i));positions.append([base,base+i//mw,base+i%mw])
                current=base+max(mh,mw)
                for token in suffix:hidden.append(decoder.io.embedding([token]));positions.append([current]*3);current+=1
            else:
                for i,token in enumerate(item['tokens']):hidden.append(decoder.io.embedding([token]));positions.append([i]*3)
                current=len(hidden)
            ids=decoder.prefill_sequence(hidden,[positions]);ttft=time.monotonic()-begin;output=[];durations=[];finished=False
            for i in range(item['limit']):
                output.append(ids[0])
                if ids[0] in item['stops']:finished=True;break
                if i==item['limit']-1:break
                step=time.monotonic();ids=decoder.step(ids,positions=[[current]*3]);current+=1;durations.append(time.monotonic()-step)
            print('metal_three_resident_result',json.dumps(dict(model=item['name'],timing_clock='time.monotonic',request=request,models_loaded=len(items),
                token_ids=[output],text=[item['tokenizer'].decode(output)],finished=[finished],output_tokens=[len(output)],
                ttft_loaded_seconds=ttft,total_loaded_seconds=time.monotonic()-begin,prefill_chunk=decoder.chunk,
                decode_steps_per_second=len(durations)/sum(durations) if durations else None)),flush=True)
finally:
    for item in items:item['decoder'].close()
    if vision is not None:vision.close()
    garnet.shutdown()
