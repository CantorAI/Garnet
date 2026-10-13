"""Actual image-plus-text requests with persistent Garnet Metal weights.

Arguments after xlang3: native package, source, model folder, patch metadata,
and optional output-token limit. Image preparation contains no learned ops.
"""
import sys,json,time
start=time.monotonic()
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet
from xModel.qwen35.vl_0_8b.metal_vision import MetalImageEncoder
from xModel.qwen35.vl_0_8b.metal_decoder import MetalDecoder
folder=sys.argv[3];limit=int(sys.argv[5]) if len(sys.argv)>5 else 16
if limit<1 or limit>128:raise ValueError('output-token limit must be 1 through 128')
with open(folder+'/config.json') as stream:config=json.load(stream)
with open(sys.argv[4]) as stream:metadata=json.load(stream)
batch=metadata['shape'][0];gh,gw=metadata['grid_thw'][1:]
merge=config['vision_config']['spatial_merge_size'];mh,mw=gh//merge,gw//merge
tokenizer=garnet.tokenizer();tokenizer.open(folder)
prefix=tokenizer.encode('<|im_start|>user\n<|vision_start|>')
suffix=tokenizer.encode('<|vision_end|>Describe the main objects in this image in one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
stops=[config['text_config']['eos_token_id'],tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
capacity=len(prefix)+mh*mw+len(suffix)+limit
chunk=int(sys.argv[6]) if len(sys.argv)>6 else 0
optimization=int(sys.argv[7]) if len(sys.argv)>7 else 3
if optimization not in (0,1,2,3):raise ValueError('optimization mode must be 0, 1, 2 or 3')
resident=optimization>=1;bf16_head=optimization>=2
vision=None;decoder=None
try:
    vision=MetalImageEncoder(folder+'/model.safetensors',config['vision_config'],gh,gw,batch)
    decoder=MetalDecoder(folder+'/model.safetensors',config['text_config'],batch,capacity,bf16_head=bf16_head,resident=False,bf16_projections=optimization>=3)
    if chunk:decoder.enable_prefill(chunk)
    if resident:decoder.enable_resident()
    initialization=time.monotonic()-start
    print('metal_image_initialization',dict(seconds=initialization,vision_compile_ms=vision.graph.compile_ms,
        batch=batch,capacity=capacity,resized_hw=metadata['resized_hw'],visual_tokens=mh*mw),flush=True)
    for request in range(2):
        decoder.reset();request_start=time.monotonic()
        # Re-encode each frame with resident weights; do not reuse embeddings.
        image_embeddings=vision.encode_file(metadata['patch_file'])
        vision_seconds=time.monotonic()-request_start;current=0;prefill=time.monotonic()
        hidden_tokens=[];prompt_positions=[]
        for token in prefix:
            if chunk:hidden_tokens.append(decoder.io.embedding([token]*batch));prompt_positions.append([current]*3)
            else:decoder.step([token]*batch,positions=[[current]*3 for b in range(batch)],predict=False)
            current+=1
        image_base=current
        for i in range(mh*mw):
            position=[image_base,image_base+i//mw,image_base+i%mw]
            if chunk:hidden_tokens.append(decoder.io.visual_token(image_embeddings,i));prompt_positions.append(position)
            else:decoder.step_hidden(decoder.io.visual_token(image_embeddings,i),positions=[position for b in range(batch)],predict=False)
        current=image_base+max(mh,mw)
        for i,token in enumerate(suffix):
            if chunk:hidden_tokens.append(decoder.io.embedding([token]*batch));prompt_positions.append([current]*3)
            else:next_ids=decoder.step([token]*batch,positions=[[current]*3 for b in range(batch)],predict=i==len(suffix)-1)
            current+=1
        if chunk:next_ids=decoder.prefill_sequence(hidden_tokens,[prompt_positions for b in range(batch)])
        prefill_seconds=time.monotonic()-prefill;ttft=time.monotonic()-request_start
        outputs=[[] for b in range(batch)];active=[1]*batch;durations=[]
        layer_durations=[];head_durations=[];session_durations=[];phase_durations=[]
        for i in range(limit):
            for b in range(batch):
                if active[b]:outputs[b].append(next_ids[b])
                if next_ids[b] in stops:active[b]=0
            if not any(active) or i==limit-1:break
            step=time.monotonic();next_ids=decoder.step(next_ids,active=active,positions=[[current]*3 for b in range(batch)])
            current+=1;durations.append(time.monotonic()-step)
            layer_durations.append(decoder.last_layer_ms);head_durations.append(None if resident else decoder.last_head_ms);session_durations.append(decoder.last_session_ms if resident else None)
            if resident:phase_durations.append(dict(prepare_ms=decoder.session.prepare_ms,encode_ms=decoder.session.encode_ms,gpu_ms=decoder.session.gpu_ms,wait_ms=decoder.session.wait_ms))
        if any(d<=0 for d in durations):raise ValueError('Nonpositive decode duration')
        if resident and any(d*1000+0.1<m for d,m in zip(durations,session_durations)):raise ValueError('External timer shorter than native steady-clock inference')
        decode_seconds=sum(durations)
        print('metal_image_result',json.dumps(dict(timing_clock='time.monotonic',request=request,batch=batch,prefill_chunk=chunk,
            prefill_compile_ms=decoder.prefill_compile_ms if chunk else 0,
            resident_state=resident,bf16_head=bf16_head,bf16_projections=optimization>=3,
            prefill_profile=decoder.prefill_profile if resident and chunk else None,decode_phase_ms=phase_durations,session_submissions_per_decode_step=1 if resident else None,
            text=[tokenizer.decode(ids) for ids in outputs],token_ids=outputs,finished=[a==0 for a in active],
            output_tokens=[len(ids) for ids in outputs],image_sha256=metadata['image_sha256'],resized_hw=metadata['resized_hw'],
            visual_tokens=mh*mw,prompt_tokens=len(prefix)+mh*mw+len(suffix),initialization_seconds=initialization,
            vision_seconds=vision_seconds,vision_inference_ms=vision.graph.last_inference_ms,prefill_seconds=prefill_seconds,
            ttft_loaded_seconds=ttft,total_loaded_seconds=time.monotonic()-request_start,decode_step_seconds=durations,
            ttft_first_request_seconds=initialization+ttft if request==0 else None,
            decode_layer_ms=layer_durations,decode_head_ms=head_durations,decode_session_ms=session_durations,
            decode_steps_per_second=len(durations)/decode_seconds if decode_seconds>0 else None)),flush=True)
    print('metal_graph_operations',json.dumps(dict(decode=[layer['graph'].operation_count for layer in decoder.layers],prefill=[graph.operation_count for graph in decoder.prefill_graphs.values()] if hasattr(decoder,'prefill_graphs') else [])),flush=True)
finally:
    if vision is not None:vision.close()
    if decoder is not None:decoder.close()
    garnet.shutdown()
