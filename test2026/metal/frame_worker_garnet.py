"""Persistent native image worker. File commands avoid unsupported Python imports."""
import sys,json,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet
from xModel.qwen35.vl_0_8b.metal_vision import MetalImageEncoder
from xModel.qwen35.vl_0_8b.metal_decoder import MetalDecoder
folder=sys.argv[3]
def emit(kind,payload):
    label='ready' if kind=='frame_ready' else str(payload['request'])+'-'+kind
    with open(sys.argv[5]+'.event-'+label+'.json','w') as stream:stream.write(json.dumps(payload))
with open(sys.argv[4]) as stream:workload=json.load(stream)
with open(folder+'/config.json') as stream:config=json.load(stream)
metadata=workload['image_metadata'];batch=workload['batch'];gh,gw=metadata['grid_thw'][1:]
merge=config['vision_config']['spatial_merge_size'];mh,mw=gh//merge,gw//merge
tokenizer=garnet.tokenizer();tokenizer.open(folder)
prefix=tokenizer.encode('<|im_start|>user\n<|vision_start|>')
suffix=tokenizer.encode('<|vision_end|>Describe the main objects in this image in one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
pad=tokenizer.token_id('<|image_pad|>')
prompt=prefix+[pad]*(mh*mw)+suffix
stops=[config['text_config']['eos_token_id'],tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
vision=None;decoder=None;last=-1
try:
    vision=MetalImageEncoder(folder+'/model.safetensors',config['vision_config'],gh,gw,batch)
    decoder=MetalDecoder(folder+'/model.safetensors',config['text_config'],batch,workload['capacity'])
    decoder.enable_prefill(workload['chunk'],workload.get('prefill_projection','shared'),workload.get('prefill_linear_chunk',False))
    admission=None
    if workload.get('check_extended_prefill_admission',False):
        extended='qwen35_extended_prefill' in garnet.backend_info()
        checks=[(80,'mps',True),(40,'shared',False)]
        if not extended:checks.extend([(40,'shared',True),(80,'shared',True)])
        elif batch==2:checks.append((80,'shared',True))
        before=(decoder.chunk,decoder.prefill_projection,decoder.linear_chunk,decoder.prefill_graphs,list(decoder.slots))
        rejected=[]
        for size,projection,linear in checks:
            try:decoder.enable_prefill(size,projection,linear)
            except ValueError:rejected.append([size,projection,linear])
            else:raise AssertionError('Unvalidated extended prefill admitted')
            assert decoder.chunk==before[0] and decoder.prefill_projection==before[1] and decoder.linear_chunk==before[2]
            assert decoder.prefill_graphs is before[3] and decoder.slots==before[4],'Rejected profile changed state'
        admission=dict(extended_build=extended,rejected_profiles=rejected,state_preserved=True)
    emit('frame_ready',dict(prompt_token_ids=prompt,stop_token_ids=stops,visual_tokens=mh*mw,extended_admission=admission))
    while True:
        time.sleep(0.001)
        with open(sys.argv[5]) as stream:message=json.load(stream)
        if message['request']==last:continue
        last=message['request']
        if message.get('stop',False):break
        start=time.monotonic();decoder.reset();reset_seconds=time.monotonic()-start
        embeddings=vision.encode_file(message['patch_file']);vision_seconds=time.monotonic()-start
        hidden=[];positions=[];current=0
        for token in prefix:
            hidden.append(decoder.io.embedding([token]*batch));positions.append([current]*3);current+=1
        image_base=current
        for i in range(mh*mw):
            hidden.append(decoder.io.visual_token(embeddings,i));positions.append([image_base,image_base+i//mw,image_base+i%mw])
        current=image_base+max(mh,mw)
        for token in suffix:
            hidden.append(decoder.io.embedding([token]*batch));positions.append([current]*3);current+=1
        assembled=time.monotonic()
        prediction=decoder.prefill_sequence(hidden,[positions for row in range(batch)])
        fused_head_enabled='fused_head_argmax' in garnet.backend_info()
        fused_head_decode_only_enabled='fused_head_decode_only' in garnet.backend_info()
        deferred_state_zero_enabled='deferred_state_zero' in garnet.backend_info()
        fused_head_counts=[decoder.session.fused_head_argmax_dispatches if fused_head_enabled else 0]
        first=time.monotonic()
        emit('frame_first',dict(request=last,worker_ttft_seconds=first-start))
        output=[[] for row in range(batch)];active=[1]*batch;steps=0
        for token in range(workload['max_tokens']):
            for row in range(batch):
                if active[row]:output[row].append(prediction[row])
                if prediction[row] in stops:active[row]=0
            if not any(active) or token==workload['max_tokens']-1:break
            prediction=decoder.step(prediction,active=active,positions=[[current]*3 for row in range(batch)])
            fused_head_counts.append(decoder.session.fused_head_argmax_dispatches if fused_head_enabled else 0)
            current+=1;steps+=1
        end=time.monotonic()
        decoded=[tokenizer.decode(ids) for ids in output]
        emit('frame_done',dict(request=last))
        emit('frame_result',dict(request=last,prompt_token_ids=prompt,token_ids=output,
            text=decoded,finished=[not flag for flag in active],
            decode_steps_after_first=steps,worker_ttft_seconds=first-start,worker_total_seconds=end-start,
            worker_decode_window_seconds=end-first,vision_seconds=vision_seconds,
            assembly_seconds=assembled-start-vision_seconds,prefill_wall_seconds=first-assembled,
            prefill_profile=decoder.prefill_profile,prefill_inference_ms=decoder.prefill_inference_ms,
            prefill_chunk_plan=decoder.prefill_chunk_plan,
            reset_seconds=reset_seconds,
            prefill_stream_enabled='prefill_stream' in garnet.backend_info(),
            decode_partial_submissions=decoder.session.prefill_partial_submissions if 'prefill_stream' in garnet.backend_info() else 0,
            dirty_prefix_reset_enabled='dirty_prefix_reset' in garnet.backend_info(),
            dirty_prefix_reset_bytes=decoder.session.dirty_prefix_reset_bytes if 'dirty_prefix_reset' in garnet.backend_info() else 0,
            dirty_prefix_reset_states=decoder.session.dirty_prefix_reset_states if 'dirty_prefix_reset' in garnet.backend_info() else 0,
            dirty_prefix_reset_fallback_states=decoder.session.dirty_prefix_reset_fallback_states if 'dirty_prefix_reset' in garnet.backend_info() else 0,
            direct_state_zero_enabled='direct_state_zero' in garnet.backend_info(),
            constant_head_enabled='constant_bf16_head' in garnet.backend_info(),
            residual_norm_enabled='residual_norm' in garnet.backend_info(),
            packed_prefill_norm_enabled='packed_prefill_norm' in garnet.backend_info(),
            global_half_alpha_enabled='global_half_alpha' in garnet.backend_info(),
            joined_global_mlp_enabled='joined_global_mlp' in garnet.backend_info(),
            single_coder_head_enabled='single_coder_head' in garnet.backend_info(),
            fused_head_argmax_enabled=fused_head_enabled,fused_head_argmax_counts=fused_head_counts,
            fused_head_decode_only_enabled=fused_head_decode_only_enabled,
            deferred_state_zero_enabled=deferred_state_zero_enabled,
            decode_deferred_zero_bytes=decoder.session.deferred_zero_bytes if deferred_state_zero_enabled else 0,
            decode_deferred_zero_states=decoder.session.deferred_zero_states if deferred_state_zero_enabled else 0,
            vision_global_alpha_enabled='vision_global_alpha' in garnet.backend_info(),
            vision_global_alpha_dispatches=vision.graph.vision_global_alpha_dispatches if 'vision_global_alpha' in garnet.backend_info() else 0,
            single_coder_head_counts=[decoder.session.single_coder_head_dispatches if 'single_coder_head' in garnet.backend_info() else 0],
            residual_norm_dispatches=decoder.session.residual_norm_dispatches if 'residual_norm' in garnet.backend_info() else 0,
            constant_head_dispatches=decoder.session.constant_head_dispatches if 'constant_bf16_head' in garnet.backend_info() else 0,
            vision_reencoded=True,attention_prefix_enabled='attention_prefix' in garnet.backend_info(),
            weights='BF16 checkpoint; FP32 native vision/activations/KV'))
finally:
    if vision is not None:vision.close()
    if decoder is not None:decoder.close()
    garnet.shutdown()
