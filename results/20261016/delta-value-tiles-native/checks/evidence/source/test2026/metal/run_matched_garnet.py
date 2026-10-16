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
    decoder.enable_prefill(workload.get('chunk',16),workload.get('prefill_projection','shared'),finer_chunks=workload.get('prefill_finer_chunks',False),exact_tail=workload.get('prefill_exact_tail',False))
    if workload.get('check_exact_tail_admission',False):
        before=(decoder.chunk,decoder.prefill_projection,decoder.prefill_graphs,list(decoder.slots),decoder.finer_chunks,decoder.exact_tail)
        invalid=[(22,'shared',True,False),(26,'shared',True,False),(32,'shared',False,True),(32,'mps',True,True)]
        for size,projection,finer,exact in invalid:
            rejected=False
            try:decoder.enable_prefill(size,projection,finer_chunks=finer,exact_tail=exact)
            except ValueError:rejected=True
            assert rejected,'Unvalidated exact-tail profile admitted'
            assert before==(decoder.chunk,decoder.prefill_projection,decoder.prefill_graphs,decoder.slots,decoder.finer_chunks,decoder.exact_tail),'Rejected exact-tail profile changed state'
        print('exact_tail_admission PASS',flush=True)
    single_coder_head_enabled='single_coder_head' in garnet.backend_info()
    fused_head_argmax_enabled='fused_head_argmax' in garnet.backend_info()
    screened_head_enabled='screened_head' in garnet.backend_info()
    small_screen_seed_enabled='screen_seed_1024' in garnet.backend_info()
    interleaved_screen_enabled='interleaved_screen_index' in garnet.backend_info()
    interleaved_body_enabled='interleaved_bf16_body' in garnet.backend_info()
    coalesced_block_enabled='coalesced_bf16_body' in garnet.backend_info();mlp_graph_enabled='mlp_prefill_graph' in garnet.backend_info();delta_tile64_enabled='delta_value_tile64' in garnet.backend_info()
    fused_head_decode_only_enabled='fused_head_decode_only' in garnet.backend_info()
    deferred_state_zero_enabled='deferred_state_zero' in garnet.backend_info()
    reused_zero_source_enabled='reused_zero_source' in garnet.backend_info()
    body_shape_enabled='body_shape' in garnet.backend_info()
    initialization=time.monotonic()-started
    for request in range(repeats):
        begin=time.monotonic();decoder.reset();reset_seconds=time.monotonic()-begin
        hidden=[decoder.io.embedding([token]*batch) for token in ids]
        positions=[[[i]*3 for i in range(len(ids))] for row in range(batch)]
        prediction=decoder.prefill_sequence(hidden,positions);ttft=time.monotonic()-begin
        single_head_counts=[decoder.session.single_coder_head_dispatches if single_coder_head_enabled else 0]
        fused_head_counts=[decoder.session.fused_head_argmax_dispatches if fused_head_argmax_enabled else 0]
        screened_head_counts=[decoder.session.screened_head_dispatches if screened_head_enabled else 0]
        screened_head_seed_rows=[decoder.session.screened_head_seed_rows if small_screen_seed_enabled else (8192 if screened_head_enabled else 0)]
        interleaved_screen_counts=[decoder.session.interleaved_screen_dispatches if interleaved_screen_enabled else 0]
        interleaved_body_counts=[decoder.session.interleaved_body_dispatches if interleaved_body_enabled else 0]
        coalesced_block_counts=[decoder.session.coalesced_block_dispatches if coalesced_block_enabled else 0];mlp_graph_counts=[decoder.prefill_profile.get('mlp_prefill_graph_dispatches',0) if mlp_graph_enabled else 0];delta_tile64_counts=[decoder.prefill_profile.get('delta_value_tile64_dispatches',0) if delta_tile64_enabled else 0]
        body_shape_counts=[decoder.session.body_shape_dispatches if body_shape_enabled else 0]
        body_shape_misses=[decoder.session.body_shape_misses if body_shape_enabled else 0]
        output=[[] for row in range(batch)];active=[1]*batch;durations=[];native_ms=[];phases=[]
        for token in range(limit):
            for row in range(batch):
                if active[row]:output[row].append(prediction[row])
                if prediction[row] in stops:active[row]=0
            if not any(active) or token==limit-1:break
            step=time.monotonic();prediction=decoder.step(prediction,active=active)
            durations.append(time.monotonic()-step);native_ms.append(decoder.last_session_ms)
            single_head_counts.append(decoder.session.single_coder_head_dispatches if single_coder_head_enabled else 0)
            fused_head_counts.append(decoder.session.fused_head_argmax_dispatches if fused_head_argmax_enabled else 0)
            screened_head_counts.append(decoder.session.screened_head_dispatches if screened_head_enabled else 0)
            screened_head_seed_rows.append(decoder.session.screened_head_seed_rows if small_screen_seed_enabled else (8192 if screened_head_enabled else 0))
            interleaved_screen_counts.append(decoder.session.interleaved_screen_dispatches if interleaved_screen_enabled else 0)
            interleaved_body_counts.append(decoder.session.interleaved_body_dispatches if interleaved_body_enabled else 0)
            coalesced_block_counts.append(decoder.session.coalesced_block_dispatches if coalesced_block_enabled else 0);mlp_graph_counts.append(decoder.session.mlp_prefill_graph_dispatches if mlp_graph_enabled else 0);delta_tile64_counts.append(decoder.session.delta_value_tile64_dispatches if delta_tile64_enabled else 0)
            body_shape_counts.append(decoder.session.body_shape_dispatches if body_shape_enabled else 0)
            body_shape_misses.append(decoder.session.body_shape_misses if body_shape_enabled else 0)
            phases.append(dict(prepare_ms=decoder.session.prepare_ms,encode_ms=decoder.session.encode_ms,
                gpu_ms=decoder.session.gpu_ms,wait_ms=decoder.session.wait_ms))
        total=time.monotonic()-begin
        decode_window=total-ttft
        assert all(d>0 for d in durations)
        assert all(d*1000+0.1>=n for d,n in zip(durations,native_ms))
        result=dict(runtime='garnet',model=folder,request=request,
            clock='time.monotonic',initialization_seconds=initialization,ttft_seconds=ttft,total_seconds=total,
            token_ids=output,text=[tokenizer.decode(o) for o in output],finished=[not a for a in active],
            reset_seconds=reset_seconds,
            prefill_stream_enabled='prefill_stream' in garnet.backend_info(),
            decode_partial_submissions=decoder.session.prefill_partial_submissions if 'prefill_stream' in garnet.backend_info() else 0,
            dirty_prefix_reset_enabled='dirty_prefix_reset' in garnet.backend_info(),
            dirty_prefix_reset_bytes=decoder.session.dirty_prefix_reset_bytes if 'dirty_prefix_reset' in garnet.backend_info() else 0,
            dirty_prefix_reset_states=decoder.session.dirty_prefix_reset_states if 'dirty_prefix_reset' in garnet.backend_info() else 0,
            dirty_prefix_reset_fallback_states=decoder.session.dirty_prefix_reset_fallback_states if 'dirty_prefix_reset' in garnet.backend_info() else 0,
            prompt_token_ids=ids,decode_step_seconds=durations,decode_steps_after_first=len(durations),
            decode_steps_per_second=len(durations)/sum(durations) if durations else None,
            decode_window_seconds=decode_window,
            decode_wall_steps_per_second=len(durations)/decode_window if durations else None,
            native_decode_ms=native_ms,decode_phases=phases,prefill_profile=decoder.prefill_profile,
            prefill_chunk_plan=decoder.prefill_chunk_plan,prefill_inference_ms=decoder.prefill_inference_ms,
            attention_prefix_enabled='attention_prefix' in garnet.backend_info(),
            direct_state_zero_enabled='direct_state_zero' in garnet.backend_info(),
            constant_head_enabled='constant_bf16_head' in garnet.backend_info(),
            residual_norm_enabled='residual_norm' in garnet.backend_info(),
            packed_prefill_norm_enabled='packed_prefill_norm' in garnet.backend_info(),
            global_half_alpha_enabled='global_half_alpha' in garnet.backend_info(),
            joined_global_mlp_enabled='joined_global_mlp' in garnet.backend_info(),
            single_coder_head_enabled=single_coder_head_enabled,single_coder_head_counts=single_head_counts,
            fused_head_argmax_enabled=fused_head_argmax_enabled,fused_head_argmax_counts=fused_head_counts,
            screened_head_enabled=screened_head_enabled,screened_head_counts=screened_head_counts,
            small_screen_seed_enabled=small_screen_seed_enabled,screened_head_seed_rows=screened_head_seed_rows,
            interleaved_screen_enabled=interleaved_screen_enabled,interleaved_screen_counts=interleaved_screen_counts,
            interleaved_body_enabled=interleaved_body_enabled,interleaved_body_counts=interleaved_body_counts,
            coalesced_block_enabled=coalesced_block_enabled,coalesced_block_counts=coalesced_block_counts,
            mlp_prefill_graph_enabled=mlp_graph_enabled,mlp_prefill_graph_counts=mlp_graph_counts,
            delta_value_tile64_enabled=delta_tile64_enabled,delta_value_tile64_counts=delta_tile64_counts,
            fused_head_decode_only_enabled=fused_head_decode_only_enabled,
            body_shape_enabled=body_shape_enabled,body_shape_counts=body_shape_counts,body_shape_misses=body_shape_misses,
            body_shape_pipeline_count=decoder.session.body_shape_pipeline_count if body_shape_enabled else 0,
            deferred_state_zero_enabled=deferred_state_zero_enabled,
            reused_zero_source_enabled=reused_zero_source_enabled,
            reused_zero_source_bytes=decoder.session.reused_zero_source_bytes if reused_zero_source_enabled else 0,
            reused_zero_copy_bytes=decoder.session.reused_zero_copy_bytes if reused_zero_source_enabled else 0,
            reused_zero_copy_states=decoder.session.reused_zero_copy_states if reused_zero_source_enabled else 0,
            decode_deferred_zero_bytes=decoder.session.deferred_zero_bytes if deferred_state_zero_enabled else 0,
            decode_deferred_zero_states=decoder.session.deferred_zero_states if deferred_state_zero_enabled else 0,
            vision_global_alpha_enabled='vision_global_alpha' in garnet.backend_info(),vision_global_alpha_dispatches=0,
            vector_vision_context_enabled='vector_vision_context4' in garnet.backend_info(),vision_vector_dispatches=0,
            residual_norm_dispatches=decoder.session.residual_norm_dispatches if 'residual_norm' in garnet.backend_info() else 0,
            constant_head_dispatches=decoder.session.constant_head_dispatches if 'constant_bf16_head' in garnet.backend_info() else 0,
            weights='checkpoint BF16 decode projections/head; exact FP32 expansion for MPS prefill if selected; FP32 accumulation/activations/KV')
        result['case']=workload
        print('matched_result',json.dumps(result),flush=True)
finally:
    if decoder is not None:decoder.close()
    garnet.shutdown()
