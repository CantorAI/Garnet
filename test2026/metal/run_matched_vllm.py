"""Same native prompt IDs and greedy requests through the vLLM-Metal engine.

Arguments: model, case JSON, native Garnet output log (for prompt identity).
Direct engine timing avoids comparing a web server with a native library.
"""
import json,os,sys,time,resource
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['VLLM_MLX_DEVICE']='gpu'
def main():
    started=time.monotonic()
    from vllm import LLM,SamplingParams
    model=sys.argv[1];case=json.loads(Path(sys.argv[2]).read_text())
    dtype=case.get('comparator_dtype','bfloat16')
    if dtype not in ('bfloat16','float32'):raise ValueError('Unsupported comparator dtype')
    memory_fraction=case.get('comparator_gpu_memory_utilization',.30)
    if not .1<=memory_fraction<=.6:raise ValueError('Unsupported comparator memory fraction')
    prompt=next(json.loads(line.split(' ',1)[1]) for line in Path(sys.argv[3]).read_text().splitlines() if line.startswith('matched_prompt '))
    llm=LLM(model=model,dtype=dtype,max_model_len=case['capacity'],max_num_seqs=case['batch'],
        max_num_batched_tokens=case['capacity'],enable_prefix_caching=False,
        gpu_memory_utilization=memory_fraction,disable_log_stats=True,seed=0,
        distributed_executor_backend='uni')
    initialization=time.monotonic()-started
    params=SamplingParams(temperature=0,top_p=1,repetition_penalty=1,max_tokens=case['max_tokens'],
        seed=0,stop_token_ids=prompt['stop_token_ids'])
    engine=llm.llm_engine
    assert not engine.vllm_config.cache_config.enable_prefix_caching
    assert engine.vllm_config.speculative_config is None
    print('matched_engine_config',json.dumps(dict(model_dtype=str(llm.llm_engine.model_config.dtype),
        prefix_cache_enabled=llm.llm_engine.vllm_config.cache_config.enable_prefix_caching,
        speculative_config=str(llm.llm_engine.vllm_config.speculative_config),
        scheduler=str(llm.llm_engine.vllm_config.scheduler_config))),flush=True)
    for request in range(case.get('requests',3)):
        begin=time.monotonic();wanted={str(request)+'-'+str(row) for row in range(case['batch'])}
        for row in range(case['batch']):
            engine.add_request(str(request)+'-'+str(row),{'prompt_token_ids':prompt['token_ids']},params)
        first={};completed={};durations=[];events=[];first_all=None
        while engine.has_unfinished_requests():
            step=time.monotonic();outputs=engine.step();end=time.monotonic()
            if first_all is not None:durations.append(end-step)
            for output in outputs:
                assert output.request_id in wanted and len(output.outputs)==1
                assert list(output.prompt_token_ids)==prompt['token_ids']
                choice=output.outputs[0]
                if output.request_id not in first and (choice.token_ids or output.finished):
                    first[output.request_id]=end-begin
                events.append(dict(request_id=output.request_id,elapsed_seconds=end-begin,
                    token_ids=list(choice.token_ids),finished=output.finished,finish_reason=choice.finish_reason,
                    stop_reason=choice.stop_reason))
                if output.finished:completed[output.request_id]=choice
            if first_all is None and len(first)==case['batch']:first_all=end
        assert len(completed)==case['batch'] and len(first)==case['batch']
        choices=[completed[str(request)+'-'+str(row)] for row in range(case['batch'])]
        print('matched_result',json.dumps(dict(runtime='vllm-metal',model=model,case=case,request=request,
            clock='time.monotonic',initialization_seconds=initialization,ttft_seconds=max(first.values()),
            ttft_per_row_seconds=first,total_seconds=time.monotonic()-begin,
            prompt_token_ids=prompt['token_ids'],token_ids=[list(c.token_ids) for c in choices],text=[c.text for c in choices],
            finished=[c.finish_reason=='stop' for c in choices],stop_reasons=[c.stop_reason for c in choices],
            decode_step_seconds=durations,decode_steps_after_first=len(durations),
            decode_steps_per_second=len(durations)/sum(durations) if durations else None,events=events,
            decode_window_seconds=end-first_all,
            decode_wall_steps_per_second=len(durations)/(end-first_all) if durations else None,
            prefix_caching=False,speculative_decoding=False,configured_model_dtype=str(engine.model_config.dtype),
            weights='BF16 checkpoint; vLLM-Metal BF16 model/KV' if dtype=='bfloat16' else 'Exact F32 checkpoint expansion; vLLM-Metal FP32 model/KV')),
            flush=True)
    print('matched_process',json.dumps(dict(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        memory_scope='native macOS process RSS bytes; includes library initialization')),flush=True)

if __name__=="__main__":main()
