"""Persistent vLLM image worker; unique encoder IDs for every row/frame."""
import json,os,sys,time
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['VLLM_MLX_DEVICE']='gpu'

def main():
    from PIL import Image
    from vllm import LLM,SamplingParams
    model=sys.argv[1];workload=json.loads(Path(sys.argv[2]).read_text());control=Path(sys.argv[3])
    dtype=workload.get('comparator_dtype','bfloat16')
    if dtype not in ('bfloat16','float32'):raise ValueError('Unsupported comparator dtype')
    memory_fraction=workload.get('comparator_gpu_memory_utilization',.30)
    if not .1<=memory_fraction<=.6:raise ValueError('Unsupported comparator memory fraction')
    def emit(kind,payload):
        label='ready' if kind=='frame_ready' else str(payload['request'])+'-'+kind
        Path(str(control)+'.event-'+label+'.json').write_text(json.dumps(payload))
    batch=workload['batch'];rh,rw=workload['image_metadata']['resized_hw']
    llm=LLM(model=model,dtype=dtype,max_model_len=workload['capacity'],max_num_seqs=batch,
        max_num_batched_tokens=workload['capacity'],enable_prefix_caching=False,gpu_memory_utilization=memory_fraction,
        disable_log_stats=True,seed=0,distributed_executor_backend='uni',limit_mm_per_prompt={'image':1},
        mm_processor_cache_gb=0,mm_processor_kwargs={'min_pixels':rh*rw,'max_pixels':rh*rw})
    engine=llm.llm_engine
    assert not engine.vllm_config.cache_config.enable_prefix_caching and engine.vllm_config.speculative_config is None
    expected=workload['native_prompt'];params=SamplingParams(temperature=0,max_tokens=workload['max_tokens'],
        seed=0,stop_token_ids=expected['stop_token_ids'])
    chat=('<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>'
          'Describe the main objects in this image in one sentence.<|im_end|>\n'
          '<|im_start|>assistant\n<think>\n\n</think>\n\n')
    emit('frame_ready',expected);last=-1
    while True:
        time.sleep(.001);message=json.loads(control.read_text())
        if message['request']==last:continue
        last=message['request']
        if message.get('stop'):break
        start=time.monotonic();frame=Image.frombytes('RGB',(rw,rh),Path(message['rgb_file']).read_bytes())
        identities=[]
        for row in range(batch):
            identity='garnet-frame-'+str(time.monotonic_ns())+'-'+str(row);identities.append(identity)
            engine.add_request(str(last)+'-'+str(row),{'prompt':chat,'multi_modal_data':{'image':frame},
                'multi_modal_uuids':{'image':identity}},params)
        first_rows={};first=None;final={};events=[];calls=0;first_call=None
        while engine.has_unfinished_requests():
            outputs=engine.step();now=time.monotonic();calls+=1
            for output in outputs:
                assert list(output.prompt_token_ids)==expected['prompt_token_ids']
                choice=output.outputs[0];row=int(output.request_id.rsplit('-',1)[1])
                if row not in first_rows and (choice.token_ids or output.finished):first_rows[row]=now
                events.append(dict(row=row,elapsed_seconds=now-start,token_ids=list(choice.token_ids),
                    finished=output.finished,finish_reason=choice.finish_reason,stop_reason=choice.stop_reason))
                if output.finished:final[row]=choice
            if first is None and len(first_rows)==batch:
                first=now;first_call=calls
                emit('frame_first',dict(request=last,worker_ttft_seconds=first-start))
        assert len(final)==batch and first is not None
        emit('frame_done',dict(request=last))
        ids=[]
        for row in range(batch):
            choice=final[row];tokens=list(choice.token_ids)
            if choice.finish_reason=='stop' and (not tokens or tokens[-1] not in expected['stop_token_ids']):
                stop=choice.stop_reason if isinstance(choice.stop_reason,int) else expected['stop_token_ids'][0]
                tokens.append(stop)
            ids.append(tokens)
        emit('frame_result',dict(request=last,prompt_token_ids=expected['prompt_token_ids'],token_ids=ids,
            text=[final[row].text for row in range(batch)],finished=[final[row].finish_reason=='stop' for row in range(batch)],
            decode_steps_after_first=calls-first_call,worker_ttft_seconds=first-start,worker_total_seconds=now-start,
            worker_decode_window_seconds=now-first,encoder_cache_identities=identities,events=events,
            prefix_caching=False,speculative_decoding=False,configured_model_dtype=str(engine.model_config.dtype),
            weights='BF16 checkpoint; BF16 comparator vision/model/KV' if dtype=='bfloat16' else 'Exact F32 checkpoint expansion; FP32 comparator vision/model/KV'))

if __name__=='__main__':main()
