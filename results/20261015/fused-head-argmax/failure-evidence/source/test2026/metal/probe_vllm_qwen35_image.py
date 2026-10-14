"""Actual Qwen3.5 image compatibility probe; not a matched speed claim.

Arguments: checkpoint directory, prepared-image metadata JSON.
Processor/vision reuse and speculative decoding are disabled. Each request
opens the original image again and submits it to the normal vLLM pipeline.
"""
import hashlib,json,os,sys,time
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['VLLM_MLX_DEVICE']='gpu'

def main():
    from PIL import Image
    from vllm import LLM,SamplingParams
    model=Path(sys.argv[1]);metadata=json.loads(Path(sys.argv[2]).read_text())
    image=Path(metadata['image'])
    assert hashlib.sha256(image.read_bytes()).hexdigest()==metadata['image_sha256']
    chat=('<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>'
          'Describe the main objects in this image in one sentence.<|im_end|>\n'
          '<|im_start|>assistant\n<think>\n\n</think>\n\n')
    begin=time.monotonic()
    llm=LLM(model=str(model),dtype='bfloat16',max_model_len=256,
        max_num_seqs=1,max_num_batched_tokens=256,enable_prefix_caching=False,
        gpu_memory_utilization=0.30,disable_log_stats=True,seed=0,
        distributed_executor_backend='uni',limit_mm_per_prompt={'image':1},
        mm_processor_cache_gb=0,mm_processor_kwargs={'min_pixels':65536,'max_pixels':65536})
    print('image_probe_initialization',json.dumps(dict(seconds=time.monotonic()-begin,
        model=str(model),image_sha256=metadata['image_sha256'],
        intended_resized_hw=metadata['resized_hw'],intended_visual_tokens=54,
        comparison_scope='compatibility only; not yet a matched native timing comparison')),flush=True)
    engine=llm.llm_engine
    assert not engine.vllm_config.cache_config.enable_prefix_caching
    assert engine.vllm_config.speculative_config is None
    params=SamplingParams(temperature=0,max_tokens=32,seed=0)
    for request in range(3):
        start=time.monotonic()
        with Image.open(image) as original:frame=original.convert('RGB')
        # UUIDs identify all MM caches, including the worker's encoder cache.
        # A new identity per frame prevents reuse even when pixels are equal.
        frame_id='garnet-fresh-frame-'+str(time.monotonic_ns())
        engine.add_request(str(request),{'prompt':chat,'multi_modal_data':{'image':frame},
            'multi_modal_uuids':{'image':frame_id}},params)
        first=None;events=[];final=None
        while engine.has_unfinished_requests():
            outputs=engine.step();now=time.monotonic()
            for output in outputs:
                assert output.request_id==str(request) and len(output.outputs)==1
                choice=output.outputs[0]
                if first is None and (choice.token_ids or output.finished):first=now
                events.append(dict(seconds=now-start,token_ids=list(choice.token_ids),
                    finished=output.finished,finish_reason=choice.finish_reason,stop_reason=choice.stop_reason))
                if output.finished:final=output
        assert final is not None and first is not None
        assert len(final.prompt_token_ids)==79
        assert sum(token==248056 for token in final.prompt_token_ids)==54
        print('image_probe_result',json.dumps(dict(request=request,
            ttft_seconds=first-start,total_seconds=now-start,
            prompt_token_ids=list(final.prompt_token_ids),
            encoder_cache_identity=frame_id,visual_tokens=54,
            token_ids=list(final.outputs[0].token_ids),text=final.outputs[0].text,
            finish_reason=final.outputs[0].finish_reason,events=events)),flush=True)

if __name__=='__main__':main()
