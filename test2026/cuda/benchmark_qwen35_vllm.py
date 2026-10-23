"""WSL vLLM cohort. Launch only after the previous GPU engine has exited.

Cache budget is explicit, not vLLM's default percentage of all free VRAM.
This is an engine baseline, with native preprocessing/arithmetic differences
retained as limitations until prompt/grid/token-work equivalence is audited.
"""
import argparse,json,time,os,dataclasses
from pathlib import Path
from PIL import Image
import torch,vllm
from vllm import LLM,SamplingParams
from qwen35_benchmark_workload import batch_inputs,input_hashes

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint');parser.add_argument('dataset');parser.add_argument('output')
    parser.add_argument('--batch',type=int,choices=[1,2,4,8],default=1)
    parser.add_argument('--model',choices=['Qwen3.5-2B','Qwen3.5-4B'],default='Qwen3.5-2B')
    parser.add_argument('--rounds',type=int,default=7)
    parser.add_argument('--context-length',type=int,choices=[384,512],default=384)
    parser.add_argument('--memory-diagnostic',action='store_true')
    parser.add_argument('--memory-phase-output',help='Diagnostic phase holds/RPCs; never use for primary timing')
    parser.add_argument('--variants',action='store_true')
    parser.add_argument('--matched-batch-inputs',action='store_true')
    parser.add_argument('--speculative-mtp',type=int,choices=[0,1,3,7],default=0)
    parser.add_argument('--kv-cache-mib-per-seq',type=int,default=0,help='Explicit cache budget override; capacity remains the fixed context length')
    parser.add_argument('--conv-cache-dtype',choices=['float32','bfloat16','auto'],default='float32',
        help='GDN convolution cache; recurrent SSM state remains FP32. Distinct profiles preserve arithmetic provenance.')
    args=parser.parse_args()
    assert args.kv_cache_mib_per_seq>=0
    assert not args.matched_batch_inputs or args.variants
    if args.memory_phase_output:assert args.memory_diagnostic and args.rounds>=3
    phase_events=[]
    def phase(event,workers=None):
        if not args.memory_phase_output:return
        phase_events.append({'event':event,'unix_seconds':time.time(),'frontend_process_id':os.getpid(),
            'pid_namespace':'WSL Linux; not an assumed Windows PID','model':args.model,'worker_memory':workers,
            'scope':'Diagnostic holds and worker allocator snapshots; timings invalid for performance comparison'})
        Path(args.memory_phase_output).write_text(json.dumps(phase_events,indent=2))
    batch=args.batch
    options={'model':args.checkpoint,'dtype':'bfloat16','tensor_parallel_size':1,
        'max_model_len':args.context_length,'max_num_seqs':batch,'max_num_batched_tokens':max(args.context_length,batch*256),
        'enable_prefix_caching':False,'mm_processor_cache_gb':0,'limit_mm_per_prompt':{'image':1,'video':0},
        'mm_processor_kwargs':{'min_pixels':1024,'max_pixels':65536},
        'mamba_cache_dtype':args.conv_cache_dtype,'mamba_ssm_cache_dtype':'float32',
        'kv_cache_memory_bytes':batch*(args.kv_cache_mib_per_seq or (96 if args.model=='Qwen3.5-2B' else 128))*1024**2,
        'gpu_memory_utilization':0.9,'seed':0,'disable_log_stats':False}
    if args.memory_diagnostic:options['worker_extension_cls']='qwen35_vllm_memory.MemoryExtension'
    if args.speculative_mtp:options['speculative_config']={'method':'mtp','num_speculative_tokens':args.speculative_mtp}
    Path(args.output).with_suffix('.config.json').write_text(json.dumps({'vllm':vllm.__version__,
        'torch':torch.__version__,'options':options,
        'environment':{key:os.environ.get(key) for key in ['CUDA_HOME','CUDA_VISIBLE_DEVICES','VLLM_WORKER_MULTIPROC_METHOD']},
        'arithmetic_limitation':'BF16 model, FP32 recurrent state; residual and KV arithmetic can differ from Garnet.'},indent=2))
    print(json.dumps({'event':'load_begin','unix_seconds':time.time(),'options':options}),flush=True)
    phase('load_begin')
    llm=LLM(**options)
    if args.memory_phase_output:phase('loaded',llm.collective_rpc('qwen35_memory_snapshot'))
    sampling=SamplingParams(temperature=0,max_tokens=32,seed=0)
    print(json.dumps({'event':'load_complete','unix_seconds':time.time()}),flush=True)
    prompt='<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the main objects in this image in one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    images=['frame_0.jpg','frame_10050.jpg','frame_10200.jpg','frame_10350.jpg'];rows=[]
    for trial in range(args.rounds):
        if args.memory_diagnostic and trial==2:
            print(json.dumps({'event':'memory_peaks_reset_after_warmup','workers':llm.collective_rpc('qwen35_memory_snapshot',args=(True,))}),flush=True)
            if args.memory_phase_output:
                phase('steady_before_begin',llm.collective_rpc('qwen35_memory_snapshot'));time.sleep(8);phase('steady_before_end')
        for index in range(4):
            names=[images[(index+b)%4] for b in range(batch)]
            paths=[str(Path(args.dataset)/str(trial)/name if args.variants else Path(args.dataset)/name) for name in names]
            if args.matched_batch_inputs:names,paths=batch_inputs(args.dataset,trial,index,batch,True)
            hashes=input_hashes(paths)
            print(json.dumps({'event':'request_begin','trial':trial,'images':names,'unix_seconds':time.time()}),flush=True)
            started=time.perf_counter()
            requests=[]
            for path in paths:
                with Image.open(path) as im:decoded=im.convert('RGB')
                requests.append({'prompt':prompt,'multi_modal_data':{'image':decoded}})
            results=llm.generate(requests,sampling,use_tqdm=False)
            elapsed=time.perf_counter()-started
            records=[]
            for result in results:
                metrics=result.metrics
                if dataclasses.is_dataclass(metrics):metrics=dataclasses.asdict(metrics)
                elif metrics is not None:metrics=vars(metrics)
                records.append({'text':result.outputs[0].text,'token_ids':list(result.outputs[0].token_ids),
                    'prompt_token_ids':result.prompt_token_ids,'num_cached_tokens':result.num_cached_tokens,
                    'metrics':metrics,'finish_reason':result.outputs[0].finish_reason})
                assert not result.num_cached_tokens,'prefix cache hit despite disabled policy'
            row={'engine':'vllm','model':args.model,'batch':batch,'trial':trial,'warmup':trial<2,'images':names,
                'speculative':{'enabled':bool(args.speculative_mtp),'method':'native checkpoint MTP' if args.speculative_mtp else None,'num_speculative_tokens':args.speculative_mtp},
                'wall_seconds':elapsed,'request_output_limit':32,'records':records,'process_id':os.getpid(),
                'dataset':args.dataset,'variant_round':trial if args.variants else None,
                'input_sha256':hashes,'workload_group':index,'matched_batch_inputs':args.matched_batch_inputs,
                'timing_scope':'PIL JPEG decode, LLM.generate multimodal processing/prefill/decode and detokenization; load excluded',
                'cache_policy':{'prefix':False,'image_processor':False},
                'frontend_torch_memory_allocated_bytes':torch.cuda.memory_allocated(),
                'frontend_torch_memory_reserved_bytes':torch.cuda.memory_reserved(),
                'memory_scope':'Frontend allocator only; engine worker/process VRAM requires a separate diagnostic run.'}
            if args.memory_diagnostic:row['worker_memory']=llm.collective_rpc('qwen35_memory_snapshot')
            rows.append(row);Path(args.output).write_text(json.dumps(rows,indent=2))
            print(json.dumps({'event':'request_complete','trial':trial,'images':names,'unix_seconds':time.time(),
                'wall_seconds':elapsed,'generated_tokens':[len(r['token_ids']) for r in records]}),flush=True)
    if args.memory_phase_output:
        phase('steady_after_begin',llm.collective_rpc('qwen35_memory_snapshot'));time.sleep(8);phase('steady_after_end')
        phase('complete',llm.collective_rpc('qwen35_memory_snapshot'))
    print('QWEN35_VLLM_BENCHMARK_COMPLETE',args.model,'batch',batch,'requests',len(rows),flush=True)

if __name__=="__main__":main()
