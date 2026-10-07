"""Python: run_vllm_batch_throughput.py MODEL REQUEST RESULT BATCH OUTPUT_TOKENS.

Run a homogeneous, fixed-output TP2 vLLM batch without competing GPU work.
Requires a vLLM environment; this script does not install vLLM.
"""
import asyncio
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path


assert len(sys.argv) == 6, 'expected model, request, result, batch, output tokens'
model = Path(sys.argv[1]).resolve()
request = json.loads(Path(sys.argv[2]).read_text())
result_path = Path(sys.argv[3])
batch, output_tokens = int(sys.argv[4]), int(sys.argv[5])
assert 1 <= batch <= 128 and 16 <= output_tokens <= 512
capacity = int(os.environ.get('VLLM_BATCH_CONTEXT_CAPACITY', '4096'))
assert request['input_ids'] and len(request['input_ids']) + output_tokens <= capacity <= 4096
trial_count = int(os.environ.get('VLLM_BATCH_TRIALS', '1'))
max_batched_tokens = int(os.environ.get('VLLM_BATCH_MAX_BATCHED_TOKENS', '4096'))
assert 1 <= trial_count <= 5 and batch <= max_batched_tokens <= 65536


async def main():
    import vllm
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    settings = dict(model=str(model), tensor_parallel_size=2,
                    dtype='bfloat16', kv_cache_dtype='auto',
                    max_model_len=capacity, max_num_seqs=batch,
                    max_num_batched_tokens=max_batched_tokens,
                    enable_prefix_caching=False, enable_chunked_prefill=True,
                    gpu_memory_utilization=.8, seed=0)
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**settings))
    params = SamplingParams(temperature=0, max_tokens=output_tokens,
                            ignore_eos=True, seed=0)

    def gpu_memory_mib():
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            text=True)
        return [int(value.strip()) for value in output.splitlines()]

    engine_memory_mib = gpu_memory_mib()

    async def generate(index, prefix):
        result = None
        async for event in engine.generate(
                {'prompt_token_ids': request['input_ids']}, params,
                request_id=f'{prefix}-{index}'):
            result = event
        assert result is not None and len(result.outputs[0].token_ids) == output_tokens
        if result.num_cached_tokens:
            raise RuntimeError('Prefix reuse invalidates the full-prompt timing')
        metrics = result.metrics
        assert metrics is not None
        return dict(token_ids=list(result.outputs[0].token_ids),
                    first_token_ts=metrics.first_token_ts,
                    last_token_ts=metrics.last_token_ts)

    measurements = []
    memory_samples = []
    try:
        await asyncio.gather(*(generate(i, 'warmup') for i in range(batch)))
        warmed_memory_mib = gpu_memory_mib()
        for trial in range(trial_count):
            started = time.perf_counter()
            outputs = await asyncio.gather(*(generate(i, f'bench-{trial}') for i in range(batch)))
            finished = time.perf_counter()
            # Keep staggered first-token delivery inside the decode window.
            first_all = min(row['first_token_ts'] for row in outputs)
            last_any = max(row['last_token_ts'] for row in outputs)
            decode_seconds = last_any - first_all
            aggregate_tokens = batch * (output_tokens - 1)
            measurements.append({
                'trial': trial, 'prefill_kv_reused_for_decode_trial': False,
                'token_ids_by_request': [row['token_ids'] for row in outputs],
                'identical_output_across_duplicate_requests':
                    all(row['token_ids'] == outputs[0]['token_ids'] for row in outputs),
                'benchmark_started_ts': started,
                'request_timings': [{k: row[k] for k in ('first_token_ts', 'last_token_ts')}
                                    for row in outputs],
                'request_first_token_seconds': [row['first_token_ts'] - started for row in outputs],
                'request_completion_seconds': [row['last_token_ts'] - started for row in outputs],
                'decode_seconds': decode_seconds, 'decode_output_tokens': aggregate_tokens,
                'decode_aggregate_output_tokens_per_second': aggregate_tokens / decode_seconds,
                'full_request_wall_seconds': finished - started,
                'full_request_output_tokens_per_second': batch * output_tokens / (finished - started)})
            memory_samples.append(gpu_memory_mib())
            print('Decode trial', trial, 'aggregate output tok/s',
                  measurements[-1]['decode_aggregate_output_tokens_per_second'], flush=True)
    finally:
        engine.shutdown()

    result_path.write_text(json.dumps({
        'vllm_version': vllm.__version__, 'settings': settings,
        'measurement': 'homogeneous fixed-size TP2 batch, greedy, no early stop',
        'input_tokens_per_request': len(request['input_ids']),
        'input_token_ids': request['input_ids'],
        'output_tokens_per_request': output_tokens, 'batch': batch,
        'max_context_tokens_per_request': capacity,
        'gpu_memory_mib_after_engine_start': engine_memory_mib,
        'gpu_memory_mib_after_warmup': warmed_memory_mib,
        'gpu_memory_mib_after_benchmark': memory_samples[-1],
        'gpu_memory_mib_after_trials': memory_samples,
        'decode_trials': measurements,
        **measurements[0],
    }, indent=2))


if __name__ == '__main__':
    lock_path = Path('/workspace/CantorAI/work/gpu-benchmark.lock')
    with lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        processes = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
            text=True)
        if processes.strip():
            raise RuntimeError('Refusing benchmark: GPUs already have compute processes')
        asyncio.run(main())
