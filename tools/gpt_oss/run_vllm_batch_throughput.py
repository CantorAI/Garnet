"""Python: run_vllm_batch_throughput.py MODEL REQUEST RESULT BATCH OUTPUT_TOKENS.

Run a homogeneous, fixed-output TP2 vLLM batch without competing GPU work.
Requires a vLLM environment; this script does not install vLLM.
"""
import asyncio
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path


assert len(sys.argv) == 6, 'expected model, request, result, batch, output tokens'
model = Path(sys.argv[1]).resolve()
request = json.loads(Path(sys.argv[2]).read_text())
result_path = Path(sys.argv[3])
batch, output_tokens = int(sys.argv[4]), int(sys.argv[5])
assert 1 <= batch <= 16 and 16 <= output_tokens <= 512
assert request['input_ids'] and len(request['input_ids']) + output_tokens <= 4096


async def main():
    import vllm
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    settings = dict(model=str(model), tensor_parallel_size=2,
                    dtype='bfloat16', kv_cache_dtype='auto',
                    max_model_len=4096, max_num_seqs=batch,
                    max_num_batched_tokens=4096,
                    enable_prefix_caching=False, enable_chunked_prefill=True,
                    gpu_memory_utilization=.8, seed=0)
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**settings))
    params = SamplingParams(temperature=0, max_tokens=output_tokens,
                            ignore_eos=True, seed=0)

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

    try:
        await asyncio.gather(*(generate(i, 'warmup') for i in range(batch)))
        started = time.perf_counter()
        outputs = await asyncio.gather(*(generate(i, 'bench') for i in range(batch)))
        finished = time.perf_counter()
    finally:
        engine.shutdown()

    # Bound the decode window from the earliest first token to the last token;
    # if vLLM staggers requests, that delay stays in the aggregate result.
    first_all = min(row['first_token_ts'] for row in outputs)
    last_any = max(row['last_token_ts'] for row in outputs)
    decode_seconds = last_any - first_all
    aggregate_tokens = batch * (output_tokens - 1)
    result_path.write_text(json.dumps({
        'vllm_version': vllm.__version__, 'settings': settings,
        'measurement': 'homogeneous fixed-size TP2 batch, greedy, no early stop',
        'input_tokens_per_request': len(request['input_ids']),
        'output_tokens_per_request': output_tokens, 'batch': batch,
        'token_ids_by_request': [row['token_ids'] for row in outputs],
        'request_timings': [{k: row[k] for k in ('first_token_ts', 'last_token_ts')}
                            for row in outputs],
        'decode_seconds': decode_seconds,
        'decode_output_tokens': aggregate_tokens,
        'decode_aggregate_output_tokens_per_second':
            aggregate_tokens / decode_seconds if decode_seconds > 0 else None,
        'full_request_wall_seconds': finished - started,
        'full_request_output_tokens_per_second':
            batch * output_tokens / (finished - started),
    }, indent=2))
    print('Decode aggregate output tok/s',
          aggregate_tokens / decode_seconds, flush=True)


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
