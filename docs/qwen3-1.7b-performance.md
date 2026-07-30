# Qwen3-1.7B TensorRT baseline

Measured on the local RTX 4080 with the official BF16
`Qwen/Qwen3-1.7B` checkpoint.

## Implemented paths

- Pure-text model programs in `xModel/qwen3/text_1_7b`
- Standard one-dimensional Qwen3 RoPE
- Paged-KV prompt prefill
- Single-request paged decode
- Masked continuous-batch paged decode
- Native Qwen3 chat-template prompt-to-text generation
- Native Hugging Face sharded safetensors loading and TensorRT refit
- Backend-neutral scheduler executor support for one-component text positions
  and three-component Qwen3-VL positions

## Decode results

| Graph | Batch | Average step | p95 step | Aggregate throughput |
|---|---:|---:|---:|---:|
| four-token prefill | 1 | 8.985 ms | — | — |
| single decode | 1 | 8.275 ms | 8.897 ms | 120.84 tokens/s |
| masked continuous decode | 4 | 7.614 ms | 7.953 ms | 525.34 tokens/s |

These are steady-state greedy decode measurements with GPU-resident paged KV
storage. Initial TensorRT optimization took about 4.7–4.8 minutes per static
shape on this machine; subsequent runs use the engine cache.

The native one-call test used the natural-language prompt
`Say: Garnet works.` with thinking disabled. Garnet tokenized the official
chat template to 18 tokens. On the cached rerun it generated 12 tokens in
93.49 ms total at 135.71 decode tokens/s and returned:

```text
Garnet works.
That's great to hear!
```

## Regression coverage

- Real Qwen3-1.7B single decode from two official safetensors shards
- Real Qwen3-1.7B B4 masked decode with identical-row consistency
- Real Qwen3-1.7B paged prefill followed by decode from the populated cache
- Native chat-template tokenization and readable prompt-to-text output
- Existing Phase 24 compiled-runtime production guard
- Existing Qwen3-VL-2B paged decode
