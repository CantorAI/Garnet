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
- One-GEMM QKV projection with Q/K head RMSNorm and RoPE left in the same
  TensorRT fusion region
- One-GEMM gate/up projection with fused SiLU multiply
- Online-softmax paged decode with compact split statistics instead of a full
  score-matrix workspace
- Backend-neutral scheduler executor support for one-component text positions
  and three-component Qwen3-VL positions

## Decode results

| Graph | Batch | Average step | p95 step | Aggregate throughput |
|---|---:|---:|---:|---:|
| four-token prefill | 1 | 8.985 ms | — | — |
| single decode | 1 | 7.787 ms | 7.960 ms | 128.41 tokens/s |
| masked continuous decode | 4 | 7.577 ms | 8.822 ms | 527.90 tokens/s |

These are steady-state greedy decode measurements with GPU-resident paged KV
storage. Initial TensorRT optimization took about 4.7–4.8 minutes per static
shape on this machine; subsequent runs use the engine cache.

The native one-call test used the natural-language prompt
`Say: Garnet works.` with thinking disabled. Garnet tokenized the official
chat template to 18 tokens. On the cached rerun it generated 12 tokens in
92.79 ms total at 137.25 decode tokens/s and returned:

```text
Garnet works.
That's great to hear!
```

All measurements above use the official BF16 weights. FP8 and weight
quantization are intentionally deferred to a separate quantization phase.

## Regression coverage

- Real Qwen3-1.7B single decode from two official safetensors shards
- Real Qwen3-1.7B B4 masked decode with identical-row consistency
- Real Qwen3-1.7B paged prefill followed by decode from the populated cache
- Native chat-template tokenization and readable prompt-to-text output
- Existing Phase 24 compiled-runtime production guard
- Existing Qwen3-VL-2B paged decode
