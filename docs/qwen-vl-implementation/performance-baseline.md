# Qwen3-VL-2B TensorRT Performance Baseline

Hardware: NVIDIA RTX 4080, BF16, Windows, TensorRT optimization level 5.

## 2026-07-30

| Path | Batch | Average step | P95 step | Aggregate throughput |
|---|---:|---:|---:|---:|
| Existing paged decode | 1 | 7.72 ms | 8.42 ms | about 129.5 tok/s |
| Masked continuous-decode graph | 4 | 6.86 ms | 7.09 ms | 582.87 tok/s |

The B4 result uses the complete 28-layer Qwen3-VL-2B-Instruct text decoder,
TensorRT dense lowering, the BF16 masked paged-KV plugin, and batched GPU greedy
sampling. It is not a synthetic one-layer throughput number.

The B4 result uses persistent input/output tensor addresses and TensorGraph
`cuda_graph=True` replay. The equivalent nonpersistent-binding run measured
507.65 tok/s. The first B4 engine build took 316.7 seconds. Loading the cached
engine and refitting mapped safetensors took about 2.2 seconds.

## Real-image pipeline

The reorganized `xModel/qwen3/vl_2b_instruct/qwen_vl_prefill.x` pipeline was
validated with real JPEG frames from `data/Dataset.1980Love/imgs`.

| Input | End-to-end request | Generated text |
|---|---:|---|
| `frame_0.jpg` | 90.19 ms | `A man sits in a wooden hut, surrounded by` |
| `frame_10050.jpg` | 89.19 ms | `A woman in a plaid shirt and long skirt` |
| `frame_10200.jpg` | 89.52 ms | `A man and a woman stand in a dimly` |
| `frame_10350.jpg` | 90.94 ms | `A woman in a plaid shirt and blue skirt` |

The profile used 240 native vision patches, 60 merged visual tokens, 77 total
prompt tokens, and 10 greedy output tokens. The first request measured
118.14 ms and the repeated warm request measured 89.42 ms. The one-time
rebuild after moving the model source took 144.0 seconds.

## Guardrails

- Inactive batch rows must produce zero attention output and must not modify KV.
- KV pages remain stable while requests join and leave scheduler batches.
- Decode metadata obeys `context_length = slot_position + 1`.
- Context lengths above 4,096 must execute through flash paged decode.
- Unsupported graph operations fail compilation explicitly.
- Performance comparisons use warm iterations and report both mean and p95.

## Remaining Serving Work

The measured B4 graph, backend-neutral scheduler, paged pool, and persistent
compiled decode executor are implemented. A production asynchronous serving
session still needs to own all B1/B2/B4/B8 executor instances, stream tokens to
callers, and integrate batched prefill admission. Until that session is
complete, this result is a verified continuous-decode milestone rather than a
claim that the entire concurrent HTTP serving stack is finished.
