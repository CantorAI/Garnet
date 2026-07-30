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
