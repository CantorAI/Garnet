# Parallel recurrent reductions: validated and retained

The optional `GARNET_METAL_PARALLEL_DELTA_SPLIT` kernel improves actual
Qwen3.5 image latency and decode throughput. Two native ABBA comparisons
show matching complete outputs and repeatable gains. The full 2× decode /
half-TTFT goal remains unmet. The option is OFF by default; the tested
investigation build enables it with exact-half prefill and MPS vision scores.

## Implementation and correctness

The existing recurrent kernel assigns one thread to each output value and
loops over 128 key elements serially. The candidate assigns four output
values to each SIMD group, distributes key terms over 32 lanes and uses
SIMD reductions. Each threadgroup handles 16 values. This increases useful
parallelism and shortens dependent dot-product loops. The original key-major
state layout and all learned values are retained. No weight quantization,
prefix reuse or speculation is introduced.

The fast path requires key dimension 128, value dimension divisible by 16,
and aligned state/output strides. Other shapes use the prior kernel.
Paused rows copy state bit-for-bit and produce zero output. Active FP32 dot
reductions change addition order, so their intermediate bits need not match
the old serial reduction. Independent FP64 output/state oracles use a
3e-6 absolute tolerance. Tests retain strict bitwise checks for fallback and
paused paths, pending-output rejection, abort recovery and in-place rejection.

All eleven handoff jobs pass, including actual Qwen3.5 layer/state references,
dense text/coder layer references, all three complete heads, serving/reset/
shared-state behavior and small/pretrained vision references. The option OFF
also compiles and passes CTest. All 48 requests across the two native frame
comparisons match complete control token IDs and finish.

## Actual native ABBA measurements

Same M4 / 16 GiB, Qwen3.5-0.8B, 288×192 image, 54 visual tokens, 79 prompt
IDs, chunk 16, capacity 256 and cap 32. Every frame is freshly encoded and
decoder state reset. Each run has twelve requests per batch; warm means
exclude first use in each fresh worker. Only one inference runtime runs.

| Run | Batch | Control TTFT | Candidate TTFT | TTFT reduction | Control tokens/s/row | Candidate tokens/s/row |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial valid | 1 | 549.54 ms | 491.62 ms | 10.54% | 47.15 | 48.06 |
| Initial valid | 2 | 760.61 ms | 700.48 ms | 7.91% | 42.93 | 44.56 |
| Confirm | 1 | 549.16 ms | 490.29 ms | 10.72% | 47.19 | 48.27 |
| Confirm | 2 | 764.32 ms | 703.32 ms | 7.98% | 43.20 | 44.41 |

Confirmation decode gains are 2.29% / 2.79%. Existing non-serialized session
counters attribute the change to GPU prefill: 374.90 → 317.97 ms at batch 1,
524.88 → 460.73 ms at batch 2. CPU encoding stays about 49 ms. No serialized
per-operation diagnostic timing is used as serving performance.

First-use confirmation measurements remain separate:

| Batch | Control initialization | Candidate initialization | Control first TTFT | Candidate first TTFT |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3.102 s | 3.061 s | 846.23 ms | 799.57 ms |
| 2 | 3.113 s | 3.103 s | 1,047.06 ms | 980.56 ms |

These are fresh processes with existing filesystem/shader caches, not cold
boots. Individual-process RSS remains distinct from unique unified GPU
allocations and aggregate multi-process memory.

## Fresh stock BF16 comparator results

These observed same-checkpoint results retain the prior precision limitation:
native activations/KV are FP32, stock comparator activations/KV are BF16.
All stock-comparator throughput and half-TTFT gates remain false.

| Model | Batch | Garnet tokens/s/row | vLLM tokens/s/row | Garnet TTFT | vLLM TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 56.76 | 74.35 | 71.34 ms | 49.76 ms |
| Qwen3-0.6B | 2 | 51.97 | 75.95 | 116.03 ms | 70.92 ms |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 72.58 | 92.09 | 54.80 ms | 34.78 ms |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 69.85 | 92.25 | 74.48 ms | 38.39 ms |
| Qwen3.5-0.8B image | 1 | 48.24 | 48.26 | 488.11 ms | 203.05 ms |
| Qwen3.5-0.8B image | 2 | 43.16 | 47.34 | 713.55 ms | 329.11 ms |

Stock image batch 2 still has differing outputs and comparator cap exhaustion;
it cannot establish a matched-output speed claim. Text/coder and image batch
1 outputs match. Text/coder kernels are unaffected by this recurrent change;
their timings remain close to prior native measurements.

## Separate exact-value FP32 comparison

The original learned values are compared against the previously verified
lossless F32 copies, with actual FP32 weights, tested activations and KV.
The comparator uses memory fraction 0.30 for dense models and 0.45 for VL
so the expanded model/cache fits. These are separate precision-configuration
results and do not count as beating the default BF16 baseline.

| Model | Batch | Garnet tokens/s/row | FP32 vLLM tokens/s/row | Garnet TTFT | FP32 vLLM TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 57.05 | 38.05 | 70.57 ms | 79.26 ms |
| Qwen3-0.6B | 2 | 51.53 | 24.24 | 118.52 ms | 106.00 ms |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 72.31 | 47.18 | 55.69 ms | 63.11 ms |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 69.80 | 29.41 | 74.84 ms | 81.44 ms |
| Qwen3.5-0.8B image | 1 | 48.13 | 26.28 | 490.92 ms | 261.23 ms |
| Qwen3.5-0.8B image | 2 | 44.55 | 16.65 | 699.27 ms | 436.31 ms |

All 72 FP32 study requests match complete outputs. Coder/vision batch-2
decode ratios are 2.37× / 2.68×, while half-TTFT fails everywhere. The observed
2.13× text batch-2 ratio reflects a slower comparator than the prior run
(24.24 versus 29.29 tokens/s); native text speed did not improve. It is not
a confirmed new text acceleration and must not be used as evidence that
this recurrent optimization made dense text 2× faster.

## Transfer/rebuild provenance and evidence

The first build started before failed SSH source transfers were confirmed.
C++ compilation accepted the raw embedded string, but CTest caught its
incomplete Metal shader before any inference benchmark. The next build
reused stale objects because extracted source timestamps were older. Both
failed chains and their blocked benchmark successors are retained. A source
bundle was then uploaded in small hash-checked pieces, fully SHA-verified,
atomically installed, and rebuilt with `--clean-first`. All reported valid
measurements use that verified clean rebuild. No failed attempt supplies a
speed result, and the selected control binary was never rebuilt or modified.

`candidate-source/` contains actual source, flags, helpers, verified install
manifest, hashes, failure snapshot and CTest details. Job directories record
commands, terminal exits, process guards, all request timing/token IDs and
image/model identity. Raw image buffers are excluded. The candidate library
is `9bbac5c9ad7486e1fe70690c526bebaed7fcfa06e4d162a3fe257af0a6faddef`;
the control remains
`c9f8c76aa2657669d6a7b10eba459481ac797fc64904d25360933b565f9d77d1`.

The complete goal remains active: repeated 2× decode and half-TTFT across all
agreed models/batches with matched precision. Current measurements contradict
completion, despite a useful validated native latency improvement.
