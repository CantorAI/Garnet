# Exact-value FP32 precision study; full goal still unmet

This additional comparison aligns arithmetic precision instead of relying
on the comparator's configured dtype label. All 72 requests have matching
complete generated token IDs. At batch 2, Garnet exceeds 2× the FP32
comparator's decode throughput for coder and VL. No workload reaches half
the comparator's TTFT, and text decode remains below 2×. These results do
not establish a 2× speedup over the default BF16 comparator; its prior
results remain separate and unchanged.

No new native optimization is selected in this experiment. Garnet uses the
existing validated MPS-score library, original BF16 checkpoint storage and
FP32 activations/KV. Its exact-scaled FP16 prefill cache preserves original
weight values. The comparator uses separate, exactly expanded F32 versions
of those values and verified FP32 weights, activations and KV. File hashes
and storage sizes differ; learned numerical values are identical.

## Exact value and actual runtime verification

The source models are read-only. A streaming converter expands each finite
BF16 bit pattern to its exact F32 representation (`uint32(bits) << 16`),
preserves any existing F32 bits, and verifies every destination value and
per-tensor SHA before publishing the new file. It updates only dtype config
fields; tokenizers, generation and image-processing assets are copied.

| Model | Serialized tensors | Values verified bit-exact |
| --- | ---: | ---: |
| Qwen3-0.6B | 311 | 751,632,384 |
| Qwen2.5-Coder-0.5B-Instruct | 290 | 494,032,768 |
| Qwen3.5-0.8B | 488 | 873,438,784 |

Serialized counts include entries not used by the runtime, such as redundant
tied heads; actual loaded-module counts are recorded separately. Three GPU
worker audits find all 310 / 290 / 473 loaded parameter tensors FP32, including
private paged-attention owners. Embedding, first norm and attention projection
outputs and KV dtypes are also FP32. Merely requesting `dtype=float32` with
the original BF16 files does not accomplish this, as the prior dtype audit
demonstrates.

The FP32 VL model failed at the original 0.30 memory fraction before cache
initialization. The final study uses 0.45 for VL and retains 0.30 for dense
models. Actual model length remains 256 and batch 1/2; the larger allowance
permits the FP32 model/cache to fit and is explicitly recorded. No inference
math, comparator loaders or security settings are patched for this study.
FP32 follows the installed comparator's normal dtype-dependent dispatch,
which differs from its optimized BF16 paths. Treat this as a separate
precision configuration, not an improvement over the default BF16 baseline.

## Actual sequential warm measurements

Same M4 / 16 GiB, same learned weights, prompt IDs, greedy sampling, batch,
context cap and output cap. No prefix, encoder-cache or speculative reuse.
One runtime exits before the next starts. Each configuration has two fresh
processes, each serving three requests; warm means exclude first requests.
Throughput is decode steps per second per row, including the full wall
interval between first and final prediction.

| Model | Batch | Garnet tokens/s/row | FP32 vLLM tokens/s/row | Decode ratio | Garnet TTFT | FP32 vLLM TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 57.07 | 37.27 | 1.53× | 69.98 ms | 83.79 ms |
| Qwen3-0.6B | 2 | 52.01 | 29.29 | 1.78× | 115.29 ms | 111.27 ms |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 72.36 | 47.29 | 1.53× | 55.05 ms | 62.20 ms |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 69.91 | 29.64 | 2.36× | 73.90 ms | 84.09 ms |
| Qwen3.5-0.8B image | 1 | 47.21 | 26.91 | 1.75× | 547.89 ms | 252.79 ms |
| Qwen3.5-0.8B image | 2 | 42.81 | 16.62 | 2.58× | 760.50 ms | 438.50 ms |

All 48 text/coder and 24 image requests match outputs and finish. The FP32
batch-2 image outputs match the native 28-token caption; the older BF16
comparator batch-2 result did not match and reached the cap. Text/coder
generate 8/17 tokens including stop under a 64-token cap. VL uses the same
288×192 RGB image, 54 visual tokens, 79 prompt IDs and 32-token cap. Results
are limited to these agreed prompts; they do not prove model-wide speed.

Initialization, first-use requests and individual-process high-water RSS are
retained separately in raw records. Filesystem/shader caches are not flushed,
so first-use measurements are not cold boots; RSS is not unique unified GPU
allocation or aggregate multi-process memory.

## Evidence and target audit

`study-source/` retains actual helpers and historical failed-audit sources.
`equivalent-model-metadata/` contains configs, asset identity and complete
per-tensor equivalence proofs. Commands, successful and failed exits, process
guards, library/model hashes, versions, request IDs and all timings/tokens
are saved. Models and raw image buffers are excluded. Original and expanded
checkpoint hashes were revalidated when archiving.

`exact-f32-actual-dtypes-fit-20261013/jobs/summary.json` proves actual FP32
loading and activation probes. `equivalent-f32-text-20261013/jobs/summary.json`
and `equivalent-f32-vl-20261013/jobs/frame_summary.json` are the performance
summaries. Both explicitly set `counts_as_default_bf16_speedup=false`.

All six half-TTFT gates fail; four of six FP32 decode gates also fail.
The default BF16 2× target is unchanged and unmet. The full goal stays active;
no claim of completion or general 2× acceleration is justified.
