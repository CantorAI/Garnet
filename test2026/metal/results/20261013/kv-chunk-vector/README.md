# Validated vector chunk KV-cache updates

The selected investigation build is `build-kv-chunk-vector`, library SHA-256
`564b515ab6b0eb68413ef5a54de6beff47d333f8f1fa6a1188562bf6e5c28ec4`. It retains the optional exact-half
prefill setting used by its `build-exact-half-prefill-hoisted` control.

Aligned chunk KV updates copy four FP32 values per thread using 32-bit
addressing. Head dimensions and data strides must be divisible by four,
and full token/cache/output/control addressing must fit 32 bits. Unsupported
layouts or unavailable optional pipelines retain the scalar path. Contiguous
active prefixes keep direct slot lookup; gaps, duplicate slots and paused
rows preserve latest-active-token selection. Arithmetic, precision, cache
capacity, checkpoint values and generated-output limits are unchanged.

## Validation and native control

All 11 native numerical/serving/vision checks passed. CTest additionally
checks 144 independent bitwise cache references across batch 1/2, heads 1/3,
dimensions 5/8/128, chunks 2/4/16 and four update patterns: contiguous,
paused tails, gaps/resumed rows and duplicate-slot overwrites. Odd dimensions
exercise scalar fallback; signed zeros and varied finite cache bits are kept.

The first build failed due to a mixed-type auto declaration; explicit size_t
fixes it. The failed build and dependency-gated jobs remain saved separately.
No numerical or timed trial ran from that failed attempt.

The close ABBA comparison executed 48 text/coder requests and 24 real image
requests, strictly sequentially. All complete prompts, outputs and finish
flags match. Each row averages four warm requests per build/profile; first-use
and initialization are separate, with filesystem caches unflushed. Images use
identical RGB/patch bytes and freshly execute vision on every request.

| Model | Batch | Control TTFT, ms | Candidate TTFT, ms | Reduction | Decode ratio |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 88.19 | 70.36 | 20.22% | 1.006 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 56.86 | 54.38 | 4.36% | 0.999 |
| Qwen3-0.6B | 2 | 152.57 | 114.76 | 24.79% | 1.001 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 78.02 | 75.27 | 3.53% | 0.999 |
| Qwen3.5-0.8B, image | 1 | 560.22 | 558.94 | 0.23% | 0.995 |
| Qwen3.5-0.8B, image | 2 | 798.21 | 789.79 | 1.05% | 0.997 |

The substantial Qwen3 TTFT reduction supports retaining the candidate. Coder
gains are modest; image differences are small. Decode is essentially unchanged
because this optimization affects multi-token cache updates. Startup and
individual native text/coder high-water RSS stay close to the control; full
first-use, initialization and RSS values are in startup_summary.json. RSS
is not aggregate or unique GPU allocation. The inherited exact-half path
still has the separately documented raw-control startup/memory tradeoff.

The actual batch-2 text diagnostic preserves its complete output. Summed
serialized KVChunk GPU attribution falls from 38.75 to 5.24 ms.
These per-operation submissions perturb cache/dispatch behavior and are not
serving latency or a comparator speed claim. The native ABBA table above
establishes the full-request benefit.

## Fresh actual vLLM-Metal comparison

Another 48 text/coder and 24 actual image requests compare the selected build
with the unchanged isolated vLLM-Metal installation on the same M4 16 GiB
Mac mini. Processes never overlap. BF16 checkpoint files, prompts, image
geometry, greedy generation, capacity 256, batch 1/2 and caps remain frozen.
Prefix caching, encoder reuse and speculative decoding are disabled.

| Model | Batch | Garnet tokens/s per row | vLLM tokens/s per row | Garnet TTFT, ms | vLLM TTFT, ms |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 56.92 | 74.02 | 70.00 | 49.58 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 72.48 | 92.27 | 54.97 | 34.90 |
| Qwen3-0.6B | 2 | 52.08 | 76.28 | 115.03 | 71.70 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 70.01 | 92.24 | 74.52 | 38.11 |
| Qwen3.5-0.8B, image | 1 | 47.17 | 48.41 | 562.71 | 202.02 |
| Qwen3.5-0.8B, image | 2 | 43.05 | 47.37 | 794.98 | 331.05 |

All six profiles still miss both the 2x throughput and half-TTFT targets.
Text/coder and image batch 1 preserve exact complete comparator outputs.
Image batch 2 retains the different/capped comparator caption: native finishes
28 tokens; vLLM reaches the shared 32-token cap. That row is matched-input
timing with a quality/length caveat. Native FP32 vision/activations/KV versus
comparator BF16 internal arithmetic remains explicit despite the same BF16
checkpoint values. The full objective stays active and unmet.

Commands/exits, failed-attempt evidence, source/model/input/library hashes,
numerical logs, complete outputs, runtime guards, diagnostic attribution and
separate first-use/warm records are retained. Private raw RGB/patch arrays and
binaries remain on the Mac. Protected GPT-OSS, Pi and original Mac checkouts
are untouched.
