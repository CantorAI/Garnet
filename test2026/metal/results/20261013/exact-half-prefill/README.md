# Validated exact scaled FP16 MPS prefill

The selected investigation build is `build-exact-half-prefill-hoisted`,
library SHA-256 `cf3b155e219458c7ee511279ba3f63a3a88968d696b9bed172f2219357ff52b8`.
Configure with `-DGARNET_METAL_EXACT_HALF_PREFILL=ON`; the option remains
off by default so the original low-memory raw BF16 path is available.

Body matrices up to 16 MiB are represented in FP16 using per-row powers of
two, without rounding checkpoint values. An integer codec verifies exact
representability, with whole-matrix fallback for any incompatible value/range.
Raw BF16 storage is retained for decode and large heads. Multi-token prefill
uses mixed MPS FP32 activation/output with FP16 weight storage, followed by
inverse per-output-column rescaling. Activations, accumulation and KV remain
FP32; this is an exact weight encoding, not reduced-weight-precision inference.

## Verification and native controls

The final build passed all 11 native checks, covering real dense/Qwen3.5
layers and heads, vision, partial tails, paused rows, gaps, reset and shared
graph sessions. The integer codec matches an independent floating conversion
and bit-roundtrip oracle for all 65,280 finite BF16 encodings at all 129
supported scales: 8,421,120 cases. Row-scale selection also matches frexp
for every finite positive BF16 magnitude. Signed zero, odd/padded layouts,
per-row scaling and a rejected encoding with working raw fallback are tested.

Native ABBA executed 48 text/coder requests and 24 actual image requests
against `build-delta-split`, sequentially. Every complete output, prompt and
finish flag matches. Each row averages four warm requests per build/profile.
Initialization and first-use requests remain separate; filesystem caches are
unflushed. Images use equal RGB/patch bytes and fresh vision encoding.

| Model | Batch | Control TTFT, ms | Candidate TTFT, ms | Reduction | Decode ratio |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 102.66 | 87.91 | 14.38% | 1.000 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 71.84 | 58.09 | 19.14% | 0.999 |
| Qwen3-0.6B | 2 | 174.62 | 153.82 | 11.91% | 0.996 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 100.83 | 77.93 | 22.71% | 0.994 |
| Qwen3.5-0.8B, image | 1 | 598.16 | 563.91 | 5.73% | 1.001 |
| Qwen3.5-0.8B, image | 2 | 853.21 | 800.79 | 6.14% | 0.998 |

Decode is essentially unchanged; this change targets prefill. Six serialized
actual diagnostics preserve complete outputs and confirm the mixed path runs:
392 operations/profile for text, 336 for coder and 835 for VL. Their separate
submissions perturb cache/dispatch behavior, so diagnostic times are not
serving-speed evidence.

## Startup and memory

The initial floating-conversion prototype took about 11.2 s to initialize
text, 9.0 s coder and 13.5 s VL. Integer conversion alone improved these only
modestly. Caching Metal buffer.contents once outside the per-weight loop
removes hundreds of millions of Objective-C getter calls and yields the
final measurements below. Initial/intermediate complete evidence remains
saved, including its source snapshots and matched comparator run.

| Model | Batch | Control init, s | Final init, s | Control RSS, GiB | Final RSS, GiB |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 0.986 | 1.743 | 1.506 | 2.329 |
| Qwen3-0.6B | 2 | 1.014 | 1.748 | 1.549 | 2.373 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 0.789 | 1.401 | 1.263 | 1.931 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 0.780 | 1.406 | 1.263 | 1.932 |
| Qwen3.5-0.8B | 1 | 2.142 | 2.961 | not reported | not reported |
| Qwen3.5-0.8B | 2 | 2.149 | 2.960 | not reported | not reported |

The extra FP16 body copy still costs approximately 0.7–0.8 GiB in the measured
text/coder processes. RSS is individual native-child high water, not aggregate
or unique GPU allocation. Startup still exceeds the raw control; it is not
included in the warm TTFT table. Full first-use TTFT records are preserved
in startup_summary.json and the raw request records.

## Fresh actual vLLM-Metal comparison

The final candidate completed 48 matched text/coder and 24 matched image
requests against the unchanged isolated comparator on the same M4 16 GiB
Mac mini. Processes never overlap. Checkpoints, prompts, image geometry,
greedy generation, capacity 256, batches 1/2 and caps are frozen. Prefix
caching, encoder reuse and speculative decoding remain disabled.

| Model | Batch | Garnet tokens/s per row | vLLM tokens/s per row | Garnet TTFT, ms | vLLM TTFT, ms |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 56.82 | 74.60 | 88.57 | 50.49 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 72.42 | 92.23 | 57.37 | 34.63 |
| Qwen3-0.6B | 2 | 51.73 | 76.02 | 155.14 | 66.44 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 69.96 | 92.58 | 77.95 | 38.27 |
| Qwen3.5-0.8B, image | 1 | 47.10 | 48.66 | 564.91 | 201.97 |
| Qwen3.5-0.8B, image | 2 | 43.07 | 47.48 | 799.72 | 326.65 |

All six profiles still fail both the 2x decode and half-TTFT targets. Text/coder
and image batch 1 preserve exact complete comparator outputs. Image batch 2
retains the existing different/capped comparator caption: native finishes
28 tokens; vLLM reaches the common 32-token cap. That row is a matched-input
measurement with a quality/length caveat. Native FP32 vision/activations/KV
versus comparator BF16 internal arithmetic remains explicit, despite unchanged
BF16 checkpoint weights. The full objective stays active and unmet.

Complete commands/exits, compiler flags, source/model/input/library hashes,
numerical logs, actual outputs, guards, diagnostic attribution and separate
startup/first-use/warm timings are saved here. Raw RGB/patch arrays and compiled
binaries remain private on the Mac. Source snapshots distinguish the initial
floating prototype, the pre-hoist integer version and the final candidate.
Protected GPT-OSS, Pi and original Mac checkouts remain untouched.
