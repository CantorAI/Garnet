# Validated vector state slices

The selected native build is `build-vector-slice`, library SHA-256
`fd38413aad05a4946693681ff313810eeb5adcdec5992be66ffcb491612a41a2`. Aligned grouped slices copy four FP32
values per thread using 32-bit indexing. Source widths, starts, counts and row
strides must be divisible by four, and complete input/output addressing must
fit 32 bits; other layouts retain the scalar kernel. Weights, arithmetic,
models, context and outputs are unchanged.

Serialized per-operation diagnostics on all three actual models identified
large recurrent-state slices in Qwen3.5, especially batch 2. Diagnostic
dispatches submit/wait separately and perturb GPU/cache behavior; their
timings are not serving clocks or evidence of an inference speed gain. All
six diagnostic outputs matched the frozen complete native outputs.

## Native control and validation

All 11 native validation jobs passed. A further 192 independent bitwise slice
checks cover aligned/irregular widths, starts and counts, multiple groups,
batch sizes 1/2/5, padded rows, signed zero and varied finite FP32 patterns.
CTest passed on the Mac. The close ABBA comparison executed 48 text/coder
requests and 24 actual image requests, with every prompt/output/finish flag
equal to the selected wide-prefill control. Inference processes never overlap.
Each row averages four warm requests per build/profile; first-use and
initialization measurements remain separate, with filesystem caches unflushed.

| Model | Batch | Control TTFT, ms | Candidate TTFT, ms | Decode ratio |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 103.68 | 104.16 | 0.998 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 71.88 | 72.35 | 1.001 |
| Qwen3-0.6B | 2 | 173.78 | 173.78 | 1.001 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 101.85 | 99.94 | 1.000 |
| Qwen3.5-0.8B, image | 1 | 692.62 | 614.38 | 1.044 |
| Qwen3.5-0.8B, image | 2 | 1134.52 | 912.22 | 1.108 |

Text/coder decode is essentially unchanged. The real-image workload gains
4.4%/10.8% decode and reduces TTFT 11.3%/19.6% for batch 1/2, which justifies
retaining the candidate. Input RGB/patch bytes match; vision is freshly
encoded on each request and both native builds complete the same captions.

## Updated vLLM-Metal comparison

Another 48 text/coder requests and 24 image requests compare the candidate
against the isolated unchanged vLLM-Metal installation, strictly sequentially.
Hardware remains the M4 Mac mini with 16 GiB. Checkpoints, prompts, batch 1/2,
capacity 256, greedy decoding and output caps are frozen. Prefix caching,
encoder reuse and speculation remain disabled.

| Model | Batch | Garnet tokens/s per row | vLLM tokens/s per row | Garnet TTFT, ms | vLLM TTFT, ms |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 56.83 | 74.65 | 104.69 | 49.42 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 72.51 | 92.32 | 72.22 | 34.93 |
| Qwen3-0.6B | 2 | 52.32 | 75.85 | 173.91 | 67.99 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 70.31 | 92.37 | 99.96 | 37.97 |
| Qwen3.5-0.8B, image | 1 | 45.67 | 48.06 | 614.39 | 204.99 |
| Qwen3.5-0.8B, image | 2 | 41.56 | 47.97 | 911.11 | 325.07 |

All six profiles still fail both the 2x decode and half-TTFT objectives.
Text/coder and image batch 1 preserve exact complete comparator outputs.
Image batch 2 retains the existing different/capped comparator-caption
limitation: native completes 28 tokens; vLLM reaches the common 32-token cap.
That row is a matched-input measurement with a quality/length caveat.
The same BF16 checkpoint weights do not imply identical internal arithmetic:
native uses FP32 vision/activations/KV, comparator uses BF16 execution.

Complete commands, exits, hashes, numerical logs, actual output IDs, process
guards, diagnostic attribution and timing summaries are saved here. Private
RGB/patch arrays and compiled binaries remain on the Mac. `candidate-source/`
snapshots the changed source and reproduction helpers. Protected GPT-OSS,
Pi and original Mac checkouts are untouched. The full objective stays active.
Further work should target GPU projection/attention execution and removing
state-copy operations; this incremental gain does not establish 2x.
