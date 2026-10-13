# Batched recurrent output projections: validated latency improvement

The new optional Qwen3.5 Metal prefill path projects recurrent outputs for
all chunk tokens together. Actual repeated image inference confirms lower
first-token latency with unchanged complete caption tokens. Decode throughput
does not improve consistently. The 2× decode / half-TTFT goal remains unmet.

## Implementation

`qwen35_linear_chunk` flattens independent input projections across batch
and token rows, then runs convolution, recurrent state updates and gated
normalization in causal token order. It gathers the gated token outputs and
performs one output projection for the chunk. Paused rows preserve state;
the existing recurrent split-output optimization remains applicable.

The API opt-in is `decoder.enable_prefill(chunk, 'shared', True)` for
Qwen3.5. The matched frame worker accepts `prefill_linear_chunk: true`.
The flag defaults to false, preserving the prior path as a control.
Chunks 32/64 are available for Qwen3.5; the resident scheduler uses smaller
final chunks to avoid unnecessary padding. For this 79-token image prompt,
plans are 16×5, 32+32+16 or 64+16, each representing 80 physical positions.
No prompt reduction, image reuse, weight quantization or speculation is used.
Checkpoint weights remain BF16 with native FP32 activations and KV.

## Actual native ABBA results

Same M4 Mac mini / 16 GiB, Qwen3.5-0.8B, 288×192 source-image processing,
54 visual tokens, 79 prompt IDs, capacity 256 and output cap 32. Each of the
two independent runs uses candidate/control then control/candidate, three
requests per worker and warm means excluding each worker's first request.
All 144 requests match frozen control prompt IDs, complete token IDs and
finished flags. Competing inference runtimes never run together.

| Run | Batch | Chunk | Control TTFT ms | Candidate TTFT ms | Control tokens/s/row | Candidate tokens/s/row |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Initial | 1 | 16 | 486.51 | 401.21 | 48.17 | 48.17 |
| Confirm | 1 | 16 | 488.92 | 402.19 | 48.05 | 48.29 |
| Initial | 1 | 32 | 488.68 | 369.77 | 47.87 | 48.48 |
| Confirm | 1 | 32 | 490.72 | 368.38 | 48.24 | 48.56 |
| Initial | 1 | 64 | 491.77 | 354.39 | 48.35 | 48.20 |
| Confirm | 1 | 64 | 489.61 | 353.50 | 48.16 | 48.14 |
| Initial | 2 | 16 | 699.26 | 608.92 | 44.24 | 44.37 |
| Confirm | 2 | 16 | 698.38 | 613.20 | 44.36 | 44.09 |
| Initial | 2 | 32 | 699.74 | 597.25 | 44.24 | 42.94 |
| Confirm | 2 | 32 | 702.80 | 582.47 | 44.57 | 44.48 |
| Initial | 2 | 64 | 700.30 | 598.20 | 44.40 | 43.66 |
| Confirm | 2 | 64 | 699.62 | 596.92 | 44.22 | 44.33 |

Batch-1 chunk 64 cuts warm TTFT by about 28% in both runs and retains
roughly 48 tokens/sec/row. Batch-2 chunk 32 cuts warm TTFT by 15–17%,
but its initial decode regression is not reproduced consistently. These
profiles are optional; no batch-2 decode gain is claimed.

The confirmation run's ordinary phase counters show batch-1 chunk-64
prefill GPU time falling from 316.64 to 197.88 ms and CPU encoding from
48.98 to 34.93 ms. Batch-2 chunk-32 GPU time falls from 461.14 to 360.44 ms
and encoding from 48.96 to 38.09 ms. GPU time is inside the wait interval,
not an additive phase. Vision remains approximately 63/123 ms for batches
1/2. These counters identify remaining GPU work; further CPU-only tuning
cannot by itself meet the full target. Serialized per-operation diagnostics
were not enabled for these speed measurements.

First-use batch-1 chunk-16 TTFT falls from approximately 790 to 568 ms;
chunk-64 first use stays around 783 ms, near its control. Batch-2 chunk-16
first use falls from approximately 980 to 763 ms; chunk-32 is around
879 versus 974 ms, while chunk-64 is slower at about 1,041 versus 972 ms.
Filesystem and driver caches were not flushed. Individual child RSS stays
around 3.4 GiB; this excludes aggregate/unique GPU-memory accounting.

## Validation and limits

All eleven handoff checks pass: kernel tests, serving/reset, mixed-image
requests, existing Qwen3.5 layers, text/coder first and last layers, all
three model heads, and small/pretrained vision. Independent NumPy checks
also pass all 24 Qwen3.5 combinations: linear/full attention × batches
1/2 × chunks 2/4/8/16/32/64, three state-carry steps each, paused rows,
partial tails, duplicate/backward slots, rejected controls and reset.
Worst normalized reference error is 0.56435 against the existing threshold
of 1. Explicit and resident executions are exactly equal.

Additional extended dense checks expose a Qwen3 batch-2 chunk-32 key-cache
rotation mismatch: worst ratio 1.51658 exceeds the established tolerance.
The runner stops before the coder's extended checks. The tolerance is
unchanged, and dense text/coder adapters explicitly retain validated
profiles 2/4/8/16. The larger dense profiles are not accepted or advertised.
The failed log and subsequent rejection/state-preservation/output checks
are retained separately. This failure does not invalidate the passing
Qwen3.5 references or its complete caption comparisons.

Candidate library SHA256:
`26f2120c3570a28607969f855f6950d626372b388361325ac49096eb30d60671`.
Preserved control SHA256:
`9bbac5c9ad7486e1fe70690c526bebaed7fcfa06e4d162a3fe257af0a6faddef`.
The candidate is a clean build with exact-half prefill, MPS vision scores
and parallel delta split enabled. Source bundles and every installed file
were SHA256-verified before atomic installation.

The final C++ edits change only indentation and normalize line endings. Subsequent Mac builds
produces the byte-identical confirmed library and passes CTest again;
`final-verification/` preserves both source hashes, commands and binary proof.
After adding the dense profile guard, twelve actual text/coder requests
(both batches) preserve complete saved outputs, and rejected profiles leave
the selected chunk, graph cache and decoder slots unchanged.

`logs/` retains actual commands, process checks, raw records, terminal
results and references. `source-snapshot/` freezes the measured source.
`derived-input-sha256.json` records sizes/hashes for omitted reproducible
RGB and patch bytes, whose full originals remain on the Mac. These are
native comparisons; stock vLLM-Metal uses a different activation/KV dtype,
so these measurements are not an exact-precision vLLM speedup claim.
