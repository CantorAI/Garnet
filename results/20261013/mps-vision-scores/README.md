# Optional FP32 MPS vision attention: validated modest latency gain

Packing query/key heads into bounded FP32 scratch and computing batched
`Q * K^T` with MPS reduces warm real-frame TTFT. It retains the existing
softmax implementation, checkpoint, input bytes, decoder and output cap.
Two native ABBA runs produce identical complete generated tokens. This
improvement is retained as `GARNET_METAL_MPS_VISION_SCORES`, OFF by default.
The 2× vLLM-Metal throughput / half-TTFT goal is still unmet in every profile.

The MPS path handles at most 1,024 flattened image rows, 512 patch tokens,
16 heads and head dimension 128. Other shapes retain the existing score
kernel. Scratch is cached by rows, tokens, hidden size and heads, reused across
vision blocks in the graph, and released with its context. Heads/batches have
separate matrices. No learned weights are changed or quantized.

## Actual native measurements

Same Apple M4 Mac mini, 16 GiB, Qwen3.5-0.8B BF16 checkpoint, 288×192 image,
54 visual tokens, 79 prompt IDs, chunk 16, capacity 256, 32-token output cap.
Every frame is freshly preprocessed and encoded; decoder state is reset.
Only one inference process runs at a time. Each ABBA run contains twelve
requests per batch, four of which are first-use requests. Warm means exclude
the first request in each fresh process.

| Run | Batch | Control TTFT | Candidate TTFT | TTFT reduction | Control tokens/s/row | Candidate tokens/s/row |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial | 1 | 563.48 ms | 547.55 ms | 2.83% | 47.14 | 47.04 |
| Initial | 2 | 796.18 ms | 762.26 ms | 4.26% | 43.07 | 42.98 |
| Repeat | 1 | 563.64 ms | 551.43 ms | 2.17% | 47.23 | 47.08 |
| Repeat | 2 | 792.48 ms | 765.85 ms | 3.36% | 43.04 | 43.09 |

All 48 native frame requests match complete control token IDs and finish.
Decode is essentially unchanged; no meaningful decode speedup is claimed.

Repeat first-use means, separate from warm results:

| Batch | Control model initialization | Candidate initialization | Control first TTFT | Candidate first TTFT |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.897 s | 2.906 s | 859.45 ms | 864.38 ms |
| 2 | 2.927 s | 2.895 s | 1,059.40 ms | 1,049.73 ms |

These are fresh processes with existing filesystem/shader caches, not cold
boots. Memory evidence remains individual-process high-water RSS, not unique
unified GPU memory; all raw process accounting is retained.

## Fresh sequential vLLM-Metal comparison

The isolated comparator and all three checkpoint paths are unchanged from
the prior investigation. Prompts, token caps, batch, image bytes and native
prompt IDs match. Prefix reuse, image encoder reuse and speculation are
disabled. These are observed same-checkpoint results: native FP32 vision,
activations and KV differ from the comparator's BF16 internal arithmetic.
That precision difference remains an acceptance limitation.

| Model | Batch | Garnet tokens/s/row | vLLM tokens/s/row | Garnet TTFT | vLLM TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 56.86 | 73.99 | 70.85 ms | 51.04 ms |
| Qwen3-0.6B | 2 | 51.99 | 75.78 | 116.94 ms | 64.13 ms |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 72.27 | 92.11 | 55.42 ms | 34.74 ms |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 69.83 | 92.24 | 74.50 ms | 38.02 ms |
| Qwen3.5-0.8B image | 1 | 47.06 | 48.19 | 548.54 ms | 205.49 ms |
| Qwen3.5-0.8B image | 2 | 42.99 | 48.31 | 763.94 ms | 327.28 ms |

Text/coder and image batch 1 have exact generated token IDs and complete
outputs. Image batch 2 still differs: native completes its 28-token caption,
while the comparator reaches the 32-token cap. Its timing is diagnostic and
cannot establish a successful matched-output speed comparison. Text/coder
are the existing short prompts (8 and 17 generated tokens including stop),
so their throughput should not be generalized to every workload.

Every decode ratio is below 2 and every TTFT fraction exceeds 0.5. All target
flags remain false. No current result supports a 2× claim.

## Validation and reproduction

All eleven handoff jobs pass: CTest, serving/reset/shared-state checks,
mixed-image pipeline, actual Qwen3.5 and dense decoder layer references,
all three complete vocabulary heads and small/pretrained vision references.
CTest includes 48 new FP64 softmax oracle cases at 4/24/216 patch tokens,
batches 1/2, heads 2/12, dimensions 8/64, changed inputs and abort recovery.
The default OFF build also compiles and passes CTest.

`candidate-source/` contains source, build/launch helpers, exact compiler
options, CTest detail and library hashes. The tested build sets Metal,
Metal xModel, exact-half prefill and MPS vision scores ON. Each job directory
records actual commands, exits, versions/hashes, process guards and results.
Summaries retain all token IDs and RGB/patch hashes; raw pixel buffers are
excluded. The selected prior control binary remains unchanged:
`564b515ab6b0eb68413ef5a54de6beff47d333f8f1fa6a1188562bf6e5c28ec4`.
The tested MPS-score library is
`c9f8c76aa2657669d6a7b10eba459481ac797fc64904d25360933b565f9d77d1`.
