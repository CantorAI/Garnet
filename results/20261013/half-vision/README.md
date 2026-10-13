# Exact-scaled half vision projections: rejected

The candidate builds and passes all eleven handoff validation jobs, including
real-model vision references and generation. It does not improve real-frame
latency or decode speed. Its four production-source changes were reverted.
The selected KV-chunk-vector library remains unchanged:
`564b515ab6b0eb68413ef5a54de6beff47d333f8f1fa6a1188562bf6e5c28ec4`.

The experiment extends the existing lossless per-output power-of-two BF16 to
FP16 storage representation to vision-block and merger matrices, using MPS
with FP32 inputs, accumulation and outputs plus inverse scale. It excludes
patch embedding, positional embeddings, norms and biases. This retains the
checkpoint values; it is not reduced-precision weight quantization.

## Actual native ABBA comparison

Same M4 16 GiB Mac mini, Qwen3.5-0.8B checkpoint, image bytes, preprocessing,
288×192 input, 54 visual tokens, 79 prompt tokens, 32-token cap, chunk 16.
Each persistent worker re-encodes every image, resets decoder state, and closes
cleanly before the next worker starts. Twelve requests per batch (24 total),
including first-use requests; warm means below exclude the first request in
each fresh process. All generated token IDs match the selected native control
and all requests finish. These are native comparisons, not new vLLM results.

| Batch | Control TTFT | Candidate TTFT | Control tokens/s/row | Candidate tokens/s/row |
| --- | ---: | ---: | ---: | ---: |
| 1 | 562.67 ms | 568.41 ms | 47.19 | 47.07 |
| 2 | 794.41 ms | 804.30 ms | 43.09 | 43.07 |

Warm TTFT worsens 1.02% and 1.24%; decode changes −0.27% and −0.06%.
No useful gain is demonstrated, so no additional confirmation or comparator
speed claim is warranted.

The separate serialized operation diagnostic executes 885 exact-half
projections, including actual 432-row vision operations. It matches the
complete frozen native output. Its altered dispatch/cache behavior prevents
using its GPU attribution as serving timing. Thus the failed speed result
cannot be attributed to an unexecuted candidate path.

## Evidence

`candidate-source/` preserves the experimental source, build flags, helpers,
and candidate/control library hashes before reverting. Each job directory
preserves actual commands, terminal exits, validation output and process
guards. `native-frame-ab-half-vision-20261013/jobs/native_frame_summary.json`
contains timings, complete token IDs and identical RGB/patch hashes.
`half-vision-profile-20261013/jobs/summary.json` contains diagnostic attribution.
Raw image buffers are excluded; their hashes remain in the evidence.

The overall goal remains unmet: same hardware and model, at least twice
vLLM-Metal decode throughput and at most half its TTFT. Existing native FP32
activation/KV arithmetic differs from the comparator's BF16 internal arithmetic;
matching checkpoint bytes alone does not establish identical precision.
