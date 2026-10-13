# Chunk convolution and preparation: verified VL prefill gain

The optional `GARNET_METAL_LINEAR_CHUNK_CONV=ON` path processes causal
convolution across a prompt chunk, writes final convolution state once and
prepares all tokens together. Each channel retains its original token order,
FP32 sum/activation and inactive-row behavior. Recurrent delta updates remain
sequential. Kernels longer than eight elements retain the existing path.
The option defaults OFF and is measured with the retained direct-write path.

## Correctness and native repeat

All 48 convolution cases are bit-identical to the existing sequential GPU
kernel for both complete outputs and final state: batches 1/2, padded/regular
channels, kernels 1/4/8 and chunks 1/4/16/64, including pause/resume and
inactive tails. Compile flags recorded in `source-binary-proof.json` confirm
this test block is enabled. Kernel, serving and mixed-image checks pass.
All 36 real Qwen3.5 linear layer comparisons pass at batches 1/2 and chunks
2/4/8/16/32/64, with three state-carry steps and existing tolerances.

Two native ABBA runs preserve complete captions, finished flags, prompt IDs,
RGB bytes and patch bytes across all 48 requests. Images, prompt, stop rules
and cap remain the previous 288x192 / 54 visual / 79 prompt / 28 output-token
workload. Batch 1 uses chunk 64 and batch 2 chunk 32, with adaptive tails.
Every runtime exits before the next starts. Means use four warm requests per
policy/profile per ABBA run; first use and initialization remain separate.

| Batch | Initial TTFT reduction | Confirmation control ms | Confirmation candidate ms | Confirmation reduction | Decode ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 11.52% | 339.76 | 297.96 | 12.30% | 0.998 |
| 2 | 13.33% | 566.82 | 490.16 | 13.52% | 1.000 |

No decode gain is claimed from this prefill change.

## Fresh matched FP32 comparison

The same M4 / 16 GiB Mac mini, learned values, FP32 operations/activations/KV,
images, prompt IDs, output limits and batches are used. Garnet retains original
BF16 weight storage; vLLM-Metal uses verified lossless FP32-expanded copies.
Checkpoint equivalence and asset hashes are checked before execution. This
is the additional FP32 study, with `counts_as_default_bf16_speedup=false`;
stock BF16 results remain separate. Prefix reuse and speculation are disabled,
encoder identities are unique and all frames are reencoded.

All 24 matched requests preserve complete outputs and finished flags.

| Batch | Garnet tokens/s/sequence | vLLM tokens/s/sequence | Decode ratio | Garnet TTFT ms | vLLM TTFT ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 47.86 | 26.78 | 1.79 | 300.90 | 258.43 |
| 2 | 44.62 | 16.80 | 2.66 | 490.04 | 438.36 |

Neither half-TTFT target is met; batch-1 decode also remains below 2x. The
full three-model goal stays active. These results cover the measured short
fixed-image prompts and batches, not every possible workload.

## Evidence

Candidate library SHA256:
`230f79802b3a5577110c59f70e05b8e1a211e9b4ff7150ef6c6ebd16d7cb8cc4`.
Native control is the preserved direct-write library
`b97250a0868da1424575f37405aa0e0f1f84383cd06185e66f574d80b58262d1`.

`completed/` contains the actual source, flags, commands, terminal statuses,
model checks, complete captions, warm/first-use/initialization timings,
resource samples and process guards. Tensor/RGB copies remain private on
the Mac; sizes and hashes are in `evidence-proof.json`. Archive SHA256:
`0c1fb5860fb135ba318c34a2ae043cb9a12371e9167fc7e27bc0bce1afdace34`.
CPU preprocessing is included in matched latency. Filesystem/driver caches
were not flushed. No source or binary from another checkout was changed.

## Matched startup intervals

Excluded from warm means.

| Batch | Runtime | Initialization s | First-use TTFT ms |
| --- | --- | ---: | ---: |
| 1 | garnet | 3.99 | 670.75 |
| 1 | vllm-metal | 34.02 | 308.00 |
| 2 | garnet | 5.08 | 759.94 |
| 2 | vllm-metal | 33.25 | 512.70 |
