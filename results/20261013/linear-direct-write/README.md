# Recurrent token output writes: repeatable VL latency gain

The optional `GARNET_METAL_LINEAR_CHUNK_DIRECT_WRITE=ON` path writes each
gated token output into its final sequence segment. It removes the growing
intermediate concatenations, retaining the same FP32 normalization/gate
arithmetic, original checkpoint values, causal state updates and active masks.
The option defaults OFF. Decoder profiles and output limits do not change.

## Validation and repeat native measurements

Six kernel layouts compare direct writes bit-for-bit with independently
computed compact outputs, including reversed writer order, repeated requests,
two batch rows and padded dimensions. Out-of-range writes are rejected.
Kernel checks, serving/reset and mixed-image checks pass. All 36 real-weight
linear layer comparisons pass: batches 1/2, chunks 2/4/8/16/32/64, and three
state-carry steps with paused rows and partial tails. Rejected controls leave
state unchanged, reset clears state and resident/explicit outputs agree.

Two sequential native ABBA runs match every complete caption, stop flag,
prompt ID, RGB byte and patch byte across all 48 requests. Each policy/batch
has two fresh workers with three requests each; warm means exclude the first
request. Batch 1 uses chunk 64, batch 2 chunk 32, with adaptive 16-token tails
and the existing recurrent chunk projection enabled. Images are 288x192,
with 54 visual and 79 total prompt tokens; cap 32 produces complete 28-token
captions. No prefix reuse or speculation is used.

| Batch | Initial TTFT reduction | Confirmation control ms | Confirmation candidate ms | Confirmation reduction | Confirmation decode ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.28% | 355.20 | 341.34 | 3.90% | 1.005 |
| 2 | 4.39% | 588.70 | 564.96 | 4.03% | 0.995 |

Decode differences are small and inconsistent; no decode gain is claimed.

## Fresh matched FP32 vLLM-Metal comparison

Same M4 / 16 GiB Mac mini, learned values, arithmetic precision, input images,
prompt IDs, output cap and batches. Garnet uses original BF16 weight storage
and FP32 operations; the comparator uses verified lossless FP32-expanded
checkpoints with FP32 parameters/activations/KV. Exact-value proofs and asset
hashes are checked before execution. This remains the additional FP32 study,
`counts_as_default_bf16_speedup=false`, with stock BF16 results separate.

All 24 matched requests preserve complete output IDs and finished flags.
Encoder identities are unique per row/frame and every image is reencoded.
Runtimes run one at a time and shut down between workers.

| Batch | Garnet tokens/s/sequence | vLLM tokens/s/sequence | Decode ratio | Garnet TTFT ms | vLLM TTFT ms | Half-TTFT met |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 47.81 | 26.74 | 1.79 | 346.11 | 257.57 | No |
| 2 | 44.77 | 16.60 | 2.70 | 563.90 | 437.89 | No |

The full three-model 2x decode / half-TTFT goal remains unmet. These are
short, fixed-image workloads; longer contexts and other batch profiles are
not established by this result. CPU preprocessing is included in matched
TTFT and total latency. Filesystem and driver caches were not flushed.

## Evidence

Candidate library SHA256:
`b97250a0868da1424575f37405aa0e0f1f84383cd06185e66f574d80b58262d1`.
Native control is the retained dense candidate:
`c0917b8369f89dde5b7688eb3df800905822aba52b1d0d5562f322f5cab2f36d`.
Older selected binaries are preserved.

`completed/` records actual source, flags, commands, terminal results,
numerical checks, complete captions, warm/first-use/initialization intervals,
resource samples and guarded process snapshots. RGB/tensor copies remain
private on the Mac; their sizes and SHA256 values are retained in
`evidence-proof.json`. The cleaned archive SHA256 is
`ab9a0a85cfb293ca3bd57879bc37231f192ebbf9930f463e319d7d3f672ff3b6`.

The first build failed because the new test used an unqualified `MetalMatrix`
type. No speed test ran from that failure. `compile-failure/` preserves its
source and terminal failure. Retry 1 qualifies `Garnet::MetalMatrix` and passes
the checks above. The source archive records the tested header; the working
header adds only a two-line caller contract comment with the same prototype.

## Startup intervals in the matched comparison

These are measured separately from warm request means.

| Batch | Runtime | Initialization s | First-use TTFT ms |
| --- | --- | ---: | ---: |
| 1 | garnet | 4.02 | 803.19 |
| 1 | vllm-metal | 33.58 | 319.16 |
| 2 | garnet | 5.09 | 854.57 |
| 2 | vllm-metal | 33.40 | 514.17 |
