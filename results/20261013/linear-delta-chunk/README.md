# Persistent chunk recurrence: verified VL latency gain

`GARNET_METAL_LINEAR_CHUNK_DELTA=ON` retains each 128-key-dimension recurrent
state block in FP32 registers throughout a prompt chunk, emits every token's
core output and writes final state once. Gated normalization then processes
all chunk outputs together. The kernel retains the existing 128-thread /
32-lane reduction structure and per-token causal order, including inactive
rows. A barrier protects shared norm partials between tokens. No checkpoint
values, output limits, prompts, input resolution or precision are changed.

The option defaults OFF and requires chunk convolution and parallel delta
split. Unsupported key/value dimensions retain the previous path. This is
measured with the other retained investigation flags enabled.

## Validation and native confirmation

All 48 recurrence cases match the existing sequential parallel GPU kernel
bit-for-bit for both complete output sequences and final state: batches 1/2,
value dimensions 16/32/128, chunks 1/4/16/64, fresh/carried state and paused
rows/tails. Existing convolution, direct-write and matrix checks also pass.
Compiled test flags are recorded explicitly. Serving, mixed-image and all
36 real-weight linear layer comparisons pass with unchanged tolerances;
resident/explicit execution, reset and rejected-control state checks pass.

Two native ABBA runs match complete captions, finished flags, prompt IDs,
RGB bytes and patch bytes across all 48 requests. The workload remains
288x192 / 54 visual / 79 prompt / 28 completed output tokens, cap 32, batch
1 chunk 64 or batch 2 chunk 32 with adaptive tails. Four warm samples per
policy/profile are averaged in each run; runtimes exit between workers.

| Batch | Initial TTFT reduction | Confirmation control ms | Confirmation candidate ms | Confirmation reduction | Decode ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 24.47% | 293.94 | 225.44 | 23.30% | 1.008 |
| 2 | 21.32% | 488.98 | 383.71 | 21.53% | 1.001 |

This is a prefill change; the small decode variation is not a claimed gain.

## Fresh matched FP32 VL comparison

Same M4 / 16 GiB Mac mini, learned values, FP32 operations/activations/KV,
images, prompts, stop rules, caps and batches. Garnet stores original BF16
weights; vLLM-Metal uses verified lossless FP32-expanded copies. Exact-value
proofs and asset hashes are checked before execution. This remains the
additional FP32 study, `counts_as_default_bf16_speedup=false`; stock BF16
results stay separate. Prefix reuse/speculation are disabled, every frame
is reencoded and encoder identities are unique. All 24 requests preserve
complete output IDs and finished flags. Runtimes run sequentially.

| Batch | Garnet tokens/s/sequence | vLLM tokens/s/sequence | Decode ratio | Garnet TTFT ms | vLLM TTFT ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 48.28 | 25.89 | 1.86 | 226.41 | 258.25 |
| 2 | 44.69 | 16.91 | 2.64 | 384.44 | 437.15 |

First-token latency is now about 12% lower than the comparator, but neither
half-TTFT target is met. Batch-1 decode remains below 2x. The complete
three-model goal remains active; these fixed-image short prompts do not
establish every context length or batch profile.

## Text/coder behavior in the shared build

A further native ABBA compares the same chunk-32 text/coder policies with
the retained dense-only candidate and preserves all 48 complete requests.
Warm decode ratios range 0.996-1.007; TTFT ratios range 1.003-1.015. No dense
performance gain is claimed. First-profile initialization is higher in this
run: text batch 1 averages 2.52 vs 1.66 seconds and coder batch 1 2.11 vs
1.40 seconds. Batch-2 initialization is near its control. This startup cost
is retained as measured evidence rather than attributed to an unverified
cause. Filesystem and driver caches were not flushed.

## Evidence and process isolation

Candidate library SHA256:
`e727c3c81e7af9f252bbd090ea506e6db6c2ffee021f2edf6e167e380d9ea5e8`.
Native VL control is the preserved convolution build
`230f79802b3a5577110c59f70e05b8e1a211e9b4ff7150ef6c6ebd16d7cb8cc4`.
Dense control remains
`c0917b8369f89dde5b7688eb3df800905822aba52b1d0d5562f322f5cab2f36d`.

The first preflight stopped before source installation because a restarted
WorldSense supervisor introduced a forwarded TRT-budget setting name.
Inspection and 21 guard checks establish the pinned source and absence of
loaded inference libraries. The retry proceeds without stopping or changing
WorldSense/Cantor services. The exact source/library checks remain active.
That guard change and its evidence are committed separately.

`completed/` preserves source, flags, commands, terminal statuses, tests,
complete captions, cold/warm timing, process/resource samples and guard
recovery. Tensor/RGB copies remain private on the Mac, with sizes/hashes in
`evidence-proof.json`. Clean archive SHA256:
`21ba7e849969202e56150d869f4b957ba002ed352243f229691f1fec9c395c58`.
CPU preprocessing is included in matched TTFT and total latency.

## Matched startup intervals

Excluded from warm means.

| Batch | Runtime | Initialization s | First-use TTFT ms |
| --- | --- | ---: | ---: |
| 1 | garnet | 3.96 | 438.48 |
| 1 | vllm-metal | 34.04 | 312.01 |
| 2 | garnet | 5.07 | 585.35 |
| 2 | vllm-metal | 33.07 | 535.35 |
