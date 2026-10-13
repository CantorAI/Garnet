# Rejected interleaved BF16 decode layout

This candidate preserves all learned BF16 bits and the original per-output
FP32 accumulation/reduction order. Four output columns by four K terms are
stored in a contiguous 32-byte block. The GPU loads two `uint4` vectors and
reconstructs the original FP32 values. Odd K retains the original kernel.
The raw BF16 buffer remains available for prefill, fallback and control.

The candidate compiles with the optional `GARNET_METAL_PACKED_DECODE` flag.
All eleven handoff checks pass, including actual model layers/heads, serving,
mixed images and vision. Seventy-two direct GPU comparisons are bit-exact
against the original kernel, covering batches 1/2, aligned and odd K,
column tails, signed zero and small BF16 values. Actual native ABBA inference
then completes 48 text/coder requests and 24 image requests with identical
prompt IDs, complete output token IDs and finished flags.

## Actual whole-model results

Same M4 Mac mini / 16 GiB, checkpoints, prompt IDs, caps and batches. Only
one inference runtime runs at once. Candidate/control then control/candidate
workers each make three requests; warm means exclude each worker's first
request. These are native comparisons, not vLLM speed claims.

| Model | Batch | Control tokens/s/row | Candidate tokens/s/row | Decode ratio | Control TTFT ms | Candidate TTFT ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 47.51 | 48.68 | 1.025 | 89.39 | 89.53 |
| Qwen3-0.6B | 2 | 47.77 | 48.30 | 1.011 | 133.86 | 131.62 |
| Qwen2.5-Coder-0.5B | 1 | 65.66 | 66.18 | 1.008 | 71.24 | 72.13 |
| Qwen2.5-Coder-0.5B | 2 | 70.11 | 69.34 | 0.989 | 75.10 | 78.48 |
| Qwen3.5-0.8B VL | 1 | 48.44 | 49.57 | 1.023 | 353.98 | 359.55 |
| Qwen3.5-0.8B VL | 2 | 44.67 | 45.99 | 1.030 | 606.70 | 607.58 |

The VLM uses the previously validated recurrent chunk path: chunk 64 with
an adaptive 16-token tail for batch 1, and chunk 16 for batch 2. Both builds
freshly encode the same 288×192 image per request, with 79 prompt IDs and
cap 32. Text/coder use chunk 16 and cap 64. No reduced prompt, image reuse,
prefix caching or speculative decoding is introduced.

Decode changes are small and mixed, while most TTFT profiles are flat or
worse. Peak individual child RSS increases from about 2.33 to 3.44 GiB
for text, 1.93 to 2.85 GiB for coder, and 3.35 to 4.75 GiB for VL batch 1.
These are high-water child RSS values, not aggregate or unique GPU memory.
Batch-1 initialization is 2.09 versus 1.94 seconds for text, 1.70 versus
1.41 seconds for coder, and 3.39 versus 3.10 seconds for VL. First-use VL
TTFT is 848 versus 777 ms. Filesystem and driver caches were not flushed.

The broad layout is rejected: it adds approximately 0.9–1.4 GiB and load
work without a useful consistent improvement across the requested models
and latency/throughput metrics. There is no confirmation or new matched
vLLM comparison of this rejected candidate. The 2× decode / half-TTFT goal
remains active and unmet.

## Environment and restoration

Recorded checkpoint/config/tokenizer assets match the prior control's
metadata. Absolute text timings differ from earlier sessions; desktop and
streaming services are present in process snapshots, so the paired control
in this run is the reference. A post-run query records no thermal/performance
warning, and swap usage is 3,109.25 MiB, equal to the observed preflight value.
This does not prove the absence of all background contention.

After the benchmarks, an unrelated release-test xlang3 process cannot be
verified by the existing ancestor-based inference guard. The rejected
restoration preflight and its evidence are retained. No further inference
was started, the guard was not relaxed, and no unrelated job was stopped.
Source-only restoration verifies all owned experiment jobs are terminal,
checks the current candidate and restored file hashes, and touches only
the four investigation source files and its created probe script.

The selected library remains byte-identical:
`26f2120c3570a28607969f855f6950d626372b388361325ac49096eb30d60671`.
The rejected private candidate remains isolated:
`6bed77dcbd5dbfe1e6bb875062bce55d42db5e819f04badaeafe701694a55ba9`.
Production source is restored locally and on the Mac. `restoration/`
preserves that proof; `source-snapshot/` and `rejected-packed-decode.diff`
preserve the actual experimental implementation. `logs/` retains exact
commands, terminal results, outputs, process snapshots and numerical checks.
Reproducible RGB/patch bytes are omitted from Git with their sizes/SHA256
retained; full originals and compiled candidates remain private on the Mac.

The later read-only launcher audit resolves the release-test service's identity
and adds exact fingerprint/library checks to the comparison guard. That
separate change and its eighteen passing checks are documented in
`../worldsense-inline-guard/`; the rejected layout stays reverted.
