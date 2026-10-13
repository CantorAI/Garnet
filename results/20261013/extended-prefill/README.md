# Extended prefill profiles: actual tests, no 2× claim

Same M4 Mac mini, Qwen3.5-0.8B checkpoint, 288×192 image, 79 prompt
tokens, capacity 256, output cap 32 and fresh image encoding per request.
Each profile uses sequential candidate/control then control/candidate
workers, with three requests per worker. Warm means exclude each worker's
first request. Complete prompt IDs, generated token IDs and finished flags
match the frozen selected control for all 48 requests in each experiment.

The fixed 32/64 profiles pass all eleven handoff jobs and independent
NumPy references for actual linear/full-attention layers, batches 1/2 and
chunks 32/64, including paused rows, partial tails and state carry. The
worst normalized reference error is below 1 at the established tolerance.

## Fixed profiles: rejected

| Batch | Chunk | Control TTFT ms | Candidate TTFT ms | Candidate/control decode ratio |
| --- | --- | ---: | ---: | ---: |
| 1 | 32 | 493.85 | 507.40 | 0.995 |
| 1 | 64 | 488.82 | 597.73 | 0.999 |
| 2 | 32 | 696.99 | 741.26 | 1.005 |
| 2 | 64 | 700.40 | 938.41 | 1.003 |

Padding expands 79 tokens to 96/128 rather than the control's 80. These
profiles worsen warm and first-use latency and are not selected.

## Adaptive final chunks: candidate only

The candidate schedules 32+32+16 or 64+16, retaining 80 physical token
positions. Default profiles of 16 or fewer keep their original schedule.

| Batch | Maximum chunk | Control TTFT ms | Candidate TTFT ms | Candidate/control decode ratio |
| --- | --- | ---: | ---: | ---: |
| 1 | 32 | 488.90 | 451.94 | 0.982 |
| 1 | 64 | 490.68 | 437.90 | 0.963 |
| 2 | 32 | 696.36 | 646.26 | 1.015 |
| 2 | 64 | 696.88 | 683.59 | 1.007 |

First-use candidate TTFT is 1.21/1.65 seconds for batch 1 and 1.39/1.89
seconds for batch 2, versus roughly 0.78/0.97 seconds for the controls.
Multiple graph profiles add first-use preparation. Filesystem and driver
caches were not flushed. Peak individual child RSS remains around 3.4 GiB;
it is not a measure of aggregate or unique GPU memory.

These are modest warm latency gains, with inconsistent decode behavior.
The selected control binary is preserved. Neither experiment demonstrates
the full 2× decode / half-TTFT goal or a new vLLM-Metal advantage.

## Evidence and execution limits

`fixed/` preserves exact commands, terminal results, source snapshots,
raw timing/output records and independent reference logs. Reproducible
RGB and patch bytes are omitted from Git; their individual sizes and SHA256
hashes are retained in `derived-input-sha256.json`. Full bytes remain in the
private Mac evidence directory. Source uploads were checked by bundle and
per-file hashes before atomic installation, and the candidate build was
cleanly rebuilt.

The first fixed-profile benchmark launch failed at the owned-predecessor
dependency gate before inference. Its failed log is preserved. The gate
now rechecks the authoritative terminal file after probing the PID, closing
a completion/publication race while retaining the process identity check
and failing if a predecessor disappears without publishing a result.
