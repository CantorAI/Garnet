# Exact scaled FP16 storage feasibility

This evidence establishes compatibility and exact weight-storage feasibility,
not a model-inference result or speed improvement. `build-delta-split` stays
selected; no production projection kernel has changed in this experiment.

## Actual MPS compatibility on the M4

The SDK defines MPSDataTypeBFloat16, but MPSMatrixMultiplication rejects it
on this macOS 15.6.1 M4. Three independent BF16 cases abort with signal 6:
FP32 activation/BF16 weight/FP32 output, BF16 activation/weight with FP32
output, and all-BF16 storage. The FP32 control succeeds at 3x7x5 and
16x1024x64. Raw assertion messages and individual nonzero child statuses
remain saved; the runner exit zero means the compatibility survey completed,
not that BF16 multiplication succeeded.

The mixed-input assertion explicitly permits FP32 A/C with FP16 B. A second
GPU probe actually executes that combination, with BF16-derived toy weights
multiplied by 2^16 before exact FP16 storage and output rescaled by 2^-16.
Every toy weight round-trips bitwise to its original BF16-expanded value.
Both GPU shapes pass against independent FP64 dot products. Worst absolute
errors are 4.386e-9 and 2.456e-7, the same reported errors as the FP32 control.
Activation and output storage remain FP32. This does not establish timing
or numerical behavior for an actual Qwen graph.

## Exhaustive checkpoint survey

A CPU-only scan checks every rank-two BF16 tensor in each pinned checkpoint,
including embeddings and vision, rather than only selected runtime projections.
Each row gets a power-of-two scale based on its maximum absolute value.
After FP16 conversion and inverse scaling, all recovered FP32 bit patterns
are compared with the original BF16-expanded bits, including signed zero.
Processing 1,024 rows per window bounds temporary memory. Checkpoints remain
unchanged; this is a proposed storage encoding, not permission to round values.

| Checkpoint | Matrices | Values checked | Nonexact values | Nonexact rows |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 198 | 751,566,848 | 2 | 2 |
| Qwen2.5-Coder-0.5B-Instruct | 169 | 493,961,216 | 0 | 0 |
| Qwen3.5-0.8B | 246 | 871,628,800 | 0 | 0 |

The Qwen3 exceptions are:

- `lm_head.weight`: 1 values in 1 rows; coordinates/original/recovered bits are in its JSON report.
- `model.embed_tokens.weight`: 1 values in 1 rows; coordinates/original/recovered bits are in its JSON report.

A production candidate must preserve these exceptional values through an
exact fallback. It must also apply the inverse per-output-column scales on
the GPU and validate FP32 range/rounding behavior, actual layer references,
complete generated outputs, initialization/memory costs and sequential
end-to-end matched performance. BF16 values that do not round-trip must never
be silently replaced with an approximate FP16 value.

The next experiment is an optional exact-scaled FP16 MPS prefill path, keeping
raw BF16 decode and falling back for incompatible matrices. No 2x/half-TTFT
target is achieved here. Compiled probes and raw checkpoint data remain
private on the Mac; commands, source hashes, complete compatibility failures,
CPU survey reports and numerical logs are retained in this folder.
