# Rejected decode dot-loop unrolling

Four variants explicitly unroll two/four original input iterations with
64/128 threads and four outputs per SIMD group. They preserve the original
BF16 bits, FP32 arithmetic, per-lane order and reduction. Scalar irregular
tails keep the old loop. This remains a component experiment; serving
source and selected binaries are unchanged.

Both runs pass all 544 bit-for-bit checks across real first-layer matrices,
complete vocabulary heads, batches 1/2, irregular shapes and output padding.
Each timing uses two ABBA rounds with three warm-ups and twelve samples
per position. These are isolated matrix timings, not model throughput.

| Model head | Batch | Variant | Initial GPU ratio | Repeat GPU ratio |
| --- | ---: | --- | ---: | ---: |
| Qwen3-0.6B | 1 | bf16_decode_u2_t64 | 0.993 | 1.007 |
| Qwen3-0.6B | 1 | bf16_decode_u2_t128 | 1.011 | 0.974 |
| Qwen3-0.6B | 1 | bf16_decode_u4_t64 | 1.001 | 0.997 |
| Qwen3-0.6B | 1 | bf16_decode_u4_t128 | 1.005 | 0.997 |
| Qwen3-0.6B | 2 | bf16_decode_u2_t64 | 0.997 | 0.994 |
| Qwen3-0.6B | 2 | bf16_decode_u2_t128 | 1.030 | 1.000 |
| Qwen3-0.6B | 2 | bf16_decode_u4_t64 | 0.978 | 1.001 |
| Qwen3-0.6B | 2 | bf16_decode_u4_t128 | 1.007 | 1.001 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | bf16_decode_u2_t64 | 1.012 | 1.039 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | bf16_decode_u2_t128 | 1.000 | 0.963 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | bf16_decode_u4_t64 | 0.993 | 0.990 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | bf16_decode_u4_t128 | 0.990 | 1.012 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | bf16_decode_u2_t64 | 0.991 | 0.992 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | bf16_decode_u2_t128 | 0.999 | 0.999 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | bf16_decode_u4_t64 | 0.990 | 0.997 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | bf16_decode_u4_t128 | 1.013 | 1.002 |
| Qwen3.5-0.8B | 1 | bf16_decode_u2_t64 | 1.007 | 1.032 |
| Qwen3.5-0.8B | 1 | bf16_decode_u2_t128 | 1.026 | 0.992 |
| Qwen3.5-0.8B | 1 | bf16_decode_u4_t64 | 1.014 | 0.988 |
| Qwen3.5-0.8B | 1 | bf16_decode_u4_t128 | 1.022 | 0.996 |
| Qwen3.5-0.8B | 2 | bf16_decode_u2_t64 | 0.996 | 0.991 |
| Qwen3.5-0.8B | 2 | bf16_decode_u2_t128 | 1.007 | 0.987 |
| Qwen3.5-0.8B | 2 | bf16_decode_u4_t64 | 1.000 | 0.983 |
| Qwen3.5-0.8B | 2 | bf16_decode_u4_t128 | 1.039 | 0.978 |

The initial VL head gains did not repeat across both batches. Other small
gains fluctuate by variant/profile; no consistent broad gain supports a
full-model promotion. All variants are rejected for serving; no vLLM or
2x/half-TTFT claim is made. The full goal remains active and unmet.

Actual commands, terminal statuses, source/generated shaders, matrix hashes,
timing windows, bit checks and process guards are in completed/. Private
matrix bytes and binaries remain on the Mac with hashes in evidence-proof.json.

Archive SHA256: f2a5ca4822a3e7b08146ef158bdf74cf7f862c40275ef81beedc643fa8f840c9.
