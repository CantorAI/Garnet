# Rejected wider-output decode kernels

Four candidates use eight/sixteen outputs per SIMD group with 64/128 threads.
The original control uses four outputs and 128 threads. All preserve original
BF16 bits, FP32 input/accumulation/output, per-lane dot-product order and
SIMD reductions. Serving code and selected libraries are unchanged.

All 272 actual component checks are bit-identical, including irregular shapes,
input/output strides, padding and full vocabulary heads for all three models,
at batches 1/2. Each profile uses two ABBA rounds, three warm-ups and twelve
samples per position. These are matrix-component GPU timings, not model
decode or vLLM speedups. No claim is made about an unmeasured full model.

| Model | Outputs / threads | Pooled large-matrix GPU ratio |
| --- | --- | ---: |
| Qwen3-0.6B | bf16_decode_c8_t64 | 0.996 |
| Qwen3-0.6B | bf16_decode_c8_t128 | 0.988 |
| Qwen3-0.6B | bf16_decode_c16_t64 | 0.798 |
| Qwen3-0.6B | bf16_decode_c16_t128 | 0.797 |
| Qwen2.5-Coder-0.5B-Instruct | bf16_decode_c8_t64 | 0.970 |
| Qwen2.5-Coder-0.5B-Instruct | bf16_decode_c8_t128 | 0.981 |
| Qwen2.5-Coder-0.5B-Instruct | bf16_decode_c16_t64 | 0.810 |
| Qwen2.5-Coder-0.5B-Instruct | bf16_decode_c16_t128 | 0.805 |
| Qwen3.5-0.8B | bf16_decode_c8_t64 | 1.002 |
| Qwen3.5-0.8B | bf16_decode_c8_t128 | 0.975 |
| Qwen3.5-0.8B | bf16_decode_c16_t64 | 0.804 |
| Qwen3.5-0.8B | bf16_decode_c16_t128 | 0.792 |

The table pools only the collected matrices with more than 256 outputs, once
per collected matrix/batch; it is not weighted by full-model layer counts.
Ratios below one are slower. Eight outputs shows no useful broad gain;
sixteen is materially slower. No candidate was promoted or repeated merely
to search for a favorable sample. The selected small-output policy is unchanged.

Actual commands, statuses, generated shader, original source, raw timing and
bit-check records, matrix hashes and process guards are in evidence/.
Private matrix bytes and the probe binary remain on the Mac with hashes in
evidence-proof.json. Archive SHA256: 0b083e817295c4c766dc687d52fe7fc534e50e7615acad7d93efc70e569492b9.

The full 2x decode / half-TTFT goal remains active and unmet.
