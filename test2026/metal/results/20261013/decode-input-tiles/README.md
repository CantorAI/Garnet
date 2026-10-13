# Rejected BF16 decode input tiles

The selected serving build remains `build-kv-chunk-vector`. No production
kernel changed in this experiment. Two isolated variants load 8 or 16 input
values per lane instead of 4, retaining four output columns per SIMD group,
128 threads per group, unchanged BF16 weights and FP32 input/accumulation.
Batch 2 shares each loaded weight across both input rows.

The final run covers the first-layer projections and actual runtime head of
each of the three pinned checkpoints. Head selection follows Garnet: the VL
language embedding, or the dense embedding/lm_head according to config
tie_word_embeddings. Irregular K=37/129/132 fixtures, output tails and padded
input/output rows are included. All 136 variant/batch/matrix numerical checks
pass the declared error tolerance and preserve output padding. Results are
not bit-exact because reduction order changes; raw errors are retained.

Each component uses two ABBA rounds, three warmups and 12 timed GPU command
buffers per window. These repeat one matrix and perturb cache reuse compared
with a full decoder. Ratios below are summed component GPU times, not model
tokens/s or a vLLM comparison. No full-model candidate is justified by them.

| Model | Batch | Input values/lane | Component sum ratio | Full-head ratio |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 8 | 0.982 | 0.994 |
| Qwen3-0.6B | 1 | 16 | 0.985 | 0.990 |
| Qwen3-0.6B | 2 | 8 | 0.964 | 0.968 |
| Qwen3-0.6B | 2 | 16 | 0.950 | 0.955 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 8 | 1.070 | 1.081 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 16 | 0.978 | 0.979 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 8 | 0.998 | 0.996 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 16 | 0.991 | 0.994 |
| Qwen3.5-0.8B | 1 | 8 | 0.976 | 0.979 |
| Qwen3.5-0.8B | 1 | 16 | 0.983 | 0.984 |
| Qwen3.5-0.8B | 2 | 8 | 0.979 | 0.977 |
| Qwen3.5-0.8B | 2 | 16 | 0.964 | 0.961 |

Neither variant establishes a useful consistent advantage, so both are
rejected for integration. There is no new serving benchmark, vLLM advantage
or 2x/half-TTFT success claim. The full objective remains active and unmet.
The 8.1% coder batch-1 head signal in the final scan is not repeatable in the
preceding complete-head scan, which measures a ratio of 0.998 for the same
case. Both actual runs remain saved; that isolated signal is not selected.

The first scan omitted the coder tied-embedding head; its 132 checks remain
saved as incomplete head coverage. The second scan included that head with
136 checks. The final scan explicitly follows the runtime configuration and
adds a head-presence assertion. Historical preparation sources are reconstructed
and verified against their original recorded SHA-256 before archival, so
current source is not presented as the source of earlier runs.

Complete commands/exits, hardware/source/checkpoint/matrix hashes, GPU device
limits, numerical errors, padding checks, process guards and ABBA windows are
retained. Raw checkpoint-derived matrices and binaries remain private on the
Mac. Competing inference runtimes never overlap; protected GPT-OSS, Pi and
original Mac checkouts are untouched.
