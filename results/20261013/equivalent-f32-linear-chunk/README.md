# Selected recurrent chunk build: current matched FP32 comparison

The selected Garnet build still falls short of the full 2× decode /
half-TTFT goal. Actual repeated comparisons confirm batch-2 coder and VL
decode ratios above 2×, but all first-token targets and the other decode
profiles remain unmet. No target is inferred from a different batch or model.

## Precision and workload

Native original BF16 storage and FP32 operations are compared with the
previously verified lossless FP32-expanded checkpoints and FP32 vLLM-Metal
model/KV arithmetic. Every learned value is identical, even though serialized
file bytes and weight-storage width differ. Original checkpoint assets and
the full equivalent checkpoint hashes are checked before execution, and the
exact-value proofs are retained. Earlier loaded-parameter audits demonstrate
the comparator actually loads FP32 tensors from these expanded files.

This is an additional FP32 comparison, not a replacement for stock BF16
vLLM-Metal results. `counts_as_default_bf16_speedup` remains false. There is
no quantization, prefix reuse, speculative decoding or simultaneous
inference runtime execution. Both use the same M4 / 16 GiB Mac mini, prompt
IDs, stop rules, caps, batch sizes and input images. All 72 requests match
complete normalized output IDs and finished flags.

Dense text/coder use chunk 16, capacity 256 and cap 64. Their prompts contain
22 and 26 IDs, respectively, and produce 8 and 17 output tokens including
stop. VL uses the retained batched recurrent output projection, chunk 64
for batch 1 or chunk 32 for batch 2 with adaptive 16-token tails. Each fresh
288×192 frame has 54 visual tokens and 79 prompt IDs; cap 32 yields complete
28-token captions. Input RGB/patch hashes match and comparator encoder-cache
identities are unique per row/frame.

## Actual warm measurements

Two fresh workers per runtime/profile each make three requests. Means use
the four non-first requests per runtime/profile. Every runtime shuts down
before the next starts; actual commands, statuses and process checks are saved.

| Model | Batch | Garnet tokens/s/row | vLLM FP32 tokens/s/row | Decode ratio | Garnet TTFT ms | vLLM TTFT ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 56.62 | 38.11 | 1.49 | 70.82 | 79.64 |
| Qwen3-0.6B | 2 | 51.76 | 26.85 | 1.93 | 120.40 | 97.64 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 72.32 | 47.23 | 1.53 | 54.85 | 61.62 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 69.90 | 29.57 | 2.36 | 74.62 | 84.82 |
| Qwen3.5-0.8B VL | 1 | 47.99 | 26.65 | 1.80 | 364.27 | 266.52 |
| Qwen3.5-0.8B VL | 2 | 42.76 | 16.64 | 2.57 | 609.65 | 438.67 |

The earlier observed text batch-2 ratio above 2× is not reproduced here;
its current ratio is 1.93. It is not reported as an achieved target. The
new recurrent projection reduces native VL latency versus its earlier
native control, but the comparator is still faster to the first word.

## Initialization and first use

These are separate measured intervals, not included in the warm means.
Filesystem and driver caches were not flushed. vLLM may perform preparation
during initialization, so first-use TTFT alone is not a complete startup cost.

| Model | Batch | Garnet initialization s | vLLM initialization s | Garnet first-use TTFT ms | vLLM first-use TTFT ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Text | 1 | 1.94 | 18.34 | 183.05 | 68.55 |
| Text | 2 | 2.60 | 17.04 | 228.64 | 105.17 |
| Coder | 1 | 2.14 | 16.62 | 157.20 | 62.34 |
| Coder | 2 | 2.13 | 16.60 | 177.80 | 94.14 |
| VL | 1 | 4.00 | 33.93 | 904.38 | 310.09 |
| VL | 2 | 5.06 | 33.50 | 883.49 | 513.87 |

## Evidence and remaining work

The selected native library SHA256 remains
`26f2120c3570a28607969f855f6950d626372b388361325ac49096eb30d60671`.
`logs/` contains actual output sequences, warm timing records, terminal
results, resource/process checks, equivalence proofs and metadata. Scripts
record their own hashes and use the validated direct WorldSense launcher
guard. Reproducible RGB/patch bytes are omitted with sizes/SHA256 retained;
their originals remain private on the Mac. Serialized per-operation profiling
is disabled for these measurements.

The current dense prompts both span two 16-token profiles padded to 32
positions. A larger single profile could reduce repeated weight passes, but
the earlier dense key-cache reference failure must be investigated and
resolved through independent numerical evidence before such profiles are
accepted. Decode remains GPU dominated. The rejected interleaved weight
layout is not used, and no further speedup is attributed to it.
