# One-output SIMD decode: rejected

Three isolated Metal variants compute one output column per SIMD group,
with threadgroups of 32, 64 and 128 threads. The control computes four output
columns per SIMD group with 128 threads. Learned BF16 bits, FP32 input,
per-column accumulation sequence, reduction order and batch-2 weight sharing
are unchanged. There is no quantization or weight-layout expansion.

The actual M4/16 GiB run passes 204 bit-exact comparisons on first-layer
matrices and full vocabulary heads from all three checkpoints, plus irregular
synthetic tails. Both batches 1/2 and output-padding preservation are checked.
Every profile runs two ABBA rounds, three warmups per position, and twelve
samples per position. One GPU/inference job runs at a time.

Full-head control/candidate median GPU time ratios:

| Model / batch | 32 threads | 64 threads | 128 threads |
| --- | ---: | ---: | ---: |
| Text / 1 | 0.959 | 0.980 | 0.984 |
| Text / 2 | 0.923 | 0.922 | 0.985 |
| Coder / 1 | 1.011 | 0.966 | 0.999 |
| Coder / 2 | 0.929 | 0.950 | 0.977 |
| Qwen3.5 VL / 1 | 1.070 | 0.937 | 0.861 |
| Qwen3.5 VL / 2 | 0.820 | 0.810 | 0.954 |

Greater than one means faster. There is no consistent head improvement:
the single positive VL batch-1 result is paired with a substantial batch-2
regression. No serving integration, favorable-sample repeat, full-model A/B
or new vLLM comparison follows this rejected candidate. The selected library
is unchanged. Component ratios are not model tokens/sec claims.

Commands, source hashes, guards, terminal statuses, numerical checks and raw
timing windows are in completed/. Exact probe sources are in source/ and
test2026/metal/. Raw matrices and binaries remain on the Mac; their sizes
and hashes are in evidence-proof.json. Archive SHA256:
e03d4e8405b4f33b8dfce1c561f680621190aecd24272602c5669bdaf3a55021.

The full three-model 2x decode / half-TTFT goal remains active and unmet.
