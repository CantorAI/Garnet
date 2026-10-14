# Recorded real BF16 projections: no broad speed benefit

This component probe uses the current Garnet `bf16_decode` and
`bf16_decode_small` shader source, preserving the selected small-output
policy (N <= 256), original compact BF16 learned values and FP32 math.
It compares the normal native pipeline/direct encoding against a recorded
one-command indirect buffer. A second direct control enables indirect
support on the pipeline, isolating the support flag from replay.

Actual execution on the Apple M4 / 16 GiB, macOS 15.6.1 (24G90), exited
0. Source snapshots, extracted shader, compiler version, binary identity,
guarded process monitoring, commands and terminal states are archived.
The owning durable PID 98061 completed in 22.82 seconds; no probe remains
running. The selected native serving libraries are unchanged.

Coverage is 34 matrix fixtures across 68 batch profiles (rows 1/2):
25 trained first-layer/body/head matrices and nine synthetic tail
fixtures, with the three tail shapes repeated for each model. The three
Qwen checkpoint hashes, fixture hashes, BF16 dtypes, shapes and actual
checkpoint tensor slices were verified before execution. An earlier
progress update called these 68 matrices; 68 is the batch-profile count.

Four fresh input variants (original, negated, half and double) are
independently prepared. All 16,320 subsequent GPU executions, including
warmups, match the current native reference bit for bit, with unchanged
output padding. The 272 reference executions retain the prior independent
FP64 gamma(ceil(K/32)+32)*sumabs+1e-7 envelope; the worst normalized error
is 0.0100017036, below 1. This tests actual updated input reads, rather
than reusing previous prediction results.

Two ABBA rounds per control/profile retain 1,088 windows with twelve raw
encoding/wall/GPU samples each, totaling 13,056 timed executions. Input
updates and output checks are outside the timing interval for every
policy. Recording/setup time is separate. GPU, encoding and wall clocks
overlap and are not added. Each sample uses a fresh command buffer and
waits for completion; competing inference runtimes never run together.

| Comparison: indirect / direct, ratio of pooled means | Encoding | Wall | GPU |
| --- | ---: | ---: | ---: |
| Normal native pipeline | 1.14106 | 0.99634 | 0.97472 |
| Direct pipeline with indirect support | 1.15862 | 1.02070 | 1.00451 |

Median per-profile wall ratios are 1.03197 and 1.03591 respectively.
The small pooled GPU difference does not establish a broad wall-time gain;
CPU resource declarations cost more than this single-dispatch replay
saves. These are projection components, not full-model TPS or TTFT,
and cache behavior differs from traversing all model layers. No fresh
vLLM run or stock-BF16 advantage is claimed.

The gate does not justify replacing every individual projection with an
indirect buffer. A separately validated multi-operation MLP sequence may
amortize encoding differently; this report provides no evidence about
that untested sequence. The full 2x decode / half-TTFT target remains unmet.

Commands:

```text
python3 launch_indirect_bf16_projection.py ROOT ROOT/logs/indirect-bf16-projection-20261014
python3 run_indirect_bf16_projection.py ROOT OUTPUT/jobs
python3 archive_indirect_bf16.py ROOT ROOT/indirect-bf16-evidence.tar.gz
python summarize_indirect_bf16.py EVIDENCE_DIR SUMMARY_JSON
```

The archive SHA-256 is
`ec292180c2bccf0f529c2c2ab3fdd717730a0b5e3b300154ab3b02711bf62249`.
Local verification checks all 19 file hashes, actual binary/source/shader
identities, four terminal stages, fixture coverage, exact/padding and
FP64 gates, monitored samples, all sample counts and recomputed means.
The initial summarizer expected twice the actual fixture count; that
postprocessing assertion was corrected to the authoritative manifests,
without changing or rerunning the measurements. The successful derived
report is `summary.json`. Raw measurements retain their byte identities.
