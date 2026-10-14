# Recorded four-kernel MLP: clean numerics, unstable wall-time gains

This probe records the original gate projection, up projection, SiLU
multiply and down projection into one GPU indirect-command plan. The
learned BF16 weights, FP32 math, original shaders and dispatch widths
remain. One plan orders all four kernels; another permits independent
gate/up overlap and explicitly waits for both before SiLU, then waits
for SiLU before down. Barrier flags are set before command encoding,
following Apple's documented before-command ordering. A direct control
with indirect pipeline support separates that flag from replay.

Actual initial and fresh-process confirmation jobs (PIDs 99355 / 99806)
completed with exit 0 in 8.41 / 8.40 seconds. Both cover all three frozen
Qwen checkpoints and batch 1/2, using the first trained MLP layer.
Checkpoint hashes, BF16 tensor shapes, fixture hashes and exact canonical
tensor slices were verified. Inputs are freshly uploaded in four variants
(original, negated, half and double). Both runs retain all four intermediate
buffers, complete outputs and padding bit for bit in 4,320 total validated
executions. Independent FP64 projections and a double-precision SiLU
reference keep the existing bounds; worst normalized errors are
0.00324921315 / 0.00217654517 respectively. No tolerance is widened.

Each run has 24 separately checked reference executions and 1,728 timed
executions in 144 ABBA windows. Two rounds and two windows per policy
provide 48 timed samples per model/batch/comparison. CPU input updates and
output checks are outside the measured encoding/wall/GPU clocks for both
policies. Recording/setup is separate; overlapping clocks are not added.

| Component | Serial wall fraction, initial / confirmation | Overlap wall fraction, initial / confirmation |
| --- | ---: | ---: |
| Text B1 | 1.07570 / 0.94843 | 1.09201 / 1.02925 |
| Text B2 | 1.07441 / 0.96611 | 0.99103 / 0.95080 |
| Coder B1 | 0.91593 / 0.90884 | 0.97341 / 1.00120 |
| Coder B2 | 1.07353 / 1.06286 | 0.96694 / 1.02711 |
| VL B1 | 0.95681 / 0.98930 | 0.95603 / 1.00700 |
| VL B2 | 1.01458 / 0.89872 | 0.98200 / 1.08140 |

These are ratios of pooled wall means within each run, indirect/native;
smaller is faster. The overlap-plan gains do not repeat. Serial coder B1
has a repeated pooled reduction around 8–9%, but that comes from round 0:
round-1 fractions are 1.0010 and 1.0141, flat/slower. First-round/order or
warmup sensitivity is plausible but not directly established by frequency
telemetry. Thus a pooled figure alone is insufficient evidence for native
integration. CPU encoding generally improves around 10%, yet whole-kernel
wall timings are mixed. This is a component gate with one layer repeatedly
cached, not traversal of every layer or a full-model throughput result.

No runtime integration is admitted on these measurements. Selected Garnet
libraries and the GPT-OSS/Pi checkouts remain unchanged. There is no fresh
vLLM comparison, no prediction/prefix/image-result reuse and no stock-BF16
advantage claim. The full 2x decode / half-TTFT goal remains unmet.

Executed commands:

```text
python3 launch_indirect_mlp_probe.py ROOT ROOT/logs/indirect-mlp-probe-20261014
python3 launch_indirect_mlp_probe.py ROOT ROOT/logs/indirect-mlp-confirm-20261014
python3 archive_indirect_mlp.py ROOT ARCHIVE [JOB_NAME]
python summarize_indirect_mlp.py EVIDENCE_DIR SUMMARY_JSON [JOB_NAME]
```

Initial archive SHA-256:
`2766a4ef395693c3e06167c57d2647261f8a0fe9fd38900f372fcd87edaf0b54`.
Confirmation:
`afceb229c7ba21edcb185dc6f78f003ef1fa68bbc78c9bf3894e23cddd4a2637`.
The local verifier checks all 17 files in each archive, terminal stages,
actual binary/source/shader/asset identities, guarded samples, exact and
independent numerical gates, sample counts and recomputed clocks. Raw
records and executed sources retain their byte identities in `evidence`
and `confirmation/evidence`; derived reports are the two `summary.json`
files. GPU frequency was not pinned, OS caches were not flushed, and
0.5-second monitoring does not exclude arbitrary shorter external jobs.
