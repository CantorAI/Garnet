# Vision FP32 products versus exact common half alpha

This new component experiment preserves all learned values and gives repeated
GPU reductions for vision-block projections. It does not yet establish image
inference latency, throughput or the overall 2x decode / half-TTFT goal.
No production route or selected library changes in this experiment.

The earlier per-row-scaled half vision trial failed to improve serving latency.
This trial instead compares the current expanded-FP32 MPS product against one
exact common-scale half-storage product with the inverse power of two in MPS
alpha. Both take FP32 inputs and produce FP32 outputs. There is no rescale
encoder or intermediate result matrix. Raw BF16 values restore bitwise from
the encoded half representation; there is no rounding or clipping of weights.
Internal MPS kernel counts are not claimed.

The real M4 trials cover all **50** trained 2D block/merger projection matrices
from the frozen Qwen3.5-0.8B checkpoint: 48 block products and two merger
products. Patch embedding, positional embeddings, norms and biases keep their
existing paths. Matrix slices and the complete checkpoint SHA-256 are checked
before running. Block products use actual physical row counts 216/432;
merger products use 54/108. These correspond to the unchanged batch-1/2
288x192 image grid, 216 patches and 54 visual tokens per image.

All 50 matrices admit an exact common half scale, with no exclusions. Across
both successful runs, **36,890 validated GPU executions** have zero numerical
difference from the FP32 control and satisfy the independent unchanged
`5e-6 + 5e-5*abs(reference)` FP64 envelope. Worst normalized FP64 error is
0.433996587, below 1. Numerical equality is not a signed-zero bit-identity
claim. Every eligible output and padding element is checked. Fresh inputs are
deterministic sine vectors scaled by four exact power-of-two variants; actual
image-dependent activations and complete image inference remain to be tested.

Each run contains 100 matrix/row profiles and 9,600 timed executions. Each
profile alternates common warming for at least 250 ms, then two ABBA rounds,
four windows/round and twelve samples/window. Setup, uploads, allocation and
validation are outside the recorded encoding, GPU and wall clocks.

| Product group / rows | Mean per-profile GPU fraction, initial / repeat | Wall fraction, initial / repeat | Encoding fraction, initial / repeat |
| --- | ---: | ---: | ---: |
| Blocks / 216 | 0.9168 / 0.9192 | 0.9759 / 0.9761 | 1.1062 / 1.1084 |
| Blocks / 432 | 0.9398 / 0.9346 | 0.9789 / 0.9775 | 1.1033 / 1.0820 |
| Merger / 54 | 0.9518 / 0.8976 | 1.0133 / 0.9814 | 1.1220 / 1.1455 |
| Merger / 108 | 0.8339 / 1.0012 | 0.9202 / 1.0207 | 1.1061 / 1.0969 |

Fractions are candidate/control; means above average the independently
recomputed per-profile fractions. Block GPU improvements repeat at about
8.1% for B1 and 6.5% for B2, with only 2.2–2.4% component wall reductions.
Encoding costs rise. Merger directions are inconsistent. A native experiment
should therefore initially retain FP32 merger products and test only block
products, with full bounds, fallback and ownership validation. No benefit for
full image-to-first-token latency can be inferred from these component ratios.

Integration must preserve the existing decoder row limits while separately
supporting vision rows 216/432, preserve the FP32 fallback for inexact matrices
and handle tiny vision grids without selecting an unintended decode kernel.
Actual vision dispatch counters and complete reference images/outputs are
required before any admission. The broader goal remains active and unmet.

Initial owned PID 6184 exits 0 in 155.84 seconds; fresh confirmation PID 9206
exits 0 in 155.62 seconds. The second starts after the first is terminal.
Guards reject competing inference/probe jobs every 0.5 seconds and terminate
only owned processes. OS caches/GPU frequency are uncontrolled, and shorter
external activity between samples cannot be categorically excluded.

Initial archive SHA-256
`e43f9b6da9518232697de67ed06c092856cbb8bfa5f549c00fd25484de700e65`
(20 files). Combined archive SHA-256
`234f4684b6c253db1e76b61977821cd687e37807db6fcd6a0b9f4c972eea8fa2`
(37 files plus 50 canonical matrix proofs). Local `summary.json` verifies every
file hash, actual source snapshots, both terminal command/monitor records,
checkpoint/matrix identities, complete numerical gates, warm durations, all
ABBA windows and independently recomputed raw timing means. Commands, compiler
output, device observations and exact source are preserved in `evidence/`.

```text
python make_vision_global_alpha_probe.py
python3 launch_vision_global_alpha_probe.py ROOT ROOT/logs/vision-global-alpha-probe-20261014
python3 launch_vision_global_alpha_probe.py ROOT ROOT/logs/vision-global-alpha-probe-confirm-20261014
python3 archive_vision_global_alpha_probe.py ROOT ARCHIVE
python verify_vision_global_alpha_probe.py RESULTS REPORT_JSON
```
