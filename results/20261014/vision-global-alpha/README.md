# Vision FP32 products versus exact common half alpha

The component experiment preserves learned values and gives repeated GPU
reductions for vision-block projections. Full native image trials now pass
nineteen numerical stages and 96 complete requests, including a fresh repeat.
Warm TTFT decreases modestly, while decode throughput remains essentially flat.
The optional integration stays disabled in the selected comparison bundle;
there is no fresh vLLM comparison for it and the overall goal remains unmet.

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
image-dependent activations are covered separately by the native trials below.

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

## Optional native integration

`GARNET_METAL_VISION_GLOBAL_ALPHA` defaults OFF and requires global half alpha.
Only BF16 2D block QKV, attention output and MLP projection weights are selected.
Norms, biases, patch/position embeddings and both merger products retain their
existing FP32 path. Inexact common scales also fall back to the original FP32
matrix. A separate shared-weight cache key preserves the vision role.

Tagged vision weights use the existing lossless representation but always
take the MPS common-alpha route, including tiny grids with one or two rows.
Vision-only rows up to the existing grid limit are supported, while decoder
weights retain their original 128-row limit. Learned BF16 buffers, precision,
shape/device checks, alias guards and output ownership remain intact. Context
Begin/Abort and graph clear/run reset a separate counter; graph Run publishes
it after successful GPU submission. Image workers report actual per-image
counts, rather than relying on the enabled feature marker.

Installation matched all fifteen original source hashes against branch base
`2394abcb` and preserved their exact bytes before installing sixteen candidate
files. The isolated candidate builds and both CTests pass on the actual M4.
The new test checks rows 1/2/54/108/216/432 against original FP32 MPS and
independent FP64 references with four fresh inputs, unchanged decoder limits,
inexact-scale fallback representation, out-of-profile rows and Begin/Abort/
readiness recovery. Candidate library SHA-256:
`0997e2ba00ce52766b0e8b1fa0b46f44bcff6b081883e9a8c7ba93c2f3635d23`.

Owned validation PID **13040** exited 0 in 1201.13 seconds, passing all nineteen
original full-model/cache/image/head/prefill stages. Dependent PID **16264**
exited 0 in 1178.98 seconds including its validation wait, completing all six
native ABBA profiles (72 requests). Fresh VL confirmation PID **40142** exited
0 in 69.25 seconds (24 requests). Every candidate VL request records 48 actual
block-product dispatches; controls and text/coder record zero. Frozen complete
outputs and fresh image identities match. Other experimental paths stay off.

| VL batch | Initial TTFT control / candidate ms | Repeat TTFT control / candidate ms | Repeat decode control / candidate tokens/s per row |
| --- | ---: | ---: | ---: |
| 1 | 196.500 / 192.566 | 194.484 / 192.671 | 47.889 / 47.757 |
| 2 | 359.646 / 356.024 | 358.828 / 352.366 | 44.447 / 44.394 |

TTFT reductions are 2.0% / 1.0% initially and 0.9% / 1.8% on repeat.
Repeat decode changes are -0.28% / -0.12%. First request after initialization
on repeat is 333.159 / 331.487 ms for B1 and 552.286 / 548.798 ms for B2.
Caches were not flushed, so these are not hardware-cold measurements.

Observed repeat vision-only wall time is 53.927 / 54.182 ms for B1 and
114.225 / 111.209 ms for B2. Vision excludes the separately measured reset
using same-origin clock subtraction. Preprocessing and text prefill also vary;
the complete TTFT difference cannot be attributed solely to vision GPU work.
The small serving gain does not resolve the broader speed targets. The original
selection is retained until a useful combined candidate has fresh matched
comparison evidence. GPT-OSS and Pi checkouts remain unchanged.

The combined native archive SHA-256 is
`d0eb98a2947d7dd7d99fe393380af0e9e0ef4800118414684b970916611a41a2`.
Both local audits verify all 811 archived files, nineteen numerical stages,
72 initial / 24 confirmation outputs, source/binary identities and real
dispatch counts. Reports are `native/verification.json`, `native/confirmation.json`
and their observed phase summaries. Historical `native/pending` is superseded.

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
python add_vision_global_alpha_counters.py
python prepare_vision_global_alpha_sources.py
python3 install_vision_global_alpha_sources.py ROOT
python3 launch_vision_global_alpha_checks.py ROOT ROOT/logs/vision-global-alpha-checks-20261014 checks
python prepare_vision_global_alpha_profiles.py
python3 queue_vision_global_alpha_profiles.py ROOT ROOT/logs/vision-global-alpha-native-20261014
python3 archive_vision_global_alpha_native.py ROOT ARCHIVE
python verify_vision_global_alpha_native.py RESULTS REPORT_JSON
```

Counter preparation is a one-time editing helper and rejects reapplication to
the completed source. Rebuild from the committed implementation or archived
source bundle rather than rerunning it on an already modified branch.
