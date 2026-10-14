# Vision FP32 products versus exact common half alpha

This new component experiment preserves all learned values and gives repeated
GPU reductions for vision-block projections. It does not yet establish image
inference latency, throughput or the overall 2x decode / half-TTFT goal.
The component trial leaves selected libraries unchanged. An optional native
integration is now built and under full validation; no serving gain is yet
established for it.

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

Owned validation PID **13040** is running all nineteen original full-model/
cache/image/head/prefill stages at `ROOT/logs/vision-global-alpha-checks-20261014`.
Dependent PID **16264**, directory `ROOT/logs/vision-global-alpha-native-20261014`,
verifies that exact live handle before waiting and starts all six sequential
native ABBA profiles only after terminal validation exit 0. It requires actual
48 block-product dispatches for every candidate VL image and zero for controls
and text/coder, complete frozen outputs and fresh image identities. Other head,
norm, reset and joined-MLP experiments stay disabled.

The bounded historical snapshot in `native/pending` verifies both current
handles, unchanged selected libraries, source bundle and real CTest output.
Only configure/build/CTest are complete in that snapshot; the remaining
sixteen stages and all inference measurements are still required. Inspect
authoritative current handles/terminal records before continuing. The prepared
terminal archiver and local verifier check all original cases, source/binary
identities, actual image use and independently recomputed timing phases.
No native image-latency or decode gain is yet claimed. The broader goal remains
active and unmet; GPT-OSS and Pi checkouts remain unchanged.

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
