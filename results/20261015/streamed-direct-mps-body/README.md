# Streamed direct-row native MPS: rejected for integration

Actual owned PID 61259 exited 0 after 18.4861 s. The original B1/B2 inputs
and trained body weights are retained. Each submission encodes seven real
projections for text/coder or fifteen across VL linear/full-attention layers.
This is a set of independent per-layer projections, not an executing decoder.
All weights/inputs/outputs remain resident and no generation, prefix or image
is reused. The two VL matrices lacking common-scale half admission retain
their original raw BF16 route.

Three static candidate policies are tested against raw BF16: all admitted
projections, admitted MLP projections only, or admitted matrices with at least
1,048,576 weight elements. The experiment uses the same isolated context copy
as the individual direct-row study. Every actual forced MPS count, complete
output value count and numerical comparison is checked, including the raw
fallback matrices. No original backend source or library is modified.

The independent audit verifies 50 files, 18 group numerical checks and 1152
timed command submissions. All checks pass unchanged FP64/original-control
bounds. Each policy/batch/model has 250 ms common warmup, two ABBA rounds,
two untimed operations and eight timed submissions per position. Guards sample
every 0.5 s and no competing inference/probe job intentionally overlaps.

| Model / batch | GPU fraction, all / MLP / large | Wall fraction, all / MLP / large |
| --- | ---: | ---: |
| Text / 1 | 0.850 / 0.947 / 0.891 | 1.178 / 1.089 / 1.195 |
| Text / 2 | 1.110 / 0.995 / 1.021 | 1.287 / 1.097 / 1.277 |
| Coder / 1 | 0.989 / 0.946 / 0.965 | 1.292 / 1.081 / 1.106 |
| Coder / 2 | 1.045 / 1.046 / 0.961 | 1.311 / 1.160 / 1.067 |
| VL / 1 | 0.928 / 0.945 / 0.944 | 1.256 / 1.111 / 1.175 |
| VL / 2 | 0.948 / 0.969 / 1.199 | 1.254 / 1.115 / 1.361 |

Some GPU intervals improve, but every observed wall policy is 6.7–36.1%
slower. Amortizing submission overhead does not remove the penalty in these
sets. This is sufficient to reject these routes for integration without a
favorable-sample repeat; it does not establish the cause or all future workload
behavior. It does not measure full-model decode TPS, TTFT, cold startup or a
new vLLM ratio. The full goal remains unmet.

Archive SHA-256:
`6c8f799ad8800d89ee2be8d6de1a77c1402c259858b308f125b72c32b8820528`.
Source snapshots, generated context, pinned original dependencies, compiler
definitions, individual-admission evidence, actual counters, guards and every
timing sample are preserved. The follow-up
[API-validation option test](../fastencode-streamed-mps-body/README.md) also
keeps arithmetic precision unchanged and does not remove the wall penalty.

```text
python3 launch_streamed_direct_mps_body.py ROOT ROOT/logs/streamed-direct-mps-body-20261015 checks
python3 archive_streamed_direct_mps_body.py ROOT ARCHIVE
python -X utf8 verify_streamed_direct_mps_body.py RESULTS VERIFICATION_JSON
```
