# Eight-lane FP32 decode: numerically valid, slower, rejected

The full optimization goal still needs batch-1 decode speed, particularly for
text and coder. This isolated experiment tests four independent eight-lane
groups per SIMD, each calculating two outputs with FP32 input, multiplication
and accumulation. It retains native BF16 learned values and their exact FP32
expansion. It changes reduction order and per-thread work distribution; it
does not claim bit equality or model-level correctness before execution.

This differs from the rejected sixteen-lane/four-output implementation that
retained two original partial accumulators per thread. The new kernel uses
one two-output accumulator per row, aiming to reduce register demand while
processing eight outputs per SIMD. Actual component performance was slower.
It tests 64- and 128-thread groups, B1/B2, real first-layer and full-head
checkpoint matrices, plus the established irregular tail and stride cases.
The control selects the existing small-output kernel for N≤256 and the
selected general kernel otherwise.

The established independent FP64 and control-comparison bounds remain
unchanged: both policies must satisfy
`gamma(ceil(K/32)+32)*sumabs+1e-7` against FP64, and the candidate must satisfy
`5e-6+5e-5*abs(control)` against native control. Padding and finite values are
checked before timing. A longer per-lane sum does not justify relaxing those
bounds. A failure rejects the experiment. A component gain still requires
complete model/cache/head/image validation and complete generated outputs
before any integration or end-to-end speed claim.

Owned queue PID **33256** was launched at
`ROOT/logs/decode-eight-lane-20261015`. It verifies exact owned dependency
**27124**, waits for its actual terminal exit 0, archives combined evidence,
then invokes one guarded component runner. The queue itself runs no inference
or GPU probe while waiting. Guards sample every 0.5 seconds and reject other
inference/probe jobs; no timed runtimes are intentionally overlapped.

PID **33256** exited 0 after 1082.43 seconds including the dependency wait.
Actual Mac compilation and all 136 matrix/batch/variant checks passed within
the unchanged bounds. Worst FP64 envelope fraction is 0.009709; worst native
comparison envelope fraction is 0.072507. Independent verification checked
53 archived files and 34 omitted raw-matrix identities.

For large trained matrices, unweighted sum-of-median GPU speedups are
0.87–0.93x across all model/batch/variant groups; corresponding wall speedups
are 0.91–0.99x. GPU time is about 7–15% longer and wall time 1–10% longer.
`component-by-batch.json` retains B1/B2 separately and both GPU and wall
clocks. This pooling is not weighted by layer count and is not a full-model
measurement. No favorable repeat, production integration or serving-speed
claim follows this broad regression. Production sources and retained
libraries are unchanged. The full optimization goal remains active and unmet.

Archive SHA-256:
`f56694dab8c65f81862fa79b12b670fb85f942f648c46e0e58bbde34d2d56f30`.

```text
python3 queue_decode_eight_lane_probe.py ROOT ROOT/logs/decode-eight-lane-20261015
python3 run_decode_eight_lane_probe.py ROOT OUTPUT
python3 archive_decode_eight_lane.py ROOT ARCHIVE
python verify_decode_eight_lane.py RESULTS
python summarize_decode_eight_lane.py RESULTS
```

The archive preserves terminal evidence even if the numerical gate rejects
the kernel; regenerable raw matrices are represented by exact byte lengths,
checkpoint identities and hashes. The success audit independently recomputes
component medians and ratios, checks all original 136 matrix/batch/variant
cases and monitor records, verifies the selected small-output control and
unchanged numerical bounds, and confirms the dependency's successful terminal
archive. It must not be used as a success audit for a failed or partial run.
