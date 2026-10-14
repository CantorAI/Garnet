# Eight-lane FP32 decode: component probe queued

The full optimization goal still needs batch-1 decode speed, particularly for
text and coder. This isolated experiment tests four independent eight-lane
groups per SIMD, each calculating two outputs with FP32 input, multiplication
and accumulation. It retains native BF16 learned values and their exact FP32
expansion. It changes reduction order and per-thread work distribution; it
does not claim bit equality or model-level correctness before execution.

This differs from the rejected sixteen-lane/four-output implementation that
retained two original partial accumulators per thread. The new kernel uses
one two-output accumulator per row, aiming to reduce register demand while
processing eight outputs per SIMD. Actual performance remains unmeasured.
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

This is a queued experiment, not a success claim. Generated shader/harness
construction succeeded locally; actual Metal compilation, numerical checks
and timings require terminal Mac evidence. Production sources and retained
libraries are unchanged. The main optimization goal remains active and unmet.

```text
python3 queue_decode_eight_lane_probe.py ROOT ROOT/logs/decode-eight-lane-20261015
python3 run_decode_eight_lane_probe.py ROOT OUTPUT
```
