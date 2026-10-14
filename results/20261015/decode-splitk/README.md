# FP32 split-K decode: isolated component test in progress

This experiment divides each learned BF16-weight dot product into two or four
parallel FP32 partial sums, then reduces them in FP32. It retains exact native
BF16 expansion and FP32 activations/output; reduction order changes. The
first kernel uses the original 32-lane/four-output layout. The second kernel
reads only complete partial sums and writes logical outputs, preserving row
padding. Both encoders share one tracked command buffer; GPU span and wall
time include both kernels and their dependency.

The control remains the selected small-output kernel for N≤256 and the native
general kernel otherwise. Tests retain all original real first-layer/full-head
matrices, B1/B2, synthetic irregular K/N tails and padded input/output strides.
Both policies must meet the unchanged independent FP64 envelope
`gamma(ceil(K/32)+32)*sumabs+1e-7`; candidate-versus-control remains
`5e-6+5e-5*abs(control)`. A failure rejects the method. Component timing alone
does not establish complete model correctness or serving throughput.

Owned PID **55556** started at `ROOT/logs/decode-splitk-20261015` only after
initial combined run, its repeat and the eight-lane queue all reached terminal
exit 0. The earlier evidence is archived and independently verified. Existing
0.5-second guards remain intact and reject other inference/probe jobs. No
competing runtime is deliberately run simultaneously.

Generated shader/harness construction passed locally. Actual Mac compilation,
all numerical gates and component timing remain pending until terminal
evidence is collected. Production sources, learned values, model settings and
selected libraries are unchanged. The full three-model 2x/half-latency goal
remains active and unmet.

```text
python3 launch_decode_splitk_probe.py ROOT ROOT/logs/decode-splitk-20261015
python3 run_decode_splitk_probe.py ROOT OUTPUT
```
