# FP32 split-K decode: no broad gain, rejected for general dispatch

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

PID **55556** exited 0 after 42.07 seconds. Actual Mac compilation and all 136
matrix/batch/variant checks passed within the unchanged numerical bounds.
Independent verification checked 55 archived files and 34 omitted raw-matrix
identities. Worst FP64 envelope fraction is 0.009973; worst native comparison
envelope fraction is 0.044918. Full generated outputs were not tested because
the component did not justify general production integration.

For large trained matrices, two-part unweighted GPU speedups are text
0.986/0.969, coder 0.996/0.941 and VL 1.007/0.983 for B1/B2 respectively.
Four-part variants are slower in all six groups (GPU speedups 0.916–0.944).
Wall results also fail to establish a broad gain; the only pooled positive
direction is VL B1 two-part at 1.009x. `component-by-batch.json` retains both
GPU and wall clocks, by batch and variant. Pooling distinct matrices is not
weighted by layer count and does not establish serving tokens/sec.

The individual VL B2 MLP down projection is an exception: two/four-part GPU
speedups 1.103/1.133 and wall speedups 1.035/1.037. This does not address the
unmet B1 throughput target and is not presented as a whole-model gain. Reject
general split-K dispatch; preserve this isolated shape observation for later
profiling rather than integrating a broad regression. No favorable full-model
claim or repeat follows the general candidate's result.

Archive SHA-256:
`be89df3044b8356ee701fb8f1f6b68ca77e7c0315abace73796cfd7caa9de1f8`.
Production sources, learned values, model settings and selected libraries are
unchanged. The full three-model 2x/half-latency goal remains active and unmet.

```text
python3 launch_decode_splitk_probe.py ROOT ROOT/logs/decode-splitk-20261015
python3 run_decode_splitk_probe.py ROOT OUTPUT
python3 archive_decode_splitk.py ROOT ARCHIVE
python verify_decode_splitk.py RESULTS
python summarize_decode_splitk.py RESULTS
```
