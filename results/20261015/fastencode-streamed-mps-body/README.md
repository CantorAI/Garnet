# MPS argument-validation option: wall penalty persists

Actual owned PID 61885 exited 0 after 18.5174 s. This isolated context-copy
experiment sets only `MPSKernelOptionsSkipAPIValidation` on the common-scale
MPS product, keeping FP32 input/output, lossless learned weights, existing
Garnet shape/ownership checks and unchanged numerical bounds. Reduced
precision is not enabled. It repeats the streamed all/MLP/large policies at
original B1/B2 and retains the raw BF16 control.

Apple documents this option as a possible small CPU saving from argument
checking; the experiment tests whether that addresses the observed wall
penalty. [Apple MPS documentation](https://developer.apple.com/documentation/metalperformanceshaders/mpskerneloptions/skipapivalidation?language=objc)

Independent verification checks 52 files, the exact regenerated context and
harness, all actual MPS counts, 18 numerical group checks and 1152 timed
submissions. Every sampled numerical check passes the original FP64 envelope
and original-control comparison. The 29 trained matrices remain in each
appropriate projection set, with two VL matrices retaining their raw route.
Two ABBA rounds, warmup, 0.5 s guards and separate GPU/wall clocks follow the
preceding streamed study. All competing investigation jobs are terminal first.

| Model / batch | GPU fraction, all / MLP / large | Wall fraction, all / MLP / large |
| --- | ---: | ---: |
| Text / 1 | 0.935 / 0.931 / 0.911 | 1.213 / 1.064 / 1.221 |
| Text / 2 | 1.107 / 1.028 / 1.142 | 1.312 / 1.088 / 1.352 |
| Coder / 1 | 0.955 / 0.958 / 0.990 | 1.285 / 1.126 / 1.139 |
| Coder / 2 | 1.075 / 1.045 / 0.994 | 1.307 / 1.121 / 1.125 |
| VL / 1 | 0.935 / 0.954 / 0.957 | 1.252 / 1.107 / 1.166 |
| VL / 2 | 0.943 / 0.948 / 1.151 | 1.261 / 1.087 / 1.330 |

Every policy remains slower in wall time, by 6.4–35.2% versus its contemporary
raw control. The option is not integrated, and these results do not establish
a full-model speedup or a causal comparison against the previous option-off
trial. Independent per-layer sets are not a full decoder. No new model TPS,
TTFT, cold-start or vLLM-target claim is made. The full goal remains unmet.

Archive SHA-256:
`28f062e35a4ac43b7e6907260595b4f9d3c44ef35185e2d2b41cad15c72db450`.
All original serving sources, validated libraries and protected checkouts
remain unchanged. Commands, source/binary identities, generated code, actual
counters and raw samples are preserved in `evidence/`.

```text
python3 launch_mps_fastencode_body.py ROOT ROOT/logs/fastencode-streamed-mps-body-20261015 checks
python3 archive_streamed_direct_mps_body.py ROOT ARCHIVE fastencode-streamed-mps-body-20261015
python -X utf8 verify_streamed_direct_mps_body.py RESULTS VERIFICATION_JSON
```
