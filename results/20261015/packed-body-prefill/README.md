# Packed BF16 prefill tiles: rejected

Actual Mac PID70574 exited0 in79.9845s after the exact native validation and
full-model jobs finished. The private context copy tests FP32 SIMD matrix
tiles reading the same four-output BF16 storage. Activations and accumulation
remain FP32. Tile variants are16x64x64,32x64x32 and16x32x128. Production source
and selected libraries are unchanged.

The workload uses checkpoint-verified first-layer projection sets:7 text,
7 coder and15 VL matrices. Physical prefill rows are24/44,26/52 and80/160 for
B1/B2. The special text/coder key projection preserves its existing float
prefill policy. Four deterministic input patterns test the original FP64
forward-error bound and5e-6+5e-5*abs(control) comparison. No bound is widened.

Independent audit checks42 archived files and72 numerical cases. Nine checks
fail the original comparison envelope; all policies for text B1 and coder
B1/B2 are excluded from timing. The remaining9 profiles have576 timed
submissions across two ABBA rounds, eight samples per position. Candidate
wall fractions below divide candidate by control, so values above1 are slower.

| Model / batch | Policy1 wall fraction | Policy2 wall fraction | Policy3 wall fraction |
| --- | ---: | ---: | ---: |
| Text / 2 | 2.0261 | 1.7673 | 3.3227 |
| VL / 1 | 2.2671 | 1.7318 | 3.7046 |
| VL / 2 | 2.6200 | 1.7309 | 3.8778 |

This route is slower and has rejected numerical profiles; do not integrate
or repeat unchanged. Independent projections are repeatedly resident and
share one command submission. These measurements do not establish full-model
TPS, TTFT or a vLLM ratio. The actual harness has no synthetic ownership/abort
suite; an additional local controls draft was never uploaded or executed.
The metadata scope inherited an older runner's docstring; the exact generated
context, harness, source hashes, physical rows, commands and raw records
establish the experiment described here.

Archive SHA256:
`2f7e474d7e793a7d29a769b8dc7f7f2224f9e55c374ee825e87334a0acc22270`.

```text
python3 launch_packed_body_prefill.py ROOT ROOT/logs/packed-body-prefill-20261015 component
python3 archive_packed_body_prefill.py ROOT ARCHIVE
python -X utf8 verify_packed_body_prefill.py RESULTS VERIFICATION_JSON
```

The full three-model2x decode / half-TTFT goal remains active and unmet.
