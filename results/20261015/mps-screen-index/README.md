# MPS-derived screening dot products: rejected

Actual PID 31751 exited 0 in 43.3903 s. All numerical and input/fallback
checks pass, but every model/batch component is slower. No native integration
or serving gain is claimed. The general speed/TTFT goal remains unmet.

This separates index dot products from original BF16 refinement. The control
uses the current strict INT8 screening kernel. The candidate converts every
derived signed INT8 index value exactly to FP16 storage, then uses MPS with
FP32 query matrices, exact integer-valued FP16 index matrices and FP32 score
matrices. The unchanged strict filter applies original metadata/bounds to
those scores, and retained final logits still use original BF16 weights and
FP32 refinement. Both policies use the same exact 1024-row seed.

Every converted index value roundtrips exactly for all three complete heads.
The raw original head matches captured full-logit bits before admission.
Each policy checks retained final-score bits, every sampled rejection versus
the original full logits, greedy prediction and error semantics. Existing
candidate zero/tiny/large/NaN/Inf/mixed full-refinement fallback checks pass.
Filter sums and retention masks are not claimed bit-identical across MPS
and original dot-product reduction orders. No allowances were widened, and
the sampled checks do not certify all future inputs.

The independent audit checks 42 files, three complete index roundtrips,
51 captured-query pairs / 78 row states per policy, 33 unsafe-input cases,
3,264 raw timed executions, exact generated source, original asset hashes,
guards and both ABBA rounds. Common warmup is at least 250 ms per query;
two ABBA rounds use eight samples/window. Timings include the MPS product,
seed, bounds, screen/refine and original argmax in one command buffer. The
comparison runs after all native work and layout trials are terminal; no
other inference runtime is intentionally run simultaneously.

These means average ordinary captured steps, excluding the separate
mixed-row fixture. Fractions are candidate/control; above 1 is slower.

| Model / batch | GPU fraction | Wall fraction |
| --- | ---: | ---: |
| Qwen3-0.6B / 1 | 1.639805 | 1.562344 |
| Qwen3-0.6B / 2 | 1.676422 | 1.577403 |
| Qwen2.5-Coder-0.5B-Instruct / 1 | 1.662939 | 1.566295 |
| Qwen2.5-Coder-0.5B-Instruct / 2 | 1.635924 | 1.531293 |
| Qwen3.5-0.8B VL / 1 | 1.643010 | 1.589446 |
| Qwen3.5-0.8B VL / 2 | 1.696898 | 1.634038 |

The candidate doubles index element storage from one byte to two and adds
an MPS product / score matrix before refinement. These are implementation
differences, not a measured attribution of the entire slowdown. The 53–63%
wall penalty in every group is sufficient to reject this tested policy;
there is no positive result requiring a full-model integration or repeat.
The separate [lossless interleaved index](../interleaved-screen-index/README.md)
preserves the original dot arithmetic and gives repeated component gains.

The Objective-C++ harness compiles on the M4. The offline Metal compiler
could not execute because its optional Toolchain is absent; actual runtime
shader compilation and all GPU checks succeeded. No toolchain was installed.

Archive SHA-256:
`aeaab566b669ecefc26b6a96437bcbedf9f5ccc8a30d1e037d55e64e8e83d35c`.

```text
python3 launch_mps_screen_index.py ROOT ROOT/logs/mps-screen-index-20261015 checks
python3 archive_mps_screen_index.py ROOT ARCHIVE
python -X utf8 verify_mps_screen_index.py RESULTS VERIFICATION_JSON
```
