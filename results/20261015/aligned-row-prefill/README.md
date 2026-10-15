# Physical MPS prefill row alignment: no broad promotion

Owned PID82838 exited0 in80.0128s after native confirmation PID79571 completed.
This private context retains logical rows24/44 text,26/52 coder and80/160 VL.
Matrix allocations reserve zero-filled backing rows up to the next128-row
boundary. Three policies round only common-alpha MPS multiplication rows to
a multiple of32/64/128. Other operations keep logical shapes. Original FP32
activations, accumulation and output and lossless scaled-half weights remain.
The special float key and unsupported weight routes retain their controls.

There is no CPU/GPU activation padding copy inside timed work: backing zeros
are initialized at allocation. Allocation and increased storage are outside
component timings. Every numerical check inspects the unused backing rows
of inputs and outputs for numerical zero after submission. This is a private
component design, not a complete graph implementation or TTFT measurement.

Independent audit verifies43 archived files,72 numerical cases, independent
CPU common-half admission against hashed original raw weights, actual route
counts and zero tails. Nine cases fail the original5e-6+5e-5*abs(control)
comparison, excluding all text B1 and coder policies from timing. Bounds are
unchanged. Nine admitted profiles have576 timed submissions in two ABBA
rounds with eight samples per position. Fractions are candidate/control.

| Model / batch | Align32 GPU / wall | Align64 GPU / wall | Align128 GPU / wall |
| --- | ---: | ---: | ---: |
| Text / 2 | 1.0974 / 1.0614 | 1.0929 / 1.0748 | 1.9889 / 1.7228 |
| VL / 1 | 0.9629 / 0.9780 | 1.1469 / 1.1135 | 1.1823 / 1.1572 |
| VL / 2 | 1.0178 / 1.0150 | 1.1139 / 1.1041 | 1.3637 / 1.3379 |

VL B2 already has160 rows divisible by32, so align32 performs zero aligned
dispatches and is a no-change timing control. VL B1 align32 shows only a
2.2% component wall reduction in this single run; most policies regress,
including text B2 by6–72%. No broad integration or unchanged repeat is
justified. This does not establish model latency, decode speed or a vLLM gain.
The full2x decode / half-TTFT goal remains active and unmet.

Archive SHA256:
`4197d6710fdf567457e21320111b3f800aa1c8825aacd6dd12402a82b6ac5e68`.
Exact compiler/source/checkpoint identities, commands, guards and raw
numerical/timing records are under `evidence/`.

```text
python3 launch_aligned_row_prefill.py ROOT ROOT/logs/aligned-row-prefill-20261015 component
python3 archive_aligned_row_prefill.py ROOT ARCHIVE
python -X utf8 verify_aligned_row_prefill.py RESULTS VERIFICATION_JSON
```
