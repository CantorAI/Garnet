# Native large-matrix BF16 body layout

The full2x decode / half-TTFT goal remains unmet. This is an OFF-default native
integration of the twice-tested component's large-matrix policy, not an
established complete-model speed improvement.

The native shader is byte-identical to the tested component kernel. Only the
BF16 weight load layout changes; original values, FP32 arithmetic order and
SIMD reduction remain. CPU packing checks every value and tail zero before
publication. Masters remain available. Admission requires256 < output width
<=8192, input width divisible by4, at least1048576 values, decoder weights and
device-buffer bounds. Small/odd/vocabulary/prefill routes retain their existing
weights and kernels. A native dispatch counter is published after submission
and reset on begin/session reset/abort/failure.

Actual Mac build and all six CTests passed, including64 new cases against a
same-context original BF16 control: original output bits and FP64 envelope,
large/tail/fallback geometry, pending/foreign ownership, uninitialized input,
abort/recovery and unchanged prefill route.

Validation PID49634 completed all28 existing trained/reference/recovery stages
with exit0 in1421.9852s. Native PID49699 subsequently completed all six profiles,
exit0 in1589.1177s including its dependency wait. Independent verification
checks2744 archived files,72 complete output requests,51 additional trained-head
queries, recovery and actual route/submission/clear counters. Four first/last
VL B2 chunk80 layer profiles carry12 steps; the worst unchanged bound fraction
is0.5432836183473805.
Both policies use the same latency bundle and VL B2 `[80]` plan. Only candidate
decode uses large-body packing. Actual body counts must be positive and stable
on decode steps and zero for prefill; the legacy library's API is not accessed
for the new counter. All external inputs, output caps, stops and complete
expected outputs remain frozen. Init, first-after-init and warm behavior stay
separate, including packing/memory costs. No vLLM ratio is inferred here.

Initial full-model warm results, candidate versus the same latency bundle:

| Model / batch | Decode tokens/s control -> candidate | TTFT ms control -> candidate | Peak process RSS GiB control -> candidate |
| --- | ---: | ---: | ---: |
| Text / 1 | 62.9844 -> 66.8117 | 37.1388 -> 34.9353 | 2.278 -> 3.062 |
| Text / 2 | 61.5372 -> 64.4616 | 57.6949 -> 55.5420 | 2.379 -> 3.144 |
| Coder / 1 | 80.0947 -> 79.2433 | 26.0817 -> 29.0723 | 1.932 -> 2.524 |
| Coder / 2 | 76.3159 -> 76.8756 | 41.9688 -> 43.7415 | 1.931 -> 2.524 |
| VL / 1 | 53.6845 -> 55.7744 | 174.3181 -> 185.2596 | 3.371 -> 4.294 |
| VL / 2 | 50.3868 -> 51.1506 | 299.5025 -> 300.0011 | 3.396 -> 4.314 |

Packing also increases initialization: text2.36/2.38s ->2.84/2.86s,
coder1.98s ->2.34s, VL4.15/4.20s ->4.71/4.72s. Decode improvements are mixed,
and some TTFT measurements regress. This is not a general promotion.
Confirmation PID79571 exited0 in173.4467s. Independent verification covers
3065 combined archived files and72 complete confirmation outputs; together
the trials contain144 complete matching output requests. The join audit
checks identical libraries, source hashes, assets and policy case files.

| Model / batch | Repeat decode tokens/s control -> candidate | Repeat TTFT ms control -> candidate |
| --- | ---: | ---: |
| Text / 1 | 64.8275 -> 68.1959 | 37.3832 -> 36.2809 |
| Text / 2 | 60.5172 -> 63.3560 | 58.6367 -> 57.1369 |
| Coder / 1 | 80.1492 -> 78.9561 | 26.1120 -> 27.9232 |
| Coder / 2 | 76.9685 -> 77.0814 | 41.8480 -> 43.0820 |
| VL / 1 | 54.3879 -> 55.7711 | 183.3564 -> 182.1402 |
| VL / 2 | 50.3807 -> 51.1584 | 298.5183 -> 298.2194 |

Text decode gains repeat at5.2–6.1% B1 and4.7–4.8% B2. Coder B1 regresses
1.1–1.5%, and coder TTFT regresses in both trials. VL gains repeat at2.5–3.9%
B1 and1.5% B2, with roughly1 GB additional RSS; its TTFT direction does not
repeat. Retain an optional text candidate, preserve the coder control and
avoid a broad default promotion. No new vLLM ratio is inferred.

Initial archive SHA256:
`04b9efef8c4ceee8b358b3477a7cd7732f757235879f7fffa537966292bdbce1`.
Combined archive SHA256:
`a545b3fde30957031b9378a916313e5dcd56e43d48d3f1637ba7773ae3631379`.

Control library SHA256:
`9ee337d9db7d14bcff29a90970c4b5ac9021aee9f89a38a7f6f364051927d77b`.
The eight replaced remote files are preserved in
`logs/interleaved-body-native-original-source-20261015`. Installation happened
only after both matched comparator cohorts were terminal and independently
archived. The original Mac, GPT-OSS and Pi checkouts are preserved.

Both native jobs and the repeat are terminal and archived. Do not rerun the old
private component generator against the newly integrated production context;
its original generated source and inputs remain in the frozen component
archives for any further analysis.

```text
python -X utf8 test2026/metal/prepare_interleaved_body_native_sources.py
python3 install_interleaved_body_native_sources.py ROOT
python3 launch_interleaved_body_native_checks.py ROOT ROOT/logs/interleaved-body-native-checks-20261015 checks
python3 queue_interleaved_body_native_profiles.py ROOT ROOT/logs/interleaved-body-native-20261015
# Only after successful terminal execution:
python3 archive_interleaved_body_native.py ROOT ARCHIVE
python -X utf8 verify_interleaved_body_native.py EXTRACTED_EVIDENCE VERIFICATION_JSON
python3 launch_confirm_interleaved_body_native.py ROOT ROOT/logs/interleaved-body-native-confirm-20261015 checks
python3 archive_interleaved_body_native.py ROOT COMBINED_ARCHIVE interleaved-body-native-20261015 interleaved-body-native-confirm-20261015
python -X utf8 verify_interleaved_body_native.py RESULTS confirmation-verification.json interleaved-body-native-20261015 interleaved-body-native-confirm-20261015
python -X utf8 summarize_interleaved_body_native.py RESULTS
```
