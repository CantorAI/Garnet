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
abort/recovery and unchanged prefill route. These are primitive results;
complete trained-model validation and actual serving measurements are pending.

Validation PID49634 is running all28 existing trained/reference/recovery stages
with the new native flag. Queued inference PID49699 requires that exact job to
finish successfully before six complete native ABBA model/batch profiles.
Both policies use the same latency bundle and VL B2 `[80]` plan. Only candidate
decode uses large-body packing. Actual body counts must be positive and stable
on decode steps and zero for prefill; the legacy library's API is not accessed
for the new counter. All external inputs, output caps, stops and complete
expected outputs remain frozen. Init, first-after-init and warm behavior stay
separate, including packing/memory costs. No vLLM ratio is inferred here.

Control library SHA256:
`9ee337d9db7d14bcff29a90970c4b5ac9021aee9f89a38a7f6f364051927d77b`.
The eight replaced remote files are preserved in
`logs/interleaved-body-native-original-source-20261015`. Installation happened
only after both matched comparator cohorts were terminal and independently
archived. The original Mac, GPT-OSS and Pi checkouts are preserved.

Do not modify source/drivers while either job is live. Do not rerun the old
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
```
