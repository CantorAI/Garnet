# Native exact BF16 block storage: validation running

This OFF-default integration follows two successful private component trials
with144 exact-bit/FP64 checks,378 safety profiles and2304 timed submissions.
The selected common64-thread policy reduces component wall time in all six
model/batch groups in both runs. These are not complete-model speed results.

`GARNET_METAL_COALESCED_BF16_BODY` requires the existing interleaved body
fallback. Only decoder matrices with256<N<=8192, K divisible by128 and at
least1048576 values are candidates. Four outputs by128 K terms share a block.
Compact blocks preserve all original BF16 bits in byte/nibble planes; wide
exponent blocks read the original raw four-output buffer. Every compact value
is recovered and checked on the CPU before publication. Original BF16 masters
and exact-half/MPS prefill routes remain. This preserves model precision and
FP32 decode arithmetic; it does not reduce learned-value precision.

The native MSL payload is byte-identical to the twice-tested private kernel.
The native policy fixes64 threads for all eligible model/batch shapes. The
actual encoded route has a context counter, published through the session
only after submission and reset at begin/abort/session reset/clear/failure.
Frontend collection checks the backend marker before reading the new API,
preserving compatibility with the unchanged legacy control library.

Actual Mac configure/build succeeded. All seven CTests passed in6.06s,
including the new80-case original-bit/FP64 suite, large/tail/fallback shapes,
ownership, uninitialized input, abort/recovery and unchanged prefill routing.
Separate CPU compiler checks also pass with the new flag off and both body
flags off. `primitive-ctest.log` and `disabled-compile.json` retain these
preliminary results. Full vision-grid and51 trained-head checks also passed.
Complete trained-model validation and serving measurements are still pending.

A separate CPU-only audit checks byte/nibble reconstruction across every
checkpoint matrix matching the decoder-layer name and shape admission:
168 text,72 coder and138 VL matrices. All compact values recover exactly.
Compact block fractions are96.44%,95.99% and96.31%, respectively. The added
one-copy buffers estimate to591.06,451.21 and705.66MiB. `checkpoint-coverage/`
preserves the executed source and report. These are checkpoint counts and
one-copy storage estimates, not proof of graph usage/sharing, actual GPU
allocations, process RSS or performance. The upcoming native run measures
the real initialization and memory costs.

Validation PID88930 is running the full28 original trained/reference/recovery
stages. Queue PID88983 requires that exact handle to finish successfully and
all28 stage records to pass before six complete native ABBA profiles. No other
inference or GPU component should be started during either job. Do not modify
the pinned native sources or drivers until both jobs are terminal and archived.
Init, first-after-init, warm latency, complete outputs and peak process memory
must all remain separate in the resulting evidence.

Native comparison control SHA256:
`9ee337d9db7d14bcff29a90970c4b5ac9021aee9f89a38a7f6f364051927d77b`.
Actual newly built candidate SHA256:
`c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1`.
Both policies use the same latency bundle and VL B2 `[80]` chunk plan. Only
the candidate enables the new block storage and existing interleaved decode.
No new vLLM ratio or whole-model gain is established here.

Eight replaced files are preserved remotely under
`logs/coalesced-body-native-original-source-20261015`; the12-file manifest is
`coalesced-body-native-staging/manifest.json`. Original identities come from
commit `eaf421d160c04ab0e5259345123279c07678d278`. The original Mac checkout,
GPT-OSS checkout and Pi checkout are preserved. Never rerun the old private
context generator against this now-integrated source; its original source
and generated code remain in the frozen component evidence.

```text
python -X utf8 test2026/metal/prepare_coalesced_body_native_sources.py
python3 install_coalesced_body_native_sources.py ROOT
python3 launch_coalesced_body_native_checks.py ROOT ROOT/logs/coalesced-body-native-checks-20261015 checks
python3 queue_coalesced_body_native_profiles.py ROOT ROOT/logs/coalesced-body-native-20261015
# Only after both exact jobs finish successfully:
python3 archive_coalesced_body_native.py ROOT ARCHIVE
python -X utf8 verify_coalesced_body_native.py RESULTS VERIFICATION_JSON
```

Extra compact buffers and packing work can increase initialization and peak
memory. Whole-model memory and performance effects remain unproven. The full
three-model2x decode / half-TTFT goal remains active and unmet.
