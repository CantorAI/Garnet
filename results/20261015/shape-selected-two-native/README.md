# Shape-selected two-output native integration: correct, rejected for promotion

This private candidate uses the [three selected geometries](../two-matrix-probe/README.md)
and retains the current four-output compact kernel for every other shape.
Original BF16 learned bits, compact payload, FP32 arithmetic, input lane order,
original fallback and all prefill paths are unchanged. The rejected MLP graph
route remains disabled. This is a decode experiment; it does not address the
remaining first-token-latency gap by itself.

The [preparation record](preparation.json) freezes1,275 copied source files.
Eight existing copies change: context/header, session/header, backend marker,
two benchmark workers and the coalesced primitive. One generated shader is
added. The current retained Mac source/build and protected GPT-OSS/Pi checkouts
are not modified. New private tree:
`/Users/zsdx/garnet-metal-investigation/shape-selected-two-investigation-20261015`.

The primitive keeps all80 original cases and adds24 cases for selected shapes.
It verifies actual dispatch counts, raw original output bits, unchanged FP64
envelope, ownership, abort/recovery and decode-only routing. It also checks
that abort clears the added counters. Exact PID42435 exited0 in1423.387570s.
All28 stages were freshly executed, including seven CTests and104 primitive
cases. The separate Mac archive audit passed1,615 manifest files,51 trained
head queries/78 row states, four vision profiles, both six-case recovery suites
and four wide profiles/12 carried steps. No compile or validation stage was
reused. Worst wide original-bound ratio is0.543284. Local byte verification and the complete numerical audit have also passed.

Validation archive SHA256:
`5b9ee0dc45c59b041fba1e006dbd83e3b383950d577d02bcf3fa5a16b23f94b2`.
Candidate library SHA256:
`c22fb8f238ab995fb0d5e88ca504739068b557da566e857d0ea0e7436b0bedcb`.
No full-model speed gain is established yet.

Controller PID47970 polls that exact handle. After a terminal record it freezes
success or failure, requires an independent frozen-evidence audit of all28
original stages, then dispatches native ABBA only after success. A missing
handle without terminal evidence is an error; observation timeouts do not
restart validation. Controller success means a benchmark was dispatched,
not that it passed or that the performance goal was achieved. PID47970 has
exited0 with state native_benchmark_dispatched after the successful archive
audit. Actual native worker PID68805 exited0 in192.245289s. All six profiles
completed and both Mac and Windows frozen-evidence audits passed all72
complete outputs and actual route predicates.

The prepared benchmark uses all six original model/batch profiles and72
complete generated outputs, sequential control/candidate/candidate/control
workers, one first-after-init and two warm requests per worker. It independently
derives expected selected-route counts from original checkpoint tensor headers.
Both source closures, learned assets, commands, monitors, first/warm/init/RSS
and all complete tokens must remain auditable. The retained c186 native library
is the performance control; the original73c10e5d trained reference has a
different validation role. No fresh vLLM ratio follows from native comparison.

```text
python3 prepare_shape_selected_two_body.py ROOT ALT
python3 ALT/source/test2026/metal/launch_shape_selected_two_checks.py ALT ALT/logs/shape-selected-two-checks-20261015 checks
python3 ROOT/source/test2026/metal/finalize_shape_selected_two_checks.py ALT ALT/logs/shape-selected-two-finalize-20261015
# Controller steps, only after terminal capture:
python3 archive_shape_selected_two_checks.py ALT ARCHIVE
python3 verify_shape_selected_two_checks.py AUDIT AUDIT_RESULT ARCHIVE
python3 launch_shape_selected_two_profiles.py ALT ALT/logs/shape-selected-two-native-20261015 profiles
# After native terminal capture:
python3 archive_shape_selected_two_profiles.py ALT NATIVE_ARCHIVE [TRIAL]
python -X utf8 test2026/metal/verify_shape_selected_two_profiles.py NATIVE_AUDIT NATIVE_AUDIT_RESULT [TRIAL]
```

ROOT is `/Users/zsdx/garnet-metal-investigation`; ALT is the private tree
above. Templates supplement actual job command records. Sources/drivers stay
frozen until the owned jobs are terminal and archived. Selection used both
component cohorts, so fresh full-model confirmation and fresh repeated matched
vLLM measurements remain necessary. Full goal is active and unmet.

Follow-on controller53701 waits for exact controller47970 and then the actual
native worker handle. It freezes/audits the completed native comparison before
admitting a [separate BF16-storage graph primitive](../mlp-bf16-feed-primitive/README.md).
It does not overlap inference and does not change any consumed runtime source.

## Actual full-model comparison and rejection — 2026-10-15

The5,094-file native archive SHA256 is
`f18be8192eca6f2f7cd51590db9ceae68b2e70e7d3ce602e75216a02edb501eb`.
The audit covers1,199 serving-source identities and980 external-driver
identities, all72 complete outputs, original prompts/stops/caps, fresh image
encoding and actual selected-route counts. [Full native audit](comparison/verification.json),
[all28 fresh validation stages](checks/verification.json),
[route-aware decision](decision-analysis.json).

All ratios below compare this candidate with retained native c186. They are
not new vLLM ratios. Init, first-after-init and process RSS remain separate in
the full audit; caches were unflushed. 'Two dispatches' counts actual selected
kernel executions per decode step.

| Model | Batch | Two dispatches | Decode control / candidate tokens/s | Warm TTFT control / candidate ms | Decode ratio | TTFT fraction |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 1 | 0 | 72.8901 / 74.2688 | 36.1694 / 35.4661 | 1.018915 | 0.980555 |
| Qwen3-0.6B | 2 | 0 | 70.8262 / 70.5110 | 56.6618 / 57.3235 | 0.995550 | 1.011679 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 0 | 87.4344 / 88.0954 | 27.1829 / 26.9526 | 1.007560 | 0.991526 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 24 | 84.2726 / 82.8169 | 43.4720 / 44.1669 | 0.982725 | 1.015984 |
| Qwen3.5-0.8B | 1 | 6 | 60.8200 / 61.3327 | 174.9501 / 182.8912 | 1.008431 | 1.045391 |
| Qwen3.5-0.8B | 2 | 18 | 55.1807 / 54.3734 | 297.3130 / 295.7617 | 0.985370 | 0.994782 |

None of the three workloads actually executing selected kernels improved both
native decode and warm TTFT. Coder B2 decode regressed~1.7%, VL B2~1.5%, and
VL B1's~0.8% decode improvement accompanied~4.5% higher TTFT. Positive changes
in text B1/coder B1 cannot establish benefit from executing the selected kernel:
their actual selected dispatch count is zero. Reject promotion; retain c186.
This single mixed native cohort does not justify a fresh vLLM target claim or
a favorable-repeat search. The prior repeated matched chart remains unchanged.
