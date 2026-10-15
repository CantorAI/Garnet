# Shape-selected two-output native integration: validation running

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
that abort clears the added counters. The fresh build and seven CTests have
completed; all28 trained/native stages are still running under exact PID42435.
No complete validation or full-model performance result is claimed yet.

Controller PID47970 polls that exact handle. After a terminal record it freezes
success or failure, requires an independent frozen-evidence audit of all28
original stages, then dispatches native ABBA only after success. A missing
handle without terminal evidence is an error; observation timeouts do not
restart validation. Controller success means a benchmark was dispatched,
not that it passed or that the performance goal was achieved.

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
