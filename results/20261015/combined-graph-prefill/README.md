# Combined projection graphs: queued prefill investigation

This private experiment targets the remaining TTFT gap. The earlier
single-projection MPSGraph study synchronized every matrix separately and
showed no broad gain. This version compares one native command submission
for the full independent first-layer projection set against one synchronous
multi-result graph call. Policy1 supplies FP32 weight placeholders; policy2
embeds the original learned values as FP32 constants, allowing graph compile
and weight preparation to occur before warm windows.

All29 original first-layer matrices are included, including special key and
non-common-alpha routes. Native controls preserve their existing flags and
kernels. Learned BF16 values expand exactly to FP32 for graph storage. Graph
activation, weight, product and output tensor dtypes are asserted FP32. There
is no input deduplication, prefix reuse, speculative work or vision caching.
This is an independent projection set, not a complete dependency layer,
fused MLP or full model. Both policies must pass four original FP64/reference
input patterns before timing; original bounds are unchanged. Failed policies
must have no timed windows. Each admitted profile has two ABBA rounds and
eight wall samples per position. First graph execution is recorded separately.

Actual Mac Objective-C++ syntax checking passed. GPU execution is pending.
Owned queue PID6026 waits on exact live native validation PID88930 and native
benchmark PID88983, requires both successful terminal results, then a verified
quiet interval. Output is `ROOT/logs/combined-graph-prefill-20261015`.
It must never overlap inference with either native job. Do not edit its queued
source, driver or dependencies until terminal execution and archival.

The native control is the current newly built library's retained prefill
implementation; its coalesced decode flag is inactive at these prefill rows.
Logical rows remain24/44 text,26/52 coder and80/160 VL. Wall clocks include
encoding and synchronization; a native command GPU span must not be reported
as graph GPU time. Initial compilation, weight copies and larger constant
storage remain outside warm timing and require separate reporting.

No correctness, full-model TTFT, decode speed or comparator improvement is
established yet. The full2x decode / half-TTFT goal remains active and unmet.

```text
python3 launch_combined_graph_prefill.py ROOT ROOT/logs/combined-graph-prefill-20261015 component
# Only after the exact job is terminal:
python3 archive_combined_graph_prefill.py ROOT ARCHIVE
python -X utf8 verify_combined_graph_prefill.py RESULTS VERIFICATION_JSON
```
