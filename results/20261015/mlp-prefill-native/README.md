# Native dependent MLP graph prototype: validation in progress

Status snapshot 2026-10-15: owned PID54555, remote log folder
`ROOT/logs/mlp-prefill-native-checks-20261015`. The observed driver reports
configure, build, CTest, vision-full-grid and screened-real-heads completed;
text exact-tail checks are progressing. No terminal record has been observed.
This is not a final archive, an independent validation audit, a performance
measurement or a claim that the entire suite passed. Read later dated journal
entries for subsequent state. No source consumed by this job may change
until terminal status and archival.

Hypothesis: replace the consecutive native gate/up projections, SiLU product
and down projection with one complete FP32 MPSGraph, targeting only the final
output, to reduce prefill submissions and intermediate handling. New option
`GARNET_METAL_MLP_PREFILL_GRAPH` defaults OFF. Decode is unchanged. The route
admits supported prefill shapes, uses FP32 placeholders and losslessly expands
the original BF16 weight bits; it adds real weight-storage/cache costs.
The [17-file manifest](source-manifest.json) identifies the exact staged
prototype. Eleven original files were backed up on the Mac before replacement.

Integration flushes prior native producers, then runs the graph synchronously
on the same Metal queue with explicit completion/error handling. Output remains
logically pending until native completion. This preserves simple ordering and
abort safety but introduces a CPU wait per MLP, which may erase component
gains. Graph creation is lazy: first-request compilation and extra FP32 weight
memory must be reported separately. The graph cache is capped at 256 entries.

The validation driver retains the original 28 numerical/recovery stages and
adds a primitive CTest covering pending inputs, foreign writers, uninitialized
reads, alias rejection, abort invalidation/recovery and row fallbacks. Expected
test coverage is not an executed passing count. No tolerance coefficients
are changed. Previous synthetic-reference failures remain in the linked
[component report](../stage-rounded-mlp-graph/README.md).

Host encode time now contains synchronous graph GPU work; it must not be
described as pure CPU encoding. The native first-start/last-end GPU span can
contain intercommand gaps. New MLP dispatch and partial-submission counters
must be reconciled against independently expected model/chunk counts.

```text
python3 launch_mlp_prefill_native_checks.py ROOT ROOT/logs/mlp-prefill-native-checks-20261015 checks
# Only after terminal status:
python3 archive_mlp_prefill_native_checks.py ROOT ARCHIVE
```

Next requirements: archive terminal success OR failure, independently audit
unchanged original model gates, compare all generated outputs against the
retained c186 native control, then run repeated native timing with init/first/
warm/RSS separated. A newly selected implementation needs a fresh matched
vLLM cohort. No graph integration speedup is established; full goal unmet.
