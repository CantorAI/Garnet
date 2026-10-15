# Hybrid native special-key / combined graph prefill: queued

Owned PID35785 waits for exact matched confirmation PID34285 at
`ROOT/logs/coalesced-body-vllm-confirm-20261015`, requiring successful terminal
execution and five quiet seconds before any compile/GPU work. No competing
inference or GPU component may overlap either job.

The diagnostic isolated observed graph failures to text/coder special keys.
This fixed policy retains those original native routes at both batches and
graphs every other projection. All29 original matrices, four input patterns,
FP32 values, row shapes and bounds remain. Keys are not omitted: their native
submission runs inside every candidate wall interval. Fallback outputs must
match the original native control bit-for-bit.

Policy1 uses FP32 placeholders; policy2 original FP32 constants. Candidate
calls include native key execution/synchronization then the synchronous graph
call. VL has no such key and needs no extra native submission. First call and
setup remain separate; added storage/fallback costs are real. No memory saving
is claimed. No original source, workload or tolerance is changed.

Expected coverage:48 checks,12 first calls, up to12 admitted timing profiles
with768 submissions. These are expected counts, not executed results yet.
Failed policies must have no timed windows. This private independent
projection set is not a complete dependency layer or a model TTFT result.
Integration requires repeat benefit and full trained-model validation.
The full2x decode / half-TTFT goal remains unmet.

```text
python3 launch_hybrid_combined_graph_prefill.py ROOT ROOT/logs/hybrid-combined-graph-prefill-20261015 component
# Only after exact job termination:
python3 archive_hybrid_combined_graph_prefill.py ROOT ARCHIVE
python -X utf8 test2026/metal/verify_hybrid_combined_graph_prefill.py RESULTS RESULTS/verification.json
```
