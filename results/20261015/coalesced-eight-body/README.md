# Compact eight-output decode: prepared, not executed

Hypothesis: share each activation float4 across two adjacent original
four-output compact BF16 blocks, reducing activation loads and scheduling
overhead without changing weight bits, packing, or each output's FP32
accumulation order. This differs from the earlier
[raw eight-output layout](../wide-body-decode/README.md), which changed storage
and gave mixed/regressive results. Neither that experiment nor the retained
four-output block gains establish a gain for this new kernel.

The control is the current compact four-output kernel at 64 threads, not the
older raw BF16 layout. Candidate policies use 64/128/256 threads and eight
outputs per SIMD group. Both policies share the existing payload, raw
fallback, matrix admission threshold and weight allocations. Candidate-only
dispatch counters are asserted in the generated harness. Decode only; no
native source or serving default changes. The source factory generates a
private context/header and harness, leaving the live MLP validation untouched.

Current status: source generation and Python AST checks pass locally. Neither
the Objective-C++ probe nor the Metal shader has compiled or executed on the
Mac. No GPU correctness, latency, memory or speed claim is made.

Planned gates preserve all 29 original trained projection matrices (7 text,
7 coder, 15 VL), four input patterns, batch 1/2, exact native output bits,
the original FP64 gamma envelope and the existing native-comparison bound.
The tail/fallback/prefill, pending-output, foreign-writer, uninitialized-read,
abort and recovery cases remain. An independent frozen CPU packing audit is
reused only after original asset/checkpoint hash agreement. Reusing that audit
does not constitute another CPU replication.

Expected maximum coverage is 72 numerical records, 189 repeated safety
profiles and 1,152 timed submissions; none is an executed count yet. Only
policies passing every numerical gate may enter two ABBA rounds with eight
samples per position. One submission covers the entire independent projection
set. Common allocation/packing/input upload is outside timing. Extra compiled
pipeline/setup costs, first-call costs and process RSS are not measured by
these component wall/GPU clocks. No memory-saving claim follows from retaining
the same weight buffers.

```text
# Run the guarded component only after the current exact owned job is terminal
# and its evidence archived; no inference/validation may overlap this probe.
python3 run_coalesced_eight_body.py ROOT ROOT/logs/coalesced-eight-body-20261015/jobs
python3 archive_coalesced_eight_body.py ROOT ARCHIVE
python -X utf8 test2026/metal/verify_coalesced_eight_body.py RESULTS RESULTS/verification.json
```

A clean launcher/resource guard is required before executing this invocation
template. Selection requires repeated component evidence, unchanged original
model gates, full generated-output comparison and fresh repeated matched vLLM
measurements. The full 2x decode/half-TTFT goal remains unmet.
