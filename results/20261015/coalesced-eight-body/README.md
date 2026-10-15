# Compact eight-output decode: initial exact-bit rejection; lexical follow-up

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

Current status, 2026-10-15: the initial trial compiled and executed on the Mac,
with72 numerical records and189 safety profiles. Every numerical-bound check
passed, but54 records differed from control output bits. All18 model/batch/
policy profiles were rejected; zero timing samples were admitted. The lexical
follow-up has the same54 exact-bit failures and is also rejected. No kernel or
full-model speed gain is claimed.

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

Controller PID88447 at `ROOT/logs/mlp-prefill-native-and-eight-finalize-20261015`
waits for exact upstream controller82980, then the actual native comparison PID
recorded by that controller. It archives the terminal native comparison and
requires the separate 72-complete-output audit before launching this probe.
Any upstream or audit failure stops dispatch. Native and component GPU work
cannot overlap through this sequence. Controller exit0 means probe dispatched,
not a successful probe or performance result. The probe's own quiet guard and
original numerical/timing admission checks remain mandatory.

```text
python3 finalize_mlp_profiles_and_launch_eight.py ROOT ROOT/logs/mlp-prefill-native-and-eight-finalize-20261015
```

## Executed initial trial and follow-up — 2026-10-15

The controller88447 sequence above stopped after the upstream27/28 coverage
audit failure. The later controller99421 sequence archived native PID99045
but stopped on an omitted-source archive error. The completed native
comparison was separately repaired and strictly audited before dispatching
this probe; see the [native report](../mlp-prefill-native/README.md).

PID3818 exited0 in19.352752s after its five-second quiet guard. Its private
Objective-C++ context and runtime Metal shader compiled and executed all
three original trained projection sets. The [independent audit](initial/verification.json)
verifies49 files,72 numerical records,189 safety profiles and **zero timed
executions**. Worst FP64-bound ratio0.00720305 and original-control envelope
ratio0.181313 are below1, but only18 records are bitwise equal (one-hot input
pattern);54 are unequal. Every policy has an unequal required pattern and
is excluded from timing. No tolerance or exact-bit requirement was relaxed.
The existing original-checkpoint CPU packing audit is reused, not rerun.
Archive SHA256:
`d2be45e8cf1bf01199c476e7f97393c067e1c562f2d1bcce3d6926a260557346`.

The indexed accumulator arrays may alter compiler arithmetic; that is an
unproven explanation. A follow-up duplicates the original lexical four-output
block body twice within the same K loop, with separately named accumulators
and shared activation loads. Control, assets, four patterns, thread policies,
numerical envelopes and mandatory exact bits stay the same. The generator
`make_unrolled_coalesced_eight_shader.py` is preserved in each source snapshot.

First follow-up PID4314 exited1 in11.658886s: the generated shader referred
to an undefined accumulator because sequential textual name replacements
rewrote a newly generated name a second time. The runtime compile failed
before numerical/timing records. It was archived before any source repair:
`7e28e46398ac77f619753dc800e497672667e50351a959aa7f137669d704ac7b`.
The factory now substitutes whole tokens once and checks that generated
accumulator identifiers are exactly the four declared names. This fixes a
generator bug; it does not explain the initial exact-bit differences.

Fresh PID4586 at `ROOT/logs/coalesced-eight-body-unrolled-fixed-20261015`
exited0 in19.247992s. The [50-file independent audit](unrolled/verification.json)
verifies72 numerical records,189 safety profiles and zero timed executions.
Again18 records are bitwise equal and54 unequal, so all18 profiles are excluded
from timing. Worst FP64-bound ratio0.00720305 and original-control envelope
ratio0.141144 remain below1. Removing the indexed accumulator array did not
restore exact bits; indexing alone is not an established cause. Reject both
variants for promotion without relaxing the gate. Corrected follow-up archive:
`bf10548d6e55ef01d8299e956fa1d9039519d98a66a026d07fe09bb8057c572f`.
Both shaders remain private. Native serving code/defaults are unchanged;
full goal unmet. The next TTFT experiment changes graph scheduling rather
than pursuing an unmeasured speed claim for these rejected kernels.
