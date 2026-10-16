# Compact-aware independent decode projections

Status: actual initial component completes and passes original numerical checks; no learned group passes the prospective both-batch gain/null screen. No repeat or native integration.
This experiment targets remaining decode gaps: about0.8% text B1 and7.6% coder
B1 in the last repeated matched comparison. No inference speedup is established.

Hypothesis: put independent same-input projections in one dispatch while
retaining current lossless compact weight traffic and64-thread geometry. The
earlier [raw128 fused study](../../20261015/fused-projection-groups/README.md)
was mixed; it retained raw BF16 traffic and128-thread fused groups, with unused
SIMDs for small projections. Its measurements do not select this candidate.

Control: current native compact64 for eligible matrices, raw128 otherwise,
small64 for <=256 outputs. Candidate: a single64-thread dispatch with disjoint
projection ranges and unchanged separate output buffers. Compact projections
call the exact original compact dot-product body; small projections call the
original small body. Raw large projections use the original per-SIMD dot body
with group output stride8 instead of16 to cover every output with two SIMDs.
No split sums, reassociated reduction, SiLU fusion or concatenated weight copy.
The CPU packer checks raw fallback interleaving and every compact reconstructed
BF16 word. Wide exponent blocks retain original raw fallback. Dense special
FP32-key matrices retain their original raw decode route. Compile Metal3.1,
matching the retained native library, with original default arithmetic options.

Scope: actual layer-zero MLP gate/up for all three models; Q/K/V for text and
coder; irregular read-only pairs; B1/B2. These are learned weights with changed
synthetic inputs, not captured full-model activations. Every original output
and padding bit must match control across four input scales including signed
zero, with the original independent FP64 gamma(ceil(K/32)+32)*sumabs+1e-7 bound.
Any numerical failure stops the candidate; preserve full failure evidence.

Prospective timing design:2s alternating common warmup; ten complete ABBA
rounds, five samples per window, three window warm calls; retain all samples.
Mode2 is the same control pipelines, bindings and geometry null; mode1 is fused.
No frequency, power or thermal-cause claim is inferred from a null comparison.

Prospective selection: consider a learned model/group only when BOTH B1/B2
improve GPU AND wall time by at least3%, its own null GPU AND wall fractions
stay within5%, every original numerical group passes, and one complete unchanged
confirmation passes the same rule. Report all groups, including negative and
irregular results. No blanket fusion policy from one favorable shape. Kernel
compile/packing and allocation costs are outside these component timings and
must be reported; any later native implementation must reuse existing weight
buffers, validate ownership/fallbacks and complete all full-model profiles.
No full-model or matched-vLLM claim follows from component selection.

The launcher requires the current vector-native inference chain to be terminal,
archived and audited before compilation/GPU execution, then verifies a quiet
exclusive interval. This study must not overlap the running numerical or
performance jobs. Do not mutate any source/driver consumed by those jobs.

Local generated shader/harness and expanded Python runner parse successfully.
No compile, actual packing, GPU numerical or performance result is claimed yet.

To preserve the actual retained control, this component rejects an otherwise
interleaved-only matrix with unsupported K alignment or zero compact blocks;
it does not silently substitute raw128 for an interleaved-only native control.
Native integration would still need that fallback. Weight load/packing/buffer
allocation duration and buffer byte counts are emitted separately from timings;
output allocation is not included in that weight setup duration. No RSS saving
is inferred from these logical buffer byte counts.

Queue PID73358 watches the exact live vector continuation PID67688 at
`logs/vector-vision-native-continuation-20261016`. It waits for successful
terminal/archive/audit completion before the quiet guarded launch. It archives
each terminal component success or failure, independently audits success and
applies the prospective selection rule before allowing one unchanged complete
confirmation. No inference/GPU job is launched by queue creation. All seven
new consumed helper files are frozen while the queue is live. No existing
source or driver used by the current numerical job was modified.

Commands (ROOT is `/Users/zsdx/garnet-metal-investigation`):

```text
python3 ROOT/source/test2026/metal/queue_compact_fused_projection_groups.py ROOT ROOT/logs/compact-fused-projection-groups-queue-20261016 67688
```

The controller records the exact expanded compile/run/archive/audit commands
when dispatched. Frozen raw data, numerical gates, complete ABBA windows and
negative outcomes must be retained before any repair or next candidate.

## Dependency resumed after preserved upstream audit-path failure

Original queue73358 exits1 before launching any component: native inference
completed, but its auditor expected a wrong expanded-script filename. The
terminal controller/queue failure is preserved in the vector-native45-file
252aeda0... archive. The repaired audit passes on Mac/Windows without changing
or rerunning the native initial benchmark. Native initial passes its prospective
screen and unchanged confirmation is running under new controller93884.

New queue94685 at `logs/compact-fused-projection-groups-queue-repaired-20261016`
watches exact93884 using new dependency-only wrappers. Original component
runner, shader generator, precision/gain/null gates, cases and sampling are
unchanged. No GPU experiment has yet executed here; all old failure statuses
remain preserved. Repaired helper files freeze while queued.

## Actual initial result: numerical pass, no selection

Component97616 exits0 in51.396588375s. Repaired queue94685 exits0 after
terminal archive, independent Mac raw audit and predeclared decision. Archive
33 files, SHA256:
`28e0d5b2ca20a30347368c47ed88eb7718b4073fd3a2a26dd7509a9f38776548`.
All16 group/batch original output/padding-bit checks and unchanged FP64 bounds
pass;6400 timings remain. All eight groups and all null windows are retained.
Windows artifact/raw audit is pending transfer; no model or vLLM gain follows.

No learned group qualifies in BOTH batches. Text B1 gate/up GPU/wall fractions
1.005035/1.003054, B2 .931431/.954678; coder B1 .969690/.956895 but its
null GPU fraction .913587 fails the5% gate. Coder B2 .925013/.951680 passes
its own null, which alone does not select the pair. VL gate/up B1
1.001461/.997875, B2 .985406/.993170 misses the3% gain rule. Attention groups
also fail gain/null requirements. Do not normalize by the null, filter windows
or attribute drift to an unmeasured hardware cause. No favorable unchanged
repeat, selected group, native implementation or decode target-count update.

Windows archive/raw regeneration audit also passes33 files,16 group/batch
checks,104093 validated executions,6400 timings and worst original FP64
fraction0.00902619627882109. [Raw audit](initial/verification.json),
[full negative decision](decision-initial.json). No integration or repeat.
