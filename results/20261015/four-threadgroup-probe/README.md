# Original four-output threadgroup tuning: prepared, not executed

Hypothesis: the existing four-output compact kernel may prefer different
threadgroup sizes for individual projection shapes. Unlike the rejected
two-output candidate, this changes no shader, pipeline, payload, arithmetic,
lane reduction or weight representation. A private context hook changes only
threads/group and corresponding outputs/group. Control always uses64 threads;
candidates use64/128/256. Candidate64 is an explicit null GPU-work comparison.
CPU hook/counter overhead remains inside measured wall time.

The original20 selected matrices are measured individually at physical rows1/2,
with all29 learned assets still checked by the unchanged CPU packing admission
and runtime roundtrip proof. Four patterns, original exact output bits, original
FP64/native bounds and189 ownership/fallback/abort safety profiles are retained.
Expected counts are480 numerical records and7,680 timed submissions per cohort.
Failed numerical policies receive no timing windows. These are expected counts,
not an execution result. All weights remain resident; repeated single-projection
cache behavior differs from full inference.

Predeclared selection rule: require GPU time fraction<=0.95 and wall fraction<1
in BOTH initial and confirmation cohorts for every observed duplicate matrix
sharing a geometry. Its64-thread null GPU/wall fractions must stay within5%
of1 in both cohorts. Pick the qualifying thread count with minimum worst GPU
fraction; keep original64 elsewhere. This is a conservative selection heuristic,
not a confidence interval. Do not rerun rejected settings seeking a favorable
sample. Any selected table still needs fresh full-model validation/confirmation
and repeated matched vLLM measurements before a target claim.

Private route counters assert actual tuning selection, including the64-thread
null. There is no additional compiled Metal shader; a retained header copy is
archived solely to verify original source identity. Local context/header/harness
generation and expanded runner syntax checks pass. The separate frozen auditor
materializes every original packing/numerical/safety/raw-window assertion from
the prior verifier stack and regenerates the new private sources from snapshots.
Its SHA256 is
`f2007893419f58f0d0cff50afc19f6be89873747acc100dea5b830c2fd2478de`.
No Mac compilation, GPU execution or speed is claimed yet.

```text
python -X utf8 test2026/metal/freeze_four_threadgroup_verifier.py
python3 launch_four_threadgroup_probe.py ROOT ROOT/logs/four-threadgroup-probe-20261015 component
# Only after exact terminal capture:
python3 archive_four_threadgroup_probe.py ROOT ARCHIVE [TRIAL]
python -X utf8 test2026/metal/verify_four_threadgroup_probe_frozen.py AUDIT AUDIT_RESULT
```

ROOT is /Users/zsdx/garnet-metal-investigation. The launcher requires the last
BF16 component to be terminal, archived and audited, then its own quiet guard.
Sources remain frozen until terminal archival. Existing source/build defaults
and the protected GPT-OSS/Pi checkouts remain unchanged. Full goal remains unmet.
