# Native dependent MLP graph: correct outputs, rejected for latency and memory

Latest audited result, 2026-10-15: all 28 required validation stages are
accounted for, and native ABBA PID99045 completed six profiles with 72 exact
complete outputs. Warm TTFT regresses in all six profiles, first-after-init
TTFT increases substantially, and peak process RSS rises by 0.85–1.11 GiB.
Reject this synchronous graph integration for promotion; the option stays OFF.
This is one native cohort, not a new matched vLLM comparison or replication.
The snapshots and controller descriptions below are historical; see the final
dated correction for coverage reuse, failed audits and archive repair.

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
The [current 18-file manifest](source-manifest.json) identifies the staged
prototype and repaired guard. The initial 17-file manifest is preserved in
the first failed archive. Native numerical sources are unchanged by the
guard and retry repairs. Eleven original files were backed up initially;
the guard repair separately preserves all three replaced helper sources.

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

## Interrupted attempts and retry repair — 2026-10-15

| Attempt | Actual outcome | Frozen evidence |
| --- | --- | --- |
| PID54555, original checks | Stopped deliberately with SIGINT after the guard blocked on a changed WorldSense supervisor fingerprint; exit 1, 969.807181 s. Nine stages completed, including 8 CTests and both sets of 8 exact-tail cases. Full suite incomplete. | [1,291-file verification](failed-guard/verification.json), [manifest](failed-guard/evidence/manifest.json) |
| PID71863, fresh guard-fixed build | Exit 1, 82.209367 s. Configure/build/8 CTests passed. Vision attempt 0 was guard-rejected on PID73159; attempt 1 failed with FileExistsError because the partial output directory remained. PID73159 provenance is unproven. | [1,143-file verification](failed-retry/verification.json), [manifest](failed-retry/evidence/manifest.json) |
| PID73896, resumed checks | Exit 0, 1292.240134 s, but only 27 required stages were present. The strict 28-stage audit correctly blocked benchmark dispatch. | [1,465-file archive verification](incomplete-27/archive-verification.json), [raw manifest](incomplete-27/evidence/manifest.json) |

The verified WorldSense supervisor SHA256 is
`115a1d9dd0ae7181af4876e57bda55b9948d31c1febf1a12d40b150bce0c622e`.
Its only Garnet marker is the exact forwarded budget environment name, and
diagnostics found no loaded inference/GPU libraries in the supervisor/child.
The guard adds only this inspected source fingerprint to its existing list;
unknown workers and loaded inference/GPU libraries remain rejected on every
check. No foreign process was stopped. Rejected timings are not admitted.

First failure archive SHA256:
`eada52786b8f2072c83c810d542823ff58ad9519f07c18b6fa526b5f989465fc`.
Second failure archive SHA256:
`6d9fda15edcd9bf2e101918ed7615925b8f3d5f98cfc016ec34cd1da806b2147`.
Pinned second-build library SHA256:
`23bff0f76a303f2af78c992d4af6e4f90d5041c2e8c57bf454f924b15ac37081`.

The resume preserves partial artifacts with an explicit relocation map and
retries the unchanged command. It does not reuse interrupted numerical
results, loosen a guard or change numerical bounds. The three reused
configure/build/CTest stages must be disclosed when reporting coverage.

```text
python3 launch_resume_mlp_prefill_native_checks.py ROOT ROOT/logs/mlp-prefill-native-checks-resume-20261015 checks
# After the exact resumed job is terminal:
python3 archive_mlp_prefill_native_checks.py ROOT ARCHIVE
python -X utf8 test2026/metal/verify_mlp_prefill_native_checks.py RESULTS RESULTS/verification.json ARCHIVE
# Only after successful archive audit and identity checks:
python3 launch_mlp_prefill_native_profiles.py ROOT ROOT/logs/mlp-prefill-native-20261015
```

Dependent controller PID82980 is waiting at
`ROOT/logs/mlp-prefill-native-finalize-20261015` for exact validation PID73896.
It requires successful terminal status, unchanged source identities, a frozen
archive and the separate full validation audit before dispatching native ABBA.
Any failure stops dispatch. Controller exit0 means benchmark dispatched only,
not a passing benchmark or performance goal. Raw archive/audit/launch commands
and logs are saved under the controller folder. Sources remain frozen through
the dependent comparison and its terminal archive.

A second controller PID88447 waits for that dispatch and exact native job,
then archives and audits the full comparison before admitting the independent
compact-eight decode probe. The native comparison audit uses
`verify_mlp_prefill_native_comparison.py`. Static review found that the earlier
auditor compared the original-check archive's file count with the larger
combined archive's count. The original auditor remains frozen. Its replacement
separates these two bookkeeping counts, checks every combined-archive file,
and preserves equality of every numerical/coverage/source/library field. No
executed numerical failure is waived, and no tolerance is changed.

## Completed validation and rejected native comparison — 2026-10-15

The resumed generated driver omitted `wide-batch-trained-reference`, which
the original outer wrapper had appended. Controllers82980 and88447 exited1;
no native benchmark was dispatched by that failed sequence. Preserve the
[15-file controller/source archive](failed-coverage-audit/archive-verification.json).
Original27 archive SHA256:
`61d83b39924e0093104b01850979d1725b9e00afe2ec5875d1f8d4d91cd82153`.
Controller failure archive SHA256:
`f9584ee67250735eb50950afdfd44e3b9e3298e1e3d9e499676e387b5dea1e78`.

Completion PID97189 exited0 in101.555720s. It copied the27 frozen stages,
checked the unchanged18 source/library identities, and ran the missing wide
reference freshly: four trained profiles, twelve carried steps, worst
original-bound ratio0.543284. This is **27 reused stages plus1 fresh stage**,
not28 newly executed stages or a replication. Configure/build/CTest were
already reused from PID71863. The [complete strict audit](checks/verification.json)
checks all28 stages,8 CTests,64 primitive cases,51 head queries/78 rows,
4 vision-grid profiles and both sets of6 recovery cases. Completed archive:
`be33c99ce73243473e34d873d559b0efc8ef6d3ca8d8ae8d152ca94b852aca32`.

Controller99015 admitted native PID99045 only after the strict audit. The
native job exited0 in211.133414s. Four fresh worker positions per profile
use control/candidate/candidate/control, each with one first-after-init and
two warm requests. All72 complete outputs match the original expected tokens,
prompts and stops. Actual graph counts match independently derived layer/chunk
counts; decode graph counts are zero. Both builds use the same compact decode
policy; candidate changes prefill only. Candidate SHA remains23bff0f7...;
control SHA remainsc1869309....

| Model | Batch | Warm TTFT control/candidate ms | Decode candidate/control | First-after-init TTFT control/candidate ms | Extra peak RSS GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 35.865 / 56.815 | 0.982 | 136.481 / 1924.310 | 0.980 |
| Qwen3-0.6B | 2 | 56.495 / 78.575 | 0.976 | 146.984 / 1917.044 | 1.001 |
| Qwen2.5-Coder-0.5B | 1 | 28.072 / 45.035 | 0.983 | 104.172 / 2176.963 | 1.090 |
| Qwen2.5-Coder-0.5B | 2 | 43.784 / 58.622 | 0.957 | 118.319 / 2184.813 | 1.106 |
| Qwen3.5-0.8B VL | 1 | 174.099 / 184.942 | 0.974 | 277.396 / 2055.927 | 0.851 |
| Qwen3.5-0.8B VL | 2 | 296.096 / 327.985 | 0.991 | 390.321 / 2192.920 | 0.898 |

Initialization means range2.82–5.59s for control and2.83–5.59s for candidate;
lazy graph compilation is charged to the first request. Caches were not
flushed; these are not hardware-cold measurements. RSS is the mean process
peak per policy, not instantaneous allocation or system-wide memory. The
same decode implementation nevertheless measured0.9–4.3% slower; this cohort
does not isolate the cause. Synchronous per-MLP waits and expanded weight
storage are known costs, but their individual latency contributions are not
separately measured. No fresh vLLM ratio is derived from these numbers.

Controller99421 archived the native result, then the audit stopped: the
suffix whitelist omitted13 `.metal` files from both source and snapshot,
although all26 hashes were recorded. The original2,978-file archive remains:
`4c54a571ab781c9635943839c7fb341db27bd18632d1b7d52f329467fb63f852`.
The [original archive proof](comparison-original/archive-verification.json)
checks bytes only; it is not a successful complete-source audit.
`complete_mlp_native_source_archive.py` adds only the exact frozen source
bytes and terminal controller diagnostics, preserving all original records.
Corrected3,012-file archive:
`9d494b2795485f03def18f463673b28da754fb6b5b99cce8db6c0ac96f7c5242`.

The next audit reached the final asset check but incorrectly looked for
tokenizer JSON in `omitted`; tokenizer bytes are in `files`. The repaired
comparison auditor verifies those actual bytes against the recorded asset
hash, preserves checkpoint hash assertions and every numerical/output gate,
and verifies the pinned original manifest plus each added source hash.
It then passed all72 outputs and1,131 source-closure identities. Neither
bookkeeping correction changes a numerical bound or an inference result.
The complete-source evidence and audit are recorded in `comparison/`.

Only after this successful audit, the separate compact-eight decode probe
PID3818 was dispatched through its quiet-resource guard. It exited0 in
19.352752s; independent component archival/audit is pending at this snapshot.
The full2x decode/half-TTFT target remains unmet, and c186 is retained.
