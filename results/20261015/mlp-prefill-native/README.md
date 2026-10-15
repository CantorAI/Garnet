# Native dependent MLP graph prototype: validation in progress

Latest observation: resumed validation PID73896 at
`ROOT/logs/mlp-prefill-native-checks-resume-20261015` is live. Vision full-grid
and trained-head stages have passed; exact-tail cases are progressing. It
reuses the exact second attempt's configure/build/CTest evidence and pinned
library, then reruns all remaining original checks. No complete-suite or
performance result is established. The earlier snapshot below is historical.

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
| PID73896, resumed checks | Reuses the exact second build/CTest with unchanged library/source identities; all remaining stages rerun. Rejected outputs now move into their own attempt folder before retry. Live, not yet audited. | Remote log folder above; final archive pending |

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
