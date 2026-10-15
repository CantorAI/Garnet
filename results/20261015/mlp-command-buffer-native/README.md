# Command-buffer MLP: full original validation running; no timing result yet

The [private primitive and dependent-chain probes](../mlp-command-buffer-primitive/README.md)
passed the64 original plus64 additional dependent two-graph cases. The next
gate uses all28 original trained/native validation stages, then six complete
model profiles against retained native c186. The full2x decode/half-TTFT goal
versus matched vLLM remains unmet; this report makes no new speed claim.

Actual validation PID5961 runs in the separate tree
`ROOT/mlp-command-buffer-investigation-20261015`. Configure/build/8 CTests,
vision full-grid and trained real heads have passed; text exact-tail cases
are progressing. Terminal evidence, frozen archive and independent full audit
are pending. All consumed copied sources remain frozen through archival.

The [preparation record](preparation.json) records1,240 copied source files
and exactly three transformed copies. Context SHA256:
`8eefbc1e4ce2176284fdfebde71691a454b83ac6558db867e07053a8fce9d2f7`.
Graph header SHA256:
`48000f5890f60f4fbddfb9839a7da0f0cc92373d17c0e17afb1619f376551ca8`.
Primitive test SHA256:
`033560bd1bf725c7b71e57fee14af5d3a662430092eb37dbdecb01cd7f9993e1`.
Fresh candidate library SHA256:
`612af1059c3f86c5d7d5affe3223065b3457c1036bf10464ea3a763672a7e549`.
Existing source/build/model trees remain unchanged; historical dependencies
are linked for workflow reads, while this candidate build/output is separate.

Numerical validation retains the original trained-reference control library
73c10e5d... from build-resident-kv-inplace. The subsequent performance control
is the retained compact-body c1869309... library. These serve different roles;
the old trained-reference library is not used to inflate a performance ratio.
No reference bound, learned value, arithmetic precision, prompt, output cap,
stop token, image preprocessing or batch profile is changed.

Controller13152 at `ALT/logs/command-buffer-native-finalize-20261015` waits on
exact validation PID5961 and records its own terminal status. It archives
success or failure, then requires the separate frozen-evidence auditor before
dispatching a benchmark. The new auditor regenerates all three transformed
copies from the archived chain-probe sources and retains every original28
numerical assertion. It replaces only the old job's build/resume provenance,
root path and graph-test marker, adding the dependent-chain64-case assertion.
This validation is fresh from configure; no compile/stage reuse is claimed.

The benchmark protocol is one native ABBA cohort for all three models at
batch1/2, each worker receiving one first-after-init and two warm requests.
All72 complete generated outputs must match their original expected tokens,
prompts and stops. Model-derived graph counts remain mandatory. Explicit
native Flush counts return to the original stream count; internal MPSGraph
commits are not represented by that counter. Source snapshots cover both the
copied serving tree and external parent-tree drivers. Initialization, lazy
compile/first request, warm TTFT/decode and peak process RSS remain separate;
caches stay unflushed. No hardware-cold claim or fresh vLLM ratio follows.

```text
python3 prepare_command_buffer_native_checks.py ROOT ALT
python3 ALT/source/test2026/metal/launch_command_buffer_native_checks.py ALT ALT/logs/mlp-command-buffer-native-checks-20261015 checks
python3 finalize_command_buffer_checks_and_launch_profiles.py ALT ALT/logs/command-buffer-native-finalize-20261015
# Controller, only after exact terminal status:
python3 archive_command_buffer_native_checks.py ALT ARCHIVE
python3 verify_command_buffer_native_checks.py AUDIT AUDIT_RESULT ARCHIVE
python3 launch_command_buffer_native_profiles.py ALT ALT/logs/mlp-command-buffer-native-20261015
# Only after native terminal capture:
python3 archive_command_buffer_native_profiles.py ALT NATIVE_ARCHIVE
python -X utf8 test2026/metal/verify_command_buffer_native_profiles.py NATIVE_AUDIT NATIVE_AUDIT_RESULT
```

Here ROOT is `/Users/zsdx/garnet-metal-investigation`; ALT is its
`mlp-command-buffer-investigation-20261015` child. These are reproduction
templates; exact executed commands and raw outcomes remain in their job logs.
Controller exit0 means benchmark dispatched, never benchmark success or goal
completion. Repeated native timing and fresh matched vLLM cohorts are still
required before any selection or target claim.
