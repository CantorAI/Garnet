# GPU-private BF16 weights: compatible, no repeated broad speed gain

The probe keeps the original Garnet shaders, BF16 learned values and FP32
inputs/math/outputs, and changes only weight storage from shared to private.
Rounded staging bytes are uploaded by GPU blit and copied back for exact
comparison, including padding. Each successful profile reports actual
storage mode 2. Both batches preserve the existing small-output kernel
selection. Private upload/readback setup is outside projection timing.

The first attempt exited 1 after the trained text matrices: an additional
assertion incorrectly expected every private buffer's CPU pointer to be
null. A read-only allocation diagnostic showed private mode 2 at 1,408,
8,192 and 1,048,576 bytes; the two smaller buffers expose a non-null
pointer on this Mac. No private-pointer data access is used. The corrected
gate checks actual storage mode and exact GPU-copy readback. The failed
attempt's timings are discarded.

The corrected initial run (PID 1966) exited 0 after 37.56 seconds. Both
successful runs cover 34 fixtures / 68 batch profiles: 25 trained matrices
and nine repeated synthetic tail fixtures. Checkpoint hashes, raw fixture
hashes, BF16 dtypes, shapes and actual tensor slices are verified. Original,
negated, half and double inputs are freshly uploaded; output/padding data
is reset and checked on every warm or timed execution. Each profile adds
an alternating 250 ms common warm stage before its two ABBA rounds. This
is deliberately common to both policies, not proven GPU-frequency pinning.

The first confirmation (PID 3993) exited 1 in preflight when the guard saw
unrecognized XLang processes. A queued preflight waiter (PID 4319) was
superseded and terminated only after its exact owned command identity and
`benchmark_started=false` state were verified. No user process was stopped.
The read-only audit identified PIDs 2897 / 2985 as WorldSense CPU HTTP
services with exact supervisor/launcher identity and no inference or GPU
libraries loaded. The only inference keyword in the new supervisor is the
existing forwarded TRT-budget environment name. Imports and call symbols
were reviewed without exposing arguments, constants, credentials or
library paths. Supervisor fingerprint
`2568b1d1cefdf20d5610393c463169a3b4b2a490e2680dc3b0ca73e0ae1a84f7`
is now pinned for that literal exclusion; ancestor/inline identity,
launcher fingerprint and library checks remain. Unknown workers or loaded
inference libraries still reject.

The verified confirmation (PID 5640) waited for a five-second guarded quiet
interval and completed with exit 0 after 44.61 seconds, including preflight.
No investigation probe or waiter from these jobs remains running.

## Actual results

Across the two successful runs, all 122,898 warm/timed validation executions
match the current native outputs bit for bit. Each run includes 272 separate
FP64-envelope reference executions and 6,528 timed executions. Maximum
normalized FP64 error is 0.0100017036, within the unchanged prior bound.
Two rounds and two windows per policy give 48 timed samples per matrix/batch
and policy. Failed/preflight attempts contribute no averages.

| Model / batch | Private/shared pooled wall fraction, initial / confirmation |
| --- | ---: |
| Text B1 | 1.02688 / 0.99111 |
| Text B2 | 1.00647 / 0.99628 |
| Coder B1 | 0.94592 / 1.07914 |
| Coder B2 | 0.99176 / 0.99788 |
| VL B1 | 0.97138 / 1.01303 |
| VL B2 | 0.98141 / 0.99191 |

Smaller is faster; these are ratios of pooled component means, not whole
model TPS. The apparent initial coder/VL B1 benefit reverses. Small B2
reductions do not establish a useful broad algorithm improvement. Large
head results also vary by round. CPU encoding, wall and GPU clocks remain
separate and overlapping. Hardware frequency is not fixed, OS/driver caches
are not flushed and CPU-service load remains present. The 0.5-second guard
cannot exclude arbitrary shorter external activity.

No runtime integration is admitted. Selected native libraries, learned
values, precision, prompts, limits and protected checkouts are unchanged.
This is an allocation/projection component experiment with no fresh vLLM
comparison or model-output claim. The full 2x decode / half-TTFT goal remains
unmet.

## Commands and proof

```text
python3 launch_private_bf16_probe.py ROOT ROOT/logs/private-bf16-probe-20261014
python3 launch_private_bf16_probe.py ROOT ROOT/logs/private-bf16-retry-20261014
python3 launch_private_bf16_probe.py ROOT ROOT/logs/private-bf16-confirm-20261014
python3 launch_private_bf16_quiet.py ROOT ROOT/logs/private-bf16-confirm-clean-20261014
python3 audit_worldsense_cpu_identity.py
python3 cancel_superseded_private_wait.py
python3 launch_private_bf16_quiet.py ROOT ROOT/logs/private-bf16-confirm-verified-cpu-20261014
python3 archive_private_bf16.py ROOT ARCHIVE
python summarize_private_bf16.py EVIDENCE SUMMARY [SUCCESSFUL_JOB_NAME]
```

The initial evidence SHA-256 is
`8a7c4f31e093d28f626bc6b9c724f69d08a49e5cafa99172b0a0f3230cb822cf`
(42 files). The final combined evidence is
`18f8feed15e81ff7985e0e1ca2bf164f817ebfeb4fe3d981c1c5bccbe3377e77`
(82 files), including all terminal attempts and the sanitized CPU audit.
Local verification checks file/binary/source/shader hashes, terminal stages,
fixture coverage, private mode and byte readback, common warm duration,
complete exact checks, independent FP64 bounds, guarded samples, all
counts and recomputed timings. Raw evidence keeps its recorded byte
identities. `private_buffer_compat.mm` remains as the allocation diagnostic;
its first compilation lacked `<initializer_list>`, corrected before the
successful diagnostic execution. No benchmark timing depends on that
diagnostic's compilation failure.
