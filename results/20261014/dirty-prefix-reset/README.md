# Bounded cache reset: repeated component signal

Clearing only the previously written KV prefix substantially reduces reset
cost in a synthetic component on the actual M4. This is not yet integrated
into Garnet serving and does not prove a TTFT, decode or vLLM improvement.
Selected libraries and production source remain unchanged.

The current full-request telemetry records text batch-2 reset taking about
20–25 ms. This experiment compares temporary zero-vector allocation/copy
(current baseline), full direct CPU zero, and bounded direct CPU zero. Actual
checkpoint configurations determine all state sizes at capacity 256 and B1/B2.
Each KV head retains its full allocation; only its dirty token prefix is
cleared. All convolution and recurrent state is fully cleared in every policy.
FP32 state storage, MPS row strides and padding remain unchanged.

| Model / batch, 64 dirty tokens | Initial legacy / full-direct / prefix ms | Confirmation legacy / full-direct / prefix ms |
| --- | ---: | ---: |
| Text / 1 | 4.437 / 0.609 / 0.138 | 4.092 / 0.703 / 0.156 |
| Text / 2 | 19.633 / 1.651 / 0.368 | 19.357 / 1.432 / 0.642 |
| Coder / 1 | 0.309 / 0.052 / 0.049 | 0.310 / 0.052 / 0.050 |
| Coder / 2 | 1.108 / 0.111 / 0.112 | 1.117 / 0.156 / 0.134 |
| VL / 1 | 2.605 / 0.447 / 0.365 | 2.862 / 0.336 / 0.251 |
| VL / 2 | 8.795 / 0.745 / 0.607 | 8.955 / 0.610 / 0.447 |

These are reset-component wall means after bounded GPU writes, not full-request
estimates. Differences between component and serving cache behavior can be
large: the earlier full direct-zero experiment reduced TTFT but slowed text
batch-2 decode by 1.7–2.4%. Bounded clearing must therefore pass native repeats
before selection; component savings must not be added to old comparison results.
Coder gains over a full direct clear are small or absent.

Both runs cover 25 profiles: all three models × B1/B2 × dirty lengths
0/1/64/256, plus a padded synthetic allocation. Each run validates 300
nonzero-to-positive-zero transitions on CPU and GPU and records 3,600 timed
resets. The prefix-zero case still clears all linear state. Full-capacity
cases exercise fallback-equivalent work. Whole logical buffers, including
unwritten regions, finish at positive zero and padding retains its sentinel.

The first run checked initialization and complete post-clear buffers but did
not separately observe the GPU producer pattern. The confirmation adds that
independent observation before every numerical clear: every dirty element
matches the known finite nonzero pattern and every unwritten element is zero.
Its local audit requires `all_gpu_dirty_patterns_observed: true`. The actual
reset policies and timing scope are unchanged. No tolerance is involved.

Each profile alternates all three policies while warming for at least 250 ms,
then runs two rounds of legacy/direct/prefix/prefix/direct/legacy. Each window
has three warm operations and twelve samples. Fresh bounded GPU writes and
completion precede every reset; their cost and correctness checks are excluded
from reset timing. Initial full initialization is excluded for every policy.
OS caches and GPU frequency are uncontrolled. The independent 0.5-second
guard rejects other inference/probe jobs and only terminates owned processes.

Initial owned PID 42202 exited 0 in 37.92 seconds. Confirmation PID 43002
started after it was terminal and exited 0 in 42.46 seconds. Each archive has
16 files, exact executed sources, compiler commands, monitors and a hash/size
proof for the omitted binary. Local audits independently reconstruct model
state sizes, all profile/check/window/sample coverage and timing means.

Archive SHA-256:

- Initial: `4bf3cb9bf30bde60a8267d1d5da71682bad35befab78c1aed9c913e1ff0ca223`
- Confirmation: `076071956340e461e6657a41f1b8062003f2b84a51477190fb0d630d59ab2850`

```text
python3 launch_dirty_prefix_reset_probe.py ROOT ROOT/logs/dirty-prefix-reset-probe-20261014
python3 launch_dirty_prefix_reset_probe.py ROOT ROOT/logs/dirty-prefix-reset-confirm-20261014
python3 archive_dirty_prefix_reset_probe.py ROOT JOB ARCHIVE
python verify_dirty_prefix_reset_probe.py RESULTS VERIFICATION_JSON
```

## Native admission requirements

Any native implementation must default off, preserve original selected builds,
and track the high-water mark of validated active KV writes independently for
each session. New buffers require full zero initialization. Generic graph
outputs, unknown state writers, unsupported cache layouts, failed commands
and poisoned state require a full reset. Inactive rows must not advance dirty
extent. Gaps and overwrites still require clearing through the highest actual
write, not merely the number of generated tokens. Extent must persist across
chunk profile changes and swaps of private cache storage. Linear state remains
fully reset. Full-cache snapshots after reset must match the original reset
bit for bit, with no reuse of prior request contents.

Validation must cover all original nineteen numerical stages, multiple
request lengths, sparse/overwritten/inactive slots, two interleaved private
sessions, failed-run recovery and full-capacity boundaries before six-profile
native ABBA timing. A repeated serving gain and fresh matched vLLM comparison
are required before any selection change. The full 2x/half-latency goal is unmet.

## Optional native implementation

`GARNET_METAL_DIRTY_PREFIX_RESET` defaults OFF and is mutually exclusive with
the earlier full direct-reset experiment. The new isolated build keeps global
half alpha enabled and all other experimental paths disabled. The three
selected libraries retain their original hashes.

Only graph outputs proved to have a single KVUpdate/KVChunk writer, the
original cache input, explicit unchanged slot/active controls and a supported
four-dimensional cache layout admit the bounded reset. Admission is computed
once per compiled graph. A session tracks the highest validated active slot
plus one across decode/chunk calls and private storage swaps. It does not
infer extent from token count or clear another session's buffers. Generic
writers invalidate the proof; poisoned requests force full reset recovery.
Every new state buffer is fully initialized. Convolution/recurrent and
unsupported state retain the original full vector-copy reset. Padding is
preserved and host clearing requires completed, initialized buffers with a
closed command batch.

Workers now record actual cleared-byte counts, bounded-state counts and full
fallback counts inside the unchanged TTFT scope. Warm profile admission
checks these against the preceding complete request, prompt length, actual
decode steps and archived model configuration. First-request counters are
checked separately, without calling initialization hardware-cold latency.

Installation verified eleven original source hashes against `4053c7d4`, saved
their exact bytes and installed twelve pinned candidate files. The candidate
library SHA-256 is
`e389fc080249301b506b84ff09b242df7d6cd63128091e1ab677b4025710491c`.
The actual M4 configure/build and both CTests pass. The new unit performs 216
whole-buffer positive-zero checks after GPU writes/consumption across rows
1/2/3, groups 1/3, capacities 1/17/256, widths 1/3/128 and dirty extents
0/1/half/full. Invalid bounds, uninitialized buffers, open commands and abort
recovery are also checked. The original numerical CTest remains intact.

Owned validation PID **44364** exited 0 in 1205.88 seconds after all nineteen
original stages. Dependent PID **45551** passed six B1/B2 real-layer recovery
gates (partial clearing, unknown writer, nonfinite-head poisoning, full recovery
and session isolation), then exited 1 in 1172.16 seconds including its wait.
The first performance profile failed the actual bounded-state count assertion.
All six completed control/candidate outputs matched, but both warm candidate
requests cleared the full 58,720,256-byte text cache and recorded zero bounded
states. These incomplete timing windows are rejected, with no speed claim.

The failure archive SHA-256 is
`a55880de3c40e979ad396f0d564ab8e200edb3d37a65ee7a1204146aba16ec9f`.
`native/failure-verification.json` verifies all 461 archived files, original
nineteen stages, six recovery checks, six completed matching outputs, actual
fallback byte counts, exact source and preserved terminal failure. Evidence is
under `native/failed`; the original first candidate binary remains intact.

`native/pending` is a historical bounded snapshot of actual owned handles,
source hashes, library identities, successful configure/build/CTest stages and
the real unit output. It is not evidence that the remaining full-model/session
gates or serving measurements are complete. Preserve current source/binaries
until their terminal records and exact executed-source archives are collected.
No native TTFT/TPS gain or new vLLM comparison is claimed yet.

## Completed cache-copy admission repair

Prefill returns its cache through a fresh `Layout` reshape copy. The first
admission check expected the final writer itself to be KVUpdate/KVChunk and
conservatively invalidated tracking. The repair traces a sole resident writer
through complete row-preserving Layout copies, with unchanged rows/columns,
zero layout operation, no auxiliary output and cycle detection. It must still
end at the validated KV writer with the original cache/slot/active inputs and
matching head/capacity/dimension metadata. Add, mask, reorder, partial copy and
unknown producers remain ineligible. The numerical kernels are unchanged.

All twelve prior source hashes were pinned against `32b90ea8` and saved before
installation. A separate repaired build preserves the original candidate and
all selected binaries. Repaired library SHA-256:
`b5614830b706e3ae8bf8ca3333f492b7b79e47e1bca289041267d0fb706efaee`.
Configure/build and both CTests pass. Six new actual trained chunk-prefill
checks also pass, covering all three models and B1/B2. Sparse active slots
0/3 clear only extent 4 at capacity 8; an inactive slot 7 must not enlarge it.
Repeated full-cache snapshots verify all zero and preserve another session.
These tests specifically detect the first implementation's fallback bug.

Owned repair validation PID **66790** exited 0 in 1224.31 seconds after all
22 stages: the original nineteen gates plus three model-specific copy stages.
Dependent PID **68433** exited 0 in 1299.91 seconds including its validation
wait, passing six recovery gates and all six native ABBA profiles (72 complete
requests). Fresh confirmation PID **91803** exited 0 in 151.33 seconds after
six more recovery gates and another 72 complete requests. The second run
starts after the first is terminal. Actual cleared-byte/state/fallback counts
match archived checkpoint configuration, prompt length and preceding decode
steps for every request, including separate first-request accounting.

| Model / batch | Initial TTFT control → candidate ms | Repeat TTFT control → candidate ms | Decode candidate/control initial / repeat |
| --- | ---: | ---: | ---: |
| Text / 1 | 50.979 → 44.536 | 48.678 → 43.907 | 1.0081 / 0.9846 |
| Text / 2 | 82.604 → 66.170 | 80.444 → 66.484 | 0.9778 / 0.9831 |
| Coder / 1 | 31.620 → 32.397 | 32.617 → 32.161 | 1.0053 / 0.9934 |
| Coder / 2 | 52.059 → 50.959 | 51.434 → 49.430 | 1.0021 / 0.9985 |
| VL / 1 | 196.462 → 192.869 | 195.229 → 194.557 | 0.9949 / 0.9942 |
| VL / 2 | 359.408 → 355.739 | 360.893 → 357.563 | 0.9955 / 1.0055 |

Text TTFT reductions repeat at 12.6% / 9.8% for B1 and 19.9% / 17.4% for
B2. Text B2 decode decreases 2.2% / 1.7%; B1 throughput direction changes.
Coder B2 TTFT improves 2.1% / 3.9% with essentially flat decode. Coder B1
latency direction changes. VL warm gains are small, with no repeatable decode
improvement. This remains an optional latency/throughput tradeoff; it is not
selected as a uniform policy for the objective that requires both metrics.
No favorable-sample rerun or new vLLM speed claim follows these results.

Repeat text reset falls from 5.611 to 1.226 ms at B1 and 20.144 to 2.277 ms
at B2. Actual candidate clearing is 6,651,904 / 13,303,808 bytes; the control
library is not instrumented, so its recorded zero counters are unavailable
instrumentation, not an absence of reset. Phase reports use null for them.
Reset savings do not exactly equal full TTFT savings. The cause of the decode
tradeoff is not independently established. GPU/wait intervals overlap.

First request after initialization on repeat is 154.500 → 149.482 ms for
text B1 and 188.011 → 166.792 ms for text B2. Initialization is separately
reported and essentially unchanged. OS/driver caches were not flushed; these
are not hardware-cold end-to-end measurements. Other first-use phases and
resource observations are retained in the reports.

The combined native archive SHA-256 is
`331e44ebc223b26bcbf53ed8758eb7b825ac8a3b2cc9ecd943c3ee9cd2fca5d0`.
Both independent audits verify 1,207 files, all 22 stages, six recovery gates
per run, 72 complete outputs per run, fresh frame hashes, byte counts and raw
timing means. Initial archive SHA-256 was
`42828a0a33e9ecb9ff8d044bda0e89c7e72c3441b30e4b1bfdad9a45de1f7487`.
Reports are `native/verification.json`, `native/confirmation.json` and their
phase summaries. Historical pending snapshots are superseded. Selected
libraries, GPT-OSS and Pi checkouts are preserved; the full goal remains unmet.

```text
python prepare_dirty_prefix_reset_sources.py
python3 install_dirty_prefix_reset_sources.py ROOT
python3 launch_dirty_prefix_reset_checks.py ROOT ROOT/logs/dirty-prefix-reset-checks-20261014 checks
python3 queue_dirty_prefix_reset_profiles.py ROOT ROOT/logs/dirty-prefix-reset-native-20261014
python3 archive_dirty_prefix_reset_native.py ROOT ARCHIVE
python verify_dirty_prefix_reset_native.py EVIDENCE VERIFICATION_JSON
python prepare_dirty_prefix_reset_repair_sources.py
python3 install_dirty_prefix_reset_repair_sources.py ROOT
python3 launch_dirty_prefix_reset_repair_checks.py ROOT ROOT/logs/dirty-prefix-reset-repair-checks-20261014 checks
python3 queue_dirty_prefix_reset_repair_profiles.py ROOT ROOT/logs/dirty-prefix-reset-repair-native-20261014
python3 archive_dirty_prefix_reset_repair_native.py ROOT ARCHIVE
python verify_dirty_prefix_reset_repair_native.py EVIDENCE VERIFICATION_JSON
```
