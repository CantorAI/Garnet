# Private single-slot KV writes: retained opt-in text decode improvement

GARNET_METAL_RESIDENT_KV_INPLACE=ON writes only the newly decoded token into
each session-owned key/value cache. It replaces the full-cache copy for
eligible single-token attention graphs. The option defaults OFF. Explicit
graph calls retain their independent outputs; rank-three chunk prefill retains
the original cache path, preserving causal behavior across overwritten slots.
Learned values, FP32 arithmetic, layouts, prompts and generated tokens are
unchanged. There is no prefix/encoder reuse or speculative decoding.

Eligibility requires a fresh output storage group with exactly one KV writer,
the matching state input, and no later consumer of the old input cache. Graph
bindings are restored after encoding, under the existing graph/session locks.
The session keeps its private cache allocation; invalid controls are validated
before any encoding/state mutation. GPU failures still poison the session and
require reset. kv_inplace_updates counts selected private cache update
dispatches in the last run, including dispatches with inactive rows.

All eleven existing kernel/serving/numerical/head/vision stages pass. Another
24 real first/last-layer profiles cover all three models, batches 1/2,
capacities 8/256 and eight steps each. Their 192 steps match explicit graph
outputs/cache values exactly. Paused/all-inactive rows, boundary slots,
overwrites, rejected controls, reset, caller-owned input preservation and
shared-graph session isolation pass. Counters prove two private cache dispatches
per tested single-layer request. A separate native GPU primitive passes 16
profiles with bit-exact IEEE words, signed zero, subnormals, NaN payloads,
inactive rows, abort unreadability and recovery. The legacy control rejects
the new aliasing mode. A fresh default-OFF build passes its kernel test and
all 24 private-session profiles with zero private update dispatches.

## Sequential native comparison

Same M4/16 GiB Mac mini and frozen checkpoints/workloads. Each profile has
ABBA build order, two fresh processes per build, three requests per process,
and four non-first requests per warm mean. Every complete output, prompt ID
and finished flag matches the frozen control. VL freshly encodes the same
288x192 image and 79-token prompt, with a 32-token output cap. Text/coder
retain their prompts, 64-token caps and finer prefill policies. Per-row
decode steps/s is reported; aggregate batch-2 throughput is twice this.

| Model / batch | Control steps/s | Candidate steps/s | Decode ratio | Control TTFT ms | Candidate TTFT ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text / 1 | 55.70 | 59.84 | 1.074 | 53.28 | 51.46 |
| Text / 2 | 51.25 | 58.66 | 1.144 | 93.57 | 91.74 |
| Coder / 1 | 72.29 | 71.96 | 0.996 | 40.45 | 41.07 |
| Coder / 2 | 70.03 | 71.26 | 1.018 | 63.86 | 63.23 |
| VL / 1 | 48.22 | 48.12 | 0.998 | 219.63 | 220.63 |
| VL / 2 | 44.51 | 44.99 | 1.011 | 390.93 | 390.19 |

The text gain repeats in a second full text ABBA run:

| Text batch | Control steps/s | Candidate steps/s | Decode ratio | Control TTFT ms | Candidate TTFT ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 55.56 | 59.81 | 1.077 | 53.59 | 50.54 |
| 2 | 51.50 | 58.52 | 1.136 | 93.07 | 90.43 |

An earlier independently completed text batch-1 profile also gives 1.074.
The retained signal is text decode: about 7.4-7.7% at batch 1 and 13.6-14.4%
at batch 2. Coder/VL changes are small and mixed; no useful gain is established
for them, and they were not repeated solely to seek a favorable sample.
First-token changes are small compared with the half-TTFT target. No new
vLLM comparison ran here, and native ratios must not be presented as vLLM
speedups or stock/default BF16 advantages.

Initialization and first-use requests remain separate in raw records.
Confirmation initialization is 1.657/1.657 seconds (candidate/control,
text batch 1), and 1.681/1.683 seconds (batch 2). Filesystem/driver caches
were not flushed. No new process-memory improvement is claimed: independent
graph cache output buffers remain allocated for explicit calls.

## Guards and retained failures

Unknown external xlang3 jobs interrupted the initial validation's post-guard
and two full-suite benchmark attempts. The initial text correctness suite
had exited zero and is reused by the successful resumed validation. Its
wrapper's guard failure remains distinct from a numerical failure. Interrupted
benchmark workers are excluded from selected results. Clean completed profiles
are reused only after checking all four worker exits and monitor records.
Native workers are monitored every 0.5 seconds; only the process group created
by this benchmark is exempt, and multiple owned native workers are rejected.
External jobs were not stopped. This samples contention; it cannot prove the
absence of arbitrarily short external jobs between polls.

The per-profile manager waits for a five-second clear guard interval before
each attempt and can retry only contention failures. Its first version failed
before inference because a guard helper needed hashlib; that source/error is
preserved alongside the corrected version. Source snapshots distinguish the
initial monitored runner, the word-check preflight version and the final
filtered profile runner. No running job was restarted because observation
timed out. The selected six-profile set plus text confirmation contains 96
clean timed requests. Other initial/failed/partial records remain separately.

completed/ contains commands, statuses, guards, monitor samples, full outputs,
numerical logs, profile means and first-use data. current-source/,
original-source/ and source-history/ retain exact code versions and transfer
manifests. Raw patch/RGB arrays and standalone binaries stay on the Mac with
sizes/hashes in evidence-proof.json. Evidence archive SHA256:
204b2ce0c0731c643dd453c95e3c26df4513903acee7f8f4a32e00e3998ab75d.

Candidate library SHA256:
73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2.
Unchanged control:
cc5539990dcb4effbece7c611787a656df56b51271f1f40e1c3e8b5dbf90c8f8.
Default-OFF library:
092456f43ca1e96409b5d5917e861c9820b5843769a15560c4095ef6c2fee037.

The full three-model 2x vLLM-Metal decode / half-TTFT objective remains active
and unmet. The GPT-OSS and Pi checkouts are untouched.
