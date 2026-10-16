# WS-M5-02: limited exact-half prefill through WorldSense — rejected

The deployed seven-flag M5/4B profile is compared with only
`GARNET_METAL_EXACT_HALF_PREFILL` and `GARNET_METAL_GLOBAL_HALF_ALPHA` enabled.
Fused-head remains OFF. Original learned BF16 values, FP32 activations/results,
checkpoint, pixels, context/output caps, prompt bytes and chunk16 are preserved.
This is a measurement of the existing optional policy, not a new algorithm.

The [WorldSense fixed-frame protocol](../worldsense-m5-fused-head/README.md)
is reused in four fresh sequential control/candidate/candidate/control workers:
three fixed recorded-video frames/current or search-only Skills, real Galaxy
VLM perception and text verification, plus fixed text search planning. All
72 complete model-call outputs match, including token sequences, texts and
input identities. Warm metrics use two later repeats of every case; initial
warmup records are retained separately. Live inference bindings are temporarily
cleared, restored in `finally`, and benchmark events/states are not committed.
Video decode, scheduling, temporal states, event persistence and search
retrieval/ranking remain outside this matched timing scope.

Before timing, `protocol.json` requires ≥10% 4B TTFT and ≥5% 4B total
improvement in both comparisons, with 4B/text decode and text TTFT/total
preserved within 2%. The distinct head experiment's failed gates are not
relaxed or reused. No competing GPU probe runs during these timing windows.

| Process | 4B total p50 (s) | 4B TTFT p50 (s) | 4B decode p50 (tokens/s) |
|---|---:|---:|---:|
| control-1 | 12.997 | 4.257 | 13.389 |
| candidate-1 | 14.832 | 5.377 | 12.360 |
| candidate-2 | 14.717 | 5.230 | 12.425 |
| control-2 | 14.564 | 5.265 | 12.462 |

First comparison: 4B total rises 14.12%, TTFT rises 26.29%, decode falls
7.68%. Second: total rises 1.05%, TTFT falls only 0.66%, decode falls 0.29%.
Text preservation also fails in the first comparison. **Reject promotion.**
The original library/profile is restored; no unchanged favorable retry is
used to rescue this result. Control drift is retained and its cause is not
isolated. There is no general hardware, model or full-system speed claim.

Actual candidate counters report 1,536 common-alpha dispatches in a
representative 4B prefill, versus zero in control. Thus the policy is active;
the failure is not explained by a completely inactive feature. A header-only
checkpoint size audit, performed during this experiment after its protocol
declaration, finds only **91,750,400 / 7,138,181,120 = 1.285%** of decoder
layer-matrix bytes can pass the existing ≤16 MiB size bound, before exactness
admission. Many large 4B projections remain on the original path. This
documents a scope limitation; it does not causally attribute every regression
to size limits or MPS scheduling.

The candidate retains additional exact-half buffers/row scales alongside
original BF16 decode buffers, adds startup scanning/encoding, and changes MPS
prefill accumulation ordering. It does not quantize learned values. Native
matmul tests include exhaustive integer-codec versus floating-roundtrip
checks, exact-half normal/zero/odd-row fixtures, common-scale cache separation
and unsupported-value fallback. CTest passes. Full WorldSense token parity
passes; the original historical six-profile/19-stage suite is not claimed as
a new M5/4B execution. No model-state bit-equivalence is inferred from text
parity or from exact weight storage.

All raw calls, first/warmup requests, method/gates, builds, libraries/serving
identities, per-cohort stdout/stderr and VM/swap observations are retained.
Model time excludes loading/queues/retries/storage; native and HTTP wall time
are separate. Estimated decode uses `(tokens−1)/(total−TTFT)` and can include
EOS-prediction work. RSS is sampled after the run, not peak unified GPU memory.
The initial size-audit glob matched no checkpoint files; its failure and the
actual downloaded shard naming are recorded in the preflight JSON. No timed
case was removed. The larger immutable shared-storage follow-up is a separate
unvalidated method and cannot turn this rejection into a success.
