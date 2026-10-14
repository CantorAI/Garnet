# GPU-verified contiguous prompt-slot lookup: repeated latency gains

The existing chunk-attention kernel repeatedly scans preceding tokens for
each visible cache slot to find its latest writer. This experiment checks
the current query's active prefix once per threadgroup. If every earlier
token is active and has consecutive integer slots, the writer follows
directly from `slot - first_slot`. Otherwise it retains the original scan.
Cached slots preceding the chunk still read the old cache. Inactive queries
still produce zero output. Score, softmax, value accumulation and gating
remain FP32, with unchanged learned values and arithmetic order.

`GARNET_METAL_ATTENTION_PREFIX` defaults OFF. The GPU verifies actual control
values at every query; it does not assume that callers always use consecutive
slots. The experiment introduces no request-to-request prefix reuse, encoder
reuse or speculative decoding. The selected primary library is preserved.

Two terminal component runs pass 36 shape profiles / 1,008 changed-input
comparisons. They cover batches 1/2, chunks 16/32/80, three GQA layouts,
gated/ungated output, changed data and seven control patterns: consecutive
slots from zero, a cached prefix, paused tail, duplicate overwrites, gaps,
paused/resumed tokens and leading pauses. Output is bit-identical to the
extracted control kernel, including forced padded strides. Independent FP64
references pass with a maximum tolerance ratio of 0.005630 against the
unchanged `5e-6 + 5e-5*abs(reference)` bound.

| VL-shaped component | Initial control / shortcut GPU ms | Initial ratio | Confirmation control / shortcut GPU ms | Confirmation ratio |
| --- | ---: | ---: | ---: | ---: |
| B1, chunk 80, 8 query / 2 KV heads, dimension 256 | 2.1644 / 0.6212 | 3.4844× | 2.6735 / 0.7363 | 3.6307× |
| B2, chunk 32, same head layout | 0.3276 / 0.1804 | 1.8159× | 0.3283 / 0.1796 | 1.8281× |

The 288 ABBA timing records contain 12 completed submissions each. Synthetic
inputs and warm command-buffer spans are used; these are component gains,
not model throughput or end-to-end TTFT improvements. Absolute timings vary.
The production MSL conditional resolves to exactly the tested shortcut body;
default-OFF resolution preserves the control body.

The initial component attempt exited before shader generation because the
guard observed an unverified WorldSense PID. That PID was already missing
when audited, and the saved service source fingerprints were unchanged.
The guard now rechecks PID liveness after a failed service audit; a still-live
unknown worker remains rejected. Three process-exit regression cases verify
the disappeared-PID, live-unknown and verified-service decisions. No other
job was stopped or reclassified without verification. Later GPU runs completed
with guarded 0.5-second monitoring.

During the first full-model run, the Mac volume filled and the final worker's
monitor write failed with `ENOSPC`. The queued confirmation correctly refused
to start. That incomplete timing run is retained but excluded from conclusions.
An initial retry then stopped on an updated WorldSense supervisor fingerprint.
Read-only inspection confirmed the same pinned HTTP launch payload/file and
no inference/GPU libraries in either live process. The inspected supervisor
fingerprint was added to the existing environment-name exclusion, retaining
all loaded-library and unknown-worker checks. No WorldSense process was stopped.

Storage recovery removed 705 verified, reproducible `.bin` matrix assets from
old isolated component jobs, totaling 15.17 GiB. Before deletion, original
checkpoint, manifest, size and asset hashes were revalidated. Only individual
derived files inside the investigation logs were removed; canonical models,
libraries, source, manifests and textual evidence were retained. All three
retained checkpoint hashes are recorded in `source-history/cleanup-removed.json`.
The volume subsequently reported 19 GiB free. Clean timing reruns monitor free
space and reject runs below 2 GiB. Complete discarded runs, cleanup provenance
and supervisor audits remain available for review.

The candidate is isolated in `build-attention-prefix`. Other experimental
vision-fusion, joined-prefill and native-BF16-body flags are OFF. Standard
handoff checks, exact 22/26-token references and extended Qwen3.5 references
are recorded separately from native ABBA timing. The current matched text,
coder and VL workloads retain their prompts, image bytes, capacities,
precision, caps and prefill plans. Native outputs record the actual backend
feature marker after the measured work.

All 16 configure/build/validation stages pass: standard native handoff,
48 exact-tail comparisons and 36 extended Qwen3.5 comparisons. Two complete
healthy native ABBA runs cover all six model/batch profiles and 144 complete
outputs. Every prompt, generated token sequence, stopping state and VL patch
hash matches the frozen control. Four non-first requests per build/profile
form each warm mean. The incomplete and guard-rejected runs are excluded.

| Model | Batch | Initial TTFT reduction | Confirmation control → candidate ms | Confirmation TTFT reduction | Confirmation decode ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B text | 1 | 2.12% | 54.41 → 51.57 | 5.21% | 0.9866× |
| Qwen3-0.6B text | 2 | 4.96% | 89.53 → 86.87 | 2.97% | 1.0000× |
| Qwen2.5-Coder-0.5B | 1 | 8.15% | 38.13 → 35.33 | 7.34% | 1.0012× |
| Qwen2.5-Coder-0.5B | 2 | 9.20% | 60.89 → 56.97 | 6.44% | 0.9965× |
| Qwen3.5-0.8B VL | 1 | 5.36% | 214.24 → 205.64 | 4.01% | 1.0033× |
| Qwen3.5-0.8B VL | 2 | −0.68% | 386.08 → 381.85 | 1.10% | 1.0011× |

Warm TTFT gains repeat for text/coder at both batches and VL at B1. VL B2
does not establish a repeatable gain. Decode computation is unchanged and
the small mixed decode ratios do not establish improved token generation
speed. The selected primary/control library remains unchanged; the optional
candidate is available for the five profiles with repeated latency gains.
The next matched comparator study should retain the control for VL B2.
The CMake flag remains OFF by default while this selection is investigated.

Confirmation first-use TTFT (excluding initialization) is 154.23 → 149.69 ms
for text B1, 191.46 → 189.35 for text B2, 123.75 → 121.03 for coder B1,
145.53 → 137.06 for coder B2, 338.98 → 326.65 for VL B1 and 565.69 → 557.79
for VL B2. Some initial first-use gains do not repeat in direction; these are
worker first-use observations, not flushed hardware/driver cache measurements.
Confirmation initialization ratios are 0.9814–0.9999; individual RSS changes
range −18.71 to +11.30 MiB. No memory-allocation saving or broadly faster
startup is established by this lookup-only change.

The candidate library SHA-256 is
`f2c5c448ec4f3c8431be37b48a2634d8fa4cc31de4aa163ec367a319c9263088`.
The preserved primary/control remains
`73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2`.
The archive SHA-256 is
`42fab335acc54a5e794c9a47e3d4948b4250d7e07dac9cc8e5ed97a0a5b1458a`.
The verifier checks 1,091 archived hashes, actual compiler definitions,
control/shortcut MSL equivalence to tested source, all terminal results,
1,008 component comparisons, 3,456 timed component submissions, all 144
healthy model outputs/warm means and recorded disk-space guards.

Initialization, first-use and warm requests are separate; OS/driver caches
are not flushed. Individual worker RSS is not unique GPU memory or aggregate
process-tree memory. No new vLLM-Metal result is established by the native
control/candidate study. The overall 2x decode / half-TTFT goal remains unmet.
