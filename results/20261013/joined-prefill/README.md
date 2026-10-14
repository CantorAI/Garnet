# Bounded joined MPS prefill: modest warm gain, costly first use

The optional `GARNET_METAL_JOINED_PREFILL=ON` prototype combines adjacent
BF16 projections with the same input into one larger exact-half MPS
projection and one split-rescale kernel. Original learned values, per-row
integer scales and FP32 inputs/accumulation/outputs remain. There is no
prefix/image reuse, speculative prediction, shorter prompt or token cap.

Two component shapes had repeated signals; serving admission is limited to
44 rows / K=1024 / N=3072 (text B2) and 26 rows / K=896 / N=4864 (coder B1).
Other shapes, split-key projections, unequal strides, output aliases and
unrepresentable weights keep the existing path. Selected invalid shape/device
or pending-input/output controls reject before output mutation. Pack caches
retain their source weight owners to prevent identity reuse after disposal.
The original single command-buffer dependencies are preserved.

The flag requires exact-half prefill and defaults OFF. It is not combined
with the optional native-BF16 body decode route in these measurements. The
joined shader and graph pairing are compiled only when enabled. Session
`joined_prefill_pairs` and the prefill result dictionary expose actual
dispatch counts; unsupported profiles count zero. Serialized per-operation
diagnostics deliberately retain separate projections rather than claiming
their timings represent this serving path.

## Actual validation

Both native admission shapes pass repeated bit-exact dyadic output tests
against the original two-projection path. Tests cover output alias rejection,
invalid dimensions, uninitialized input, foreign pending output ownership,
unsupported shapes, unchanged output data and counter reset after abort.
The new test is included in CTest alongside the original kernel reference.

All eleven existing serving/numerical/head/vision stages and all three
private-cache suites (24 profiles) pass. Another 48 independent trained
first/last-layer comparisons exercise 22/26-token chunks, batches 1/2 and
three carried-state steps, including paused rows, partial tails, overwritten
slots, invalid controls, state preservation and resident/explicit equality.
Dispatch assertions prove one joined pair in each admitted single layer and
zero in unsupported cases. All 48 retain the existing error bounds; worst
observed normalized error remains 0.9936574155, below 1.

The build/check runner records 18 successful configure/build/check stages.
No numerical tolerance is widened, and benchmarks are gated on its exit 0.

## Actual native inference and memory

Same M4 / 16 GiB, frozen checkpoints, image bytes, prompts, capacities,
output caps and stopping rules. ABBA order uses two fresh persistent workers
per build and three requests per worker. Four non-first requests form each
warm mean. All 72 initial plus 24 confirmation requests preserve complete
token IDs, prompt IDs and finished flags. Every VL request freshly encodes
its image. Actual full-model counts are 28 joined pairs for text B2 and
24 for coder B1; all other profiles have zero.

| Admitted workload | Initial warm TTFT control -> candidate | Confirmation warm TTFT | Repeat reduction | Repeat decode ratio | Repeat peak RSS increase |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text B2 | 90.47 -> 87.79 ms | 88.86 -> 86.47 ms | 2.68% | 1.0105 | 335.20 MiB |
| Coder B1 | 38.11 -> 36.84 ms | 38.94 -> 38.11 ms | 2.14% | 1.0061 | 167.18 MiB |

The warm latency direction repeats, but the whole-model effect is much
smaller than the isolated projection ratio. Decode changes are small and do
not establish a useful decode algorithm improvement. Apparent changes in
unsupported profiles, whose path is unchanged, are not attributed to this
optimization. Raw six-profile and two-profile confirmation records remain.

First-use TTFT regresses substantially in confirmation: text B2
196.66 -> 255.36 ms, and coder B1 127.94 -> 188.64 ms. Pack allocation/copy
happens lazily at first use and cached packs persist for the session's
lifetime. Initialization is separate and approximately unchanged:
text 1.685/1.683 s and coder 1.377/1.376 s (control/candidate).
`first-use-summary.json` preserves both runs and all other profiles.
OS/driver caches were not flushed; these fresh-worker samples do not prove
cold hardware or an observed combined startup-plus-first-token clock.

Peak RSS is the macOS `/usr/bin/time -l` maximum resident-set field for each
worker/process family, reported as an individual-process high-water measure
in bytes. The same peak is attached to that worker's three records; it is
not a per-request measurement, aggregate worker memory or unique GPU memory.
It includes initialization and first use. The first-use cost and higher
resident memory matter when evaluating the small warm latency reduction.

The implementation is retained as a default-OFF experimental option. The
selected serving library remains unchanged; no broad promotion is warranted
from this tradeoff. There is no fresh vLLM comparison or stock BF16 advantage
claim here. All-model 2x decode / half-TTFT remains active and unmet.

## Commands and evidence

```text
python3 run_joined_prefill_checks.py INVESTIGATION_ROOT OUTPUT_DIR
python3 run_joined_prefill_profiles.py INVESTIGATION_ROOT OUTPUT_DIR
python3 run_joined_prefill_profiles.py INVESTIGATION_ROOT OUTPUT_DIR --confirm
python3 archive_joined_prefill.py INVESTIGATION_ROOT ARCHIVE_PATH
python3 verify_joined_prefill_evidence.py EVIDENCE_DIR PROOF_JSON
python3 summarize_joined_prefill_first_use.py EVIDENCE_DIR SUMMARY_JSON
```

All three durable jobs have terminal exit 0. Numerical checks, monitored
native workers and predecessor gates prevent simultaneous competing inference.
Only created worker groups are exempt; only owned processes may be stopped.
There were no discarded/interrupted profiles in the selected runs. Monitoring
every 0.5 s cannot exclude arbitrarily short jobs between samples.

Exact compiler/runtime/time-wrapper commands, versions, all source/checkpoint/
library hashes, raw output and timing windows, dispatch counters, resource
records, process monitors and original source snapshots are in `evidence/`.
The installer checks original identities and verifies the complete staged
bundle before writing only its explicit investigation paths. Source bundles
retain both before/after identities. Matrix/RGB/patch arrays and binaries are
omitted with their sizes/hashes. GPT-OSS, Pi and the Mac original checkout
were not modified.

Candidate library SHA256:
`cc41fde1f3905fcfb02fed6a50ab0f9d68933e94358352b92db4e44cb5f43dfe`.
Unchanged selected control:
`73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2`.
Archive SHA256:
`c9b528fcb30f09b9c68d0eba90e314106b6da220c94a6e598f2c6076670d9c9f`.
