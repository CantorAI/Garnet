# Retained CPU zero source: full-model investigation

Status: actual fresh candidate build and standard MPS CTest passed. The original
19 model-validation stages are running; trained lifecycle/recovery cases and all
six native ABBA profiles are queued behind that exact job. Full-model speed and
the overall 2x/half-TTFT goal remain unproven. No new vLLM ratio is claimed.

The repeated component study in `../reused-zero-source/README.md` found lower
reset-plus-dependent-read wall times with a retained CPU zero vector. That does
not prove an inference gain. Earlier direct, dirty-prefix, and deferred GPU
reset policies improved some TTFT profiles but slowed decode, particularly text
batch 2. This experiment retains the original CPU copy operation and removes
the allocation/initialization of a temporary vector for each state reset.

`GARNET_METAL_REUSED_ZERO_SOURCE` is optional and defaults OFF. Each session
allocates one positive-zero FP32 source sized for its largest private state
matrix. Build initialization includes this allocation. Reset immediately copies
all logical elements into every private state matrix through the existing
MetalContext Update operation. It does not defer clearing or reuse inference
state. Rebuild and close release the source; failed reset leaves the session
poisoned. Source bytes, copied bytes, and copied-state counts are exposed to the
measurement workers. Three alternative reset flags are mutually exclusive.

The six frozen profiles remain Qwen3-0.6B, Qwen2.5-Coder-0.5B-Instruct, and
Qwen3.5-0.8B VL, each at B1 and B2. Native control is the selected global-alpha
library. Checkpoints, output caps, stops, prompts, image bytes/resolution,
arithmetic precision, and fresh per-request image encoding are unchanged.
The comparator is not running alongside this study. Inference guards retain
their 0.5-second sampling and whole-case contention rejection.

Actual owned jobs on `/Users/zsdx/garnet-metal-investigation`:

| Job | PID | Directory |
| --- | ---: | --- |
| Fresh build and original 19 gates | 55933 | logs/reused-zero-source-checks-20261015 |
| Dependency wait, lifecycle cases, native ABBA | 57066 | logs/reused-zero-source-native-20261015 |

Both handles were confirmed live after launch. A PID is not proof of later
completion; inspect the corresponding exit.json and process state before
continuing. The queue checks the exact dependency, all 19 stage exits, and the
installed source manifest before executing recovery or measurements. Session
cases cover all three trained attention/head fixtures at B1/B2, full-logit
greedy parity, immediate snapshots, private-session isolation, repeated reset,
nonfinite poisoning/recovery, rebuild, close, and reuse after close.

Fresh candidate dylib SHA-256:
`c54ff5a158ce48bce8dd36e8eec6c2e59049013d13fbdf4db22233fb4476d6ac`.
This identifies the built artifact; it is not a correctness or performance claim.

Commands (run from the investigation source's test2026/metal directory):

```text
python3 install_reused_zero_source_sources.py ROOT
python3 launch_reused_zero_source_checks.py ROOT ROOT/logs/reused-zero-source-checks-20261015 checks
python3 queue_reused_zero_source_profiles.py ROOT ROOT/logs/reused-zero-source-native-20261015
```

After both jobs terminate successfully, archive with
`archive_reused_zero_source_inference.py`, extract into an evidence directory,
and independently run `verify_reused_zero_source_inference.py`. The verifier
checks raw-file hashes, original/candidate identities, all original stages,
six lifecycle cases, 72 complete matching outputs, fresh VL image encoding,
and exact source/copy counts calculated from the archived model configs. Warm
means exclude each fresh worker's first request. Initialization and first
request after initialization remain separate; caches are not flushed.

Require a second successful native ABBA run before selecting this policy.
Only after a useful candidate is selected should a fresh sequential vLLM-Metal
comparison establish updated runtime ratios. Do not combine new native numbers
with the older vLLM measurements as a fresh comparison.
