# Retained CPU zero source: full-model investigation

Status: actual fresh candidate build, standard MPS CTest, all 19 original
model-validation stages, six trained lifecycle/recovery cases, and two complete
six-profile native ABBA trials passed. Independent audits verified 1560 archived
files and 144 complete matching requests. The candidate is not selected:
text B2 TTFT improves in both runs, but decode slows in both runs.
The overall 2x/half-TTFT goal remains unmet. No new vLLM ratio is claimed.

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

Both jobs terminated with exit 0: validation took 1198.6444 seconds; the queue
took 1317.7301 seconds including its dependency wait. The queue checked the
exact dependency, all 19 stage exits, and the
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

## Actual initial results

Warm means from four requests per policy/profile, two fresh workers per
policy. Decode is tokens/s per row; B2 aggregate throughput is twice that rate.
These are native candidate versus native control, not a new vLLM comparison.

| Model | Batch | Control TTFT ms | Retained TTFT ms | Control decode | Retained decode | Decode ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 50.2961 | 47.3443 | 59.0259 | 59.1685 | 1.00242 |
| Qwen3-0.6B | 2 | 80.2221 | 68.6400 | 57.6058 | 56.6991 | 0.98426 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 32.6529 | 32.7672 | 72.6085 | 71.8103 | 0.98901 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 51.6816 | 50.9187 | 70.5768 | 70.7882 | 1.00300 |
| Qwen3.5-0.8B VL | 1 | 194.8253 | 192.2141 | 48.0711 | 47.8404 | 0.99520 |
| Qwen3.5-0.8B VL | 2 | 359.7486 | 356.6088 | 44.3280 | 44.2987 | 0.99934 |

Text B2 reset-call time falls from 21.6026 to 7.0560 ms, but observed prefill
inference rises from 57.0799 to 59.7889 ms and decode falls 1.57%. Faster reset
does not imply faster decode. Scratch overhead per session is 1/2 MiB for
text and VL B1/B2, and 128/256 KiB for coder. All reset source, copied-byte,
and state counts matched the archived model configs. Full initialization,
first request after initialization, and separate observed clocks are saved in
`verification.json` and `phases.json`; overlapping clocks must not be added.

Initial archive SHA-256:
`8e12b1d61004980674545d8cccdc6669fe5c0b31a2e2bf08b9cb73f9f382a106`.

The initial independent audit was uploaded before launching confirmation
PID 80622 at `logs/reused-zero-source-confirm-native-20261015`. Confirmation
pinned the exact candidate dylib and installed source manifest, retained the
same exclusive inference guards, and exited 0 after 144.6076 seconds. Both
trials completed all six profiles on attempt 0.

## Confirmation and decision

| Model | Batch | Control TTFT ms | Retained TTFT ms | Control decode | Retained decode | Decode ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 48.2805 | 46.4865 | 58.4147 | 59.3680 | 1.01632 |
| Qwen3-0.6B | 2 | 81.0828 | 70.3098 | 57.9835 | 56.5374 | 0.97506 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 32.9591 | 33.7309 | 72.3964 | 71.9557 | 0.99391 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 51.5155 | 52.0460 | 70.9548 | 71.1677 | 1.00300 |
| Qwen3.5-0.8B VL | 1 | 193.4979 | 194.4657 | 47.8640 | 48.2717 | 1.00852 |
| Qwen3.5-0.8B VL | 2 | 359.9487 | 356.5452 | 44.4928 | 44.4482 | 0.99900 |

Text B2 warm TTFT is 13.3% lower on confirmation, while per-row decode is
2.49% slower. Reset falls from 21.4782 to 7.2114 ms, but prefill inference rises
from 58.0252 to 61.6332 ms. Coder B1 decode is lower in both runs, coder B2 TTFT
changes direction, and VL effects are small or mixed. The observed prefill and
decode changes are measurements; a cache or memory-placement cause has not
been demonstrated. Keep the default OFF and the existing global-alpha
selection. This policy does not provide the broad improvement required by the
active goal, despite its component speedup and repeated text latency benefit.

Combined archive SHA-256:
`a3386cb363020fc1fa079a0f2c7e82bb4883fb7766e45070c38da694dc419b14`.
Both `verification.json` and `confirmation-verification.json` independently
recompute their respective 72 outputs and warm means against this combined
manifest. `confirmation-phases.json` preserves the repeat's observed clocks.
The protected GPT-OSS and Pi checkouts remain clean.
