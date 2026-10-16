# Streamed decoded body weights component screen

Status: original-bit/FP64/safety checks pass, but the complete six-profile
component screen rejects the candidate. No confirmation, native integration,
serving change or new matched-vLLM target count. Mac and Windows independently
regenerate all sources and produce identical audits and selection.

Hypothesis: consume each reconstructed float4 weight immediately inside its
compact/raw branch rather than retaining four vectors for a later loop. Keep
four outputs per SIMD, 64 threads, original payload/metadata/raw layout, K/j
iteration, accumulation statements and final SIMD reductions. The compiler may
already shorten this lifetime; register count/occupancy is unmeasured.
[Earlier Windows-only source preparation](../streamed-body-weights-preparation/README.md).

## Control and execution

Use a separate private copy of accepted d52's context, not the rejected width64
recurrence library. Both policies use the retained coalesced codec and fixed
64-thread geometry; the candidate selects only the streamed-weight kernel.
Actual context counters verify the original codec in both policies and the new
kernel only in the candidate. Preserve float-key and unsupported fallbacks.
The borrowed harness retains some older raw-BF16/wide/MPS comment/error labels;
the generated context, admission metadata and dispatch assertions establish
this experiment's coalesced control and streamed candidate.

Require the prior native controller to be terminal and archived with its exact
hash, and complete all evidence transfers before launch. Controller 26445 exits
0 in 35.186062542 seconds; runner exits 0 in 35.116723917 seconds. These durations
include setup/compilation/numerical checks/timing and are not inference latency.
No competing inference runtime is admitted. Freeze consumed sources until
terminal archive.

## Complete numeric and timing protocol

Verify all three checkpoints and every original raw BF16 tensor slice before
compilation. Keep all 29 fixture matrices: seven each at text/coder layer 0,
and 15 at VL's first linear layer 0 and first full-attention layer 3. These
include 20 selected matrices: 6 text, 3 coder, 11 VL. Record actual packing
counts. Run every model's numerical
mode before any model's timing mode, with four changed structured activation
patterns and B1/B2. All 24 numerical profiles and 680,832 original FP32 output
values match bit for bit and remain within the original FP64 gamma and comparison
bounds. All 63 tail/fallback/prefill, pending/foreign-writer, abort/recovery and
uninitialized-input safety profiles pass. This is first-layer projection-set
coverage, not all model layers or full requests.

Time each complete projection set in one native submission. At every six
model/batch profiles, use 2 seconds of common warmup, own-null before comparison,
10 ABBA rounds, 3 warm calls and 5 raw timing samples/window. Preserve all 2,400
GPU/wall samples and all routes. Repeated primitive fixtures here calibrate
kernels; full-model comparisons prohibit prefix/input/image reuse.

Before execution, admission metadata requires both unmet B1 text and coder
profiles to gain >=5% on GPU and wall time, all six profiles to preserve both
clocks within 2%, and all own-nulls within 5%. Compute ratios of arithmetic means
of all 100 raw samples per policy/profile. No sample filtering or null
normalization. The analysis implementation follows execution but precedes
inspection of timing values; this is not a formal preregistered study.

All six own-nulls pass. Both unmet B1 profiles miss the 5% requirement, and coder
wall fractions exceed the 1.02 preservation limit at both batches. Do not rescue
the Text B2 or VL observations with a new favorable policy. Reject without a
repeat. Results do not establish that compiler code is identical, only that this
source change fails the fixed measured screen.
[Complete source/numeric/safety/clock audit](initial/verification.json),
[identical Mac audit](initial/mac-verification.json),
[all means and fixed decision](initial/analysis.json),
[identical Mac decision](initial/mac-analysis.json).

| Model | Batch | GPU fraction | Wall fraction | Null GPU / wall | Stable null | Fixed gain/preservation pass |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Qwen3-0.6B | 1 | 1.007683 | 1.000537 | 0.987539 / 0.977747 | Yes | No |
| Qwen3-0.6B | 2 | 0.970760 | 0.957113 | 0.952557 / 0.952622 | Yes | Yes |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 1.005412 | 1.021977 | 0.975716 / 0.984307 | Yes | No |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 1.015110 | 1.028127 | 0.959671 / 0.978439 | Yes | No |
| Qwen3.5-0.8B | 1 | 0.990381 | 0.985504 | 1.001304 / 1.008732 | Yes | Yes |
| Qwen3.5-0.8B | 2 | 0.995865 | 0.995749 | 0.997377 / 1.001765 | Yes | Yes |

Fractions are candidate/control; below 1 is faster. A preservation pass outside
the unmet B1 profiles is insufficient to select this common policy.

## Costs, evidence and reproduction

No new weight layout or logical tensor storage is introduced; a private context
and extra pipeline enable the comparison. Packing, source compilation and setup
are outside kernel intervals. Register count, process RSS and whole-model init,
first-token latency and throughput benefits are unmeasured. Post-run hardware
observation confirms Apple M4, Mac16,10 and 16 GiB; Python/compiler/macOS versions
are captured with their post-run scope.
[Environment observation](initial/evidence/logs/streamed-body-weights-20261016/environment-after-run.json),
[executed commands and exits](initial/evidence/logs/streamed-body-weights-20261016/jobs/statuses.json),
[frozen source/design/assets](initial/evidence/logs/streamed-body-weights-20261016/jobs/metadata.json).

The 58-file archive preserves all generated/source snapshots and raw numeric,
guard and clock logs. Large original matrices/checkpoints and the executable
are recorded by size/hash. Archive SHA256:
`18a550c8a8c6b9c510f47ad27a4a1ef00dcaceddab52c5b1773ff65aea69849f`.

ROOT=`/Users/zsdx/garnet-metal-investigation`, TOOLS=`ROOT/streamed-body-weight-tools-20261016`,
ARCHIVE_TOOLS=`ROOT/streamed-body-weight-archive-tools-20261016`:

```text
python3 TOOLS/launch_streamed_body_weights.py ROOT ROOT/logs/streamed-body-weights-20261016
python3 ARCHIVE_TOOLS/archive_streamed_body_weights.py ROOT ROOT/logs/streamed-body-weights-20261016 ROOT/streamed-body-weights-20261016.tar.gz
python -X utf8 test2026/metal/verify_streamed_body_weights.py results/20261016/streamed-body-weights/initial results/20261016/streamed-body-weights/initial/verification.json
python -X utf8 test2026/metal/analyze_streamed_body_weights.py results/20261016/streamed-body-weights/initial results/20261016/streamed-body-weights/initial/analysis.json
```

All consumed processes are terminal and archived. The full 2x decode/half-TTFT
goal remains active and unmet; no new matched comparison was run.
