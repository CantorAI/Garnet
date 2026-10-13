# Partial output-head bounds

`partial-head-bounds-fusion-20261013` completed six native inference captures:
all three existing checkpoints, batches 1 and 2, identical saved baseline
prompts/caps/chunk profiles, and the same state-handoff native library. The
VL cases actually re-encoded the canonical real-image patches. Complete token
IDs, prompt IDs and finished flags exactly match the native matched baselines.
This diagnostic capture is not a timing comparison; it reprojects each sampled
state and writes full logits, adding work to the ordinary inference path.

The first eight head queries per stream were captured, for 72 row/step queries.
A standalone native RMS graph supplies the actual normalized FP32 inputs.
Its separate complete BF16-weight projection matches the ordinary full head
bit for bit, including signed-zero checks. Thus this probe removes the earlier
whole-row experiment's approximate CPU normalization limitation. The ordinary
complete head selected every generated token; no pruning executed on the GPU.

The offline analysis divides each vocabulary row into 32/64/128-value blocks.
After a partial dot product, Cauchy bounds cover the remaining dot product.
FP64 norm/partial calculations have conservative rounding allowances, plus
a deliberately conservative FP32 full-dot error envelope. Every bound at
every tested prefix was checked against every corresponding actual GPU logit.
Every actual winner remained eligible, and the FP64 full-dot argmax agrees
with the GPU winner for all samples. This validates the sampled calculations;
it does not certify an unimplemented GPU bound or all future input states.

## Counterfactual weight reads

These percentages are median reductions in head weight reads per captured row,
including the cost of computing 8192 complete seed rows. The seed's actual
maximum GPU logit is the available threshold. Batch 1/2 medians are identical
for these duplicated-prompt workloads; batch 2 still requires a GPU design
that shares weights across rows and tracks their union of eligible candidates.

| Model | 32-value contiguous blocks | 32-value blocks ordered by hidden energy | 128-value contiguous blocks |
| --- | --- | --- | --- |
| Qwen3-0.6B | 30.32% | 33.85% | 25.76% |
| Qwen2.5-Coder-0.5B-Instruct | 28.56% | 36.18% | 24.11% |
| Qwen3.5-0.8B, real-image queries | 23.39% | 29.24% | 19.21% |

The estimates omit bound metadata reads, reductions/norm arithmetic, dispatch,
noncontiguous gather overhead, batch-sharing effects and session integration.
The analysis itself read the complete head to construct/check the bounds.
It did not skip actual CPU/GPU inference work or measure a speed gain.
The `oracle-full-max` policy in the raw reports uses the unavailable complete
winner threshold and is explicitly optimistic feasibility data; it is not
used in the table or proposed as a serving implementation.

These results justify a measured GPU prototype with actual seed calculation,
conservative GPU bounds, shared batch reads and checks against the complete
head. Such a prototype must validate retained logits and selected tokens,
then full generated outputs and matched latency. The public complete-logit
head path must keep its semantics; returning pruned sentinels there would
change the API. The decoder body and first-token targets remain open as well.

## Evidence

Each model/batch directory contains its case, normalized queries, full generated
outputs, output validation, offline report and all candidate-survival curves.
The actual full native GPU logit JSON files remain in the private Mac directory;
their SHA-256 values are in the reports. They are omitted from this checkout's
archive to avoid retaining large reproducible arrays. Source snapshots, native
library/model hashes, all commands/exits and process guards are preserved.

Three failed captures are retained with their exact source versions. The first
two hit the native parser's `case=` keyword limitation before inference. The
explicit-loop edit did not resolve that error. The third corrected the record
key but attempted to build a graph without the required fusion wrapper and
failed during setup. The successful capture uses properly wrapped graphs.
No interrupted or failed capture is included in the successful six-case results.

No production source was changed and no inference speed improvement is claimed.
The full 2x-throughput/half-TTFT objective remains active and unmet.
