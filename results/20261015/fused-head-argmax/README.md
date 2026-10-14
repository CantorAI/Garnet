# Fused vocabulary projection and partial argmax: component gain repeats

Two actual isolated Apple M4 runs exited 0 (33.51 and 33.19 seconds). Each
checks 51 captured complete heads and 96 synthetic edge cases, then performs
3,264 timed executions. Independent verification covers all 48 archived files.
This candidate merits full-model integration and testing; it is not selected
for production yet and does not establish faster generation or a vLLM ratio.

The candidate retains the native BF16 learned bits, FP32 input/arithmetic,
four-output SIMD dot products and reductions. It replaces the output epilogue
with a 16-logit group winner and nonfinite flag. A second kernel reduces those
partial winners, preserving the lowest vocabulary index on ties. Every
vocabulary row is evaluated. Timings include both projection and reduction,
one command buffer, normal encoding and completion. CPU prediction comparisons,
compilation and asset preparation are outside the measured intervals.
Validation additionally emits every candidate logit and compares every bit
with the original projection and frozen captured logits. Validation emission
is disabled during timed executions; all timed predictions are checked.

| Model | Batch | Initial candidate/native GPU time | Repeat GPU time | Repeat wall time |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 0.9135 | 0.9230 | 0.9271 |
| Qwen3-0.6B | 2 | 0.8693 | 0.8534 | 0.8638 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 0.9180 | 0.9019 | 0.9126 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 0.8582 | 0.8577 | 0.8736 |
| Qwen3.5-0.8B | 1 | 0.8934 | 0.8979 | 0.9045 |
| Qwen3.5-0.8B | 2 | 0.8561 | 0.8518 | 0.8573 |

Lower is better. These are pooled actual captured-head queries, not complete
model timings. The repeated component GPU reduction is 7.7–14.8%; decode and
TTFT gains cannot be inferred from this percentage. Each query has two ABBA
rounds of eight executions/window after at least 250 ms paired warm-up.

Edge tests cover vocabulary sizes 1/3/17/33, inner dimension 7, B1/B2,
all-zero equal-score ties, negative-zero inputs, positive infinity and NaN
weights, plus untouched output/partial padding. Captured B2 inputs include
different actual states in the two rows. No weights are quantized or pruned.
Projection and greedy semantics are bit-exact in the executed fixtures.

The old read-only head assets are reused only after verifying every weight,
input and captured-reference digest. This is not prefix/image reuse in an
inference comparison; no new full-model serving comparison was executed.
Commands, unchanged source snapshot, generated shader, compiled binary hash,
hardware identity, guards and every timing window are preserved in `evidence/`.

Combined archive SHA-256:
`99e5fb5d92c69f169ad7bb2aa75d28d47a1df6c1bdc1f8459e575dba22b1adce`.
Initial archive SHA-256:
`fc1b04005d6dae166161cf2ae1a63fd21c9b6c7580d41556fa16165389c92008`.

```text
python3 launch_fused_head_argmax_probe.py ROOT ROOT/logs/fused-head-argmax-20261015
python3 launch_fused_head_argmax_probe.py ROOT ROOT/logs/fused-head-argmax-confirm-20261015
python3 archive_fused_head_argmax_probe.py ROOT ARCHIVE
python -X utf8 verify_fused_head_argmax_probe.py RESULTS VERIFICATION_JSON
python -X utf8 verify_fused_head_argmax_probe.py RESULTS CONFIRMATION_JSON fused-head-argmax-confirm-20261015
```

Next required work: add an optional synchronous greedy-head path with bounded
scratch, correct buffer ownership, nonfinite handling, fallback and actual-use
counters. Keep generic compiled-graph logits unchanged. Run all original 19
validation stages, add meaningful fusion/fallback/session recovery checks,
then repeat six-profile whole-model output/latency/throughput comparisons.
Only an actual whole-model gain justifies a new matched vLLM run. The selected
global-alpha reference and protected GPT-OSS/Pi checkouts remain unchanged.
