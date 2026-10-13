# Measured native GPU partial-head prototypes

Four suites actually executed native Metal seed calculation, partial bounds,
shared batch weight reads and the unchanged native greedy argmax on the M4.
Each suite checked all 51 head requests: eight captured states in each model's
batch-1 and batch-2 profiles, plus a batch-2 pair combining different actual
states for each model. That is 78 row/state checks per suite, 312 in total.

The raw control matches every original native GPU logit bit. Every retained
candidate logit also matches bit for bit, every pruned logit is strictly below
its actual seed maximum, and every selected token matches the complete head.
All suites exited zero. No checkpoint value was quantized or replaced. These
are captured-state component checks; accelerated full-model generation did
not execute, and no pruned implementation was integrated into production.

## Timed work and comparison

Both policies execute one command buffer and wait for its completion. The raw
policy runs the complete unchanged native BF16 projection and native argmax.
The candidate runs an exact 8192-row seed projection, GPU seed maximum and
hidden residual-norm preparation, the partial-head kernel, then that same
native argmax. Seed rows are reused, and weights remain shared while either
batch row needs a column. The weight residual norms are precomputed outside
timing with upward FP32 rounding and an allowance. Dynamic bound evaluation
and conservative partial/full-dot error allowances execute on the GPU.

Each request has two ABBA rounds, two warm operations per window and eight
measured operations per policy window. GPU and wall-clock medians are saved
separately. Normalization common to both policies, preparation/upload of
constant weights/norms, compilation and CPU checks/readbacks are excluded.
No other investigation inference runtime was alive during these runs.

Numbers below are medians of per-request raw/candidate GPU-time ratios over
the eight ordinary captured states. Values below 1 mean the candidate was
slower. The mixed-state cases remain in the raw reports and quality checks.

| Model | Batch | One column | Four columns | Four columns, skip final bound | Four columns, check every 256 values |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 0.986 | 0.983 | 1.059 | 1.076 |
| Qwen3-0.6B | 2 | 0.835 | 0.894 | 0.890 | 1.009 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 0.988 | 1.061 | 0.994 | 0.998 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 0.820 | 0.891 | 0.880 | 0.953 |
| Qwen3.5-0.8B | 1 | 0.848 | 0.893 | 0.881 | 0.953 |
| Qwen3.5-0.8B | 2 | 0.764 | 0.822 | 0.803 | 0.913 |

The final profile's median raw/candidate GPU milliseconds were 3.170/2.936
for text batch 1, 3.223/3.195 for text batch 2, 2.831/2.831 and 2.902/3.033
for coder, and 5.260/5.521 and 5.393/5.884 for VL. The modest isolated gains
are not full-model gains and do not meet either goal target. The coder batch-1
gain in the first wide layout did not persist in the later layouts.

Checking every 128 values counted about 26% fewer text, 24% fewer coder and
19% fewer VL weight-element visits, including the seed. The 256-value cadence
reduced bound overhead but skipped less work: about 23%, 15% and 14%. Raw JSON
calls this field `physical_weight_fraction`; it counts shared weight-element
visits from GPU iteration counters, not measured DRAM transactions. Metadata
traffic, cached/prefetched reads and bound arithmetic are reflected in timings,
not in this element count. The faster per-query cases do not establish a
general speed advantage. All four profiles are rejected for broad integration.

## API and remaining work

Pruned columns receive a finite sentinel only in the private greedy prototype.
The unchanged native argmax still checks actual nonfinite outputs. A public
complete-logit operation could not substitute these outputs without changing
its semantics. No API or production source was changed here. Any later use
needs a specialized prediction path, whole-model output validation and matched
serving timing. Decoder projections, prefill and vision remain optimization
work; the full three-model 2x/half-TTFT goal remains active and unmet.

## Evidence and reproduction

- `pruned-head-gpu-20261013`: one-column partial-head kernel.
- `pruned-head-gpu-wide-20261013`: four columns, bounds every 128 values.
- `pruned-head-gpu-no-final-20261013`: omits the bound after a complete row.
- `pruned-head-gpu-interval2-20261013`: additionally checks every 256 values;
  launched only after the verified preceding owned job exited zero.

Each directory retains exact source snapshots/hashes, combined shader, hardware,
commands/exits, pre/post process guards, input manifests, all checks and timing
windows, and summaries. Reused inputs were verified by every saved digest.
Checkpoint-derived `.bin` inputs and the compiled binary are omitted from this
checkout's archive and retained privately on the Mac. The full normalized
queries/full-logit capture evidence is linked by digest in the manifests and
the preceding `partial-head/` results. This source can regenerate the inputs.
