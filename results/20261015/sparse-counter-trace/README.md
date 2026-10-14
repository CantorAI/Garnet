# Sparse GPU counters: much lower diagnostic overhead

The actual 24-worker ABBA run exited 0 in 122.52 seconds. All 72 complete
requests match the frozen tokens, prompt IDs and finished flags; images are
freshly encoded. Independent verification covers 438 files, 12,132 timestamp
samples and 660 command summaries. Every sparse mask, encoder ordinal, sample
weight and contiguous sample index is checked. Shader payloads are unchanged.

Every vocabulary-head encoder is sampled. Other encoders rotate through one
in 31 according to `(encoder_ordinal + command_sequence) % 31 == 0`.
Original command-buffer boundaries remain intact, and profiling is not serialized.

| Model | Batch | Quiet row tokens/s | Traced row tokens/s | Observed slowdown |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 59.3027 | 57.8709 | 2.41% |
| Qwen3-0.6B | 2 | 57.4833 | 56.1775 | 2.27% |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 72.1496 | 70.4750 | 2.32% |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 70.5708 | 68.9272 | 2.33% |
| Qwen3.5-0.8B | 1 | 47.7316 | 46.9494 | 1.64% |
| Qwen3.5-0.8B | 2 | 44.3673 | 43.4579 | 2.05% |

Each mean uses four warm requests across two fresh workers, first requests
excluded. This within-run instrumentation comparison improves on the earlier
full-sampling run's 25–45% observed slowdown. It is not a serving optimization
or a fresh comparator baseline. TTFT changes in this run range from -1.59%
to +0.76%; small changes do not establish a latency improvement.

`overhead.json` reports whole-trace sampled-shape ranks, including first and
warm requests, prefill and decode together. Sparse missing shapes cannot prove
a command is decode-only, so the earlier full-trace classification heuristic
is deliberately not reused. Heads are oversampled; raw ranks are not unbiased
serving-time shares. Raw ticks are not converted to milliseconds, and overlapping
sample intervals must not be summed into an end-to-end clock.

Archive SHA-256: `923b4f010be883d7492ece7a03f36e7759e11d786431b5fa5b8ad65812c877f7`.
Commands, guards, full diagnostic source copy, checkpoint proofs and complete
outputs are in `evidence/`.

```text
python3 run_sparse_counter_trace_profiles.py ROOT ROOT/logs/counter-trace-sparse-profiles-20261015/jobs
python3 archive_sparse_counter_trace_profiles.py ROOT ARCHIVE
python -X utf8 verify_sparse_counter_trace_profiles.py EVIDENCE VERIFICATION_JSON
python -X utf8 summarize_sparse_counter_trace_profiles.py EVIDENCE OVERHEAD_JSON
```

Production selection is unchanged. The full 2x / half-latency goal remains unmet.

`prediction-roles.json` additionally aligns every forced vocabulary-head event
with the complete request's recorded prediction-call count. Each worker has
one head context and one head per prediction command; vocabulary/hidden/batch
dimensions match its preserved model configuration. Request order, exact head
counts and decode shapes are checked. This is an inference from the archived
source's synchronous call contract, not an explicit runtime phase marker.
No role is selected from absent sampled shapes.

Across the two traced workers, warm decode contributes 28 text, 64 coder and
108 VL prediction commands per batch profile; each has four separate warm
final-prefill head commands. Earlier prefill chunks and vision commands without
a head are excluded. The saved ranks isolate these cohorts, but heads remain
oversampled and intervals may overlap; ranks are not serving-time percentages.
This dataset describes the selected-global-alpha diagnostic before the new
fused-head routes. Its timing values cannot be substituted for the new build.

```text
python -X utf8 summarize_sparse_prediction_roles.py EVIDENCE PREDICTION_ROLES_JSON
```
