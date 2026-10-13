# Exact GPU packing experiments

These are isolated native Metal matrix-vector experiments on the same M4
Mac mini and original BF16 checkpoint values. They do not measure full model
inference, generated outputs, tokens/s, TTFT or a vLLM advantage. No codec was
integrated into Garnet's production backend.

All four suites exited zero. Each suite used all three existing checkpoints,
their complete output head and first-layer MLP down projection, batches 1/2,
and the unchanged native raw BF16 decode kernel as its control. Each model
process completed before the next started. Pre/post process guards retained
the verified WorldSense service and rejected other inference workers.

The CPU roundtrip checked every prepared matrix value. GPU reconstruction
checked those complete matrices, all 65,536 BF16 bit patterns and a random
4097-value tail. All 48 projection comparisons across the four suites were
bit-exact against the raw kernel for the deterministic FP32 test inputs.
Signed zeros, subnormals, infinities and NaN payload bits were covered by the
bit reconstruction tests. The nonfinite pattern cases were not multiplied.
These checks do not replace full-model numerical and generated-output tests.

Each matrix/batch had four ABBA rounds, with three untimed warm operations and
12 measured operations per policy window. A command buffer completed before
the next operation. GPU and wall medians are retained separately. Packing,
compilation, checkpoint/file hashing and CPU validation are outside GPU times.
Both raw and packed buffers were resident during the microbenchmark; the
allocation is not a claim about a production codec's memory footprint.

## Complete output-head results

Each number below is raw GPU time divided by packed GPU time within the same
suite. Values below 1 mean the packed kernel was slower. The controls used
the same four-output-columns-per-SIMD-group native kernel throughout.

| Model | Batch | 3-bit SIMD prefix | 3-bit rank/popcount | 4-bit fast path, four columns | 4-bit fast path, one column |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B | 1 | 0.575 | 0.468 | 0.843 | 0.789 |
| Qwen3.5-0.8B | 2 | 0.562 | 0.459 | 0.842 | 0.698 |
| Qwen3-0.6B | 1 | 0.599 | 0.474 | 0.867 | 0.803 |
| Qwen3-0.6B | 2 | 0.591 | 0.463 | 0.844 | 0.727 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 0.616 | 0.484 | 0.878 | 0.812 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 0.599 | 0.476 | 0.854 | 0.736 |

The 128-value 3-bit codec reduced head storage by about 27%; adding subgroup
ranks reduced it by about 25%. The 4-bit codec reduced head storage by about
22%, with roughly 98.7% of Qwen3.5 blocks avoiding its escape-handling path.
GPU unpacking cost erased these storage savings in every complete head test.
All four head profiles are rejected for integration.

The one-column 4-bit kernel improved the small Qwen3 down projection by
1.54x/1.59x in this isolated suite. Its absolute GPU times were about 0.06 ms;
raw small-matrix timings varied between suites. This is a candidate for a
closer layer/full-model test, not a demonstrated inference gain. Qwen3.5
small-projection results were mixed and coder's small projection regressed.
No full-model speed or 2x-target improvement is claimed.

## Evidence and reproduction

- `lossless-gpu-probe-20261013`: 3-bit tags, SIMD prefix escape indexing.
- `lossless-gpu-rank-probe-20261013`: 3-bit tags, rank bytes and popcounts.
- `lossless-gpu-nibble-probe-20261013`: 4-bit tags, uniform escape-free path.
- `lossless-gpu-nibble-single-probe-20261013`: same exact nibble assets, one
  output column per SIMD group. Every reused input digest was verified.

Each directory retains its exact commands/exits, hardware and source hashes,
combined Metal shader, versioned source snapshot, complete input manifests,
input SHA-256 values, preparation log, GPU checks, individual timing windows
and summary. Reproducible checkpoint-derived `.bin` files and the compiled
probe binary are excluded from this checkout's evidence; the private Mac
investigation directories retain them. The source snapshot corresponding to
each suite can regenerate its inputs from the unchanged checkpoint. The
current probe script additionally supports explicit `--single-column` and
verified `--reuse-assets` modes.

Garnet's selected native state-handoff implementation remains unchanged.
The matched full-model results in the parent report still apply. The
2x-throughput and half-TTFT goal is active and unmet for all agreed profiles.
