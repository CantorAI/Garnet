# Validated direct delta state outputs

The selected native build is `build-delta-split`, library SHA-256
`979489a5e8ab4ca7731b4a74a9c0ab434e84ef1e52563340d172c5662abbecc3`. A graph optimization recognizes a
packed delta update consumed only by its complete result and recurrent-state
slices. The update writes both independent matrices directly, eliminating
the packed temporary and both slice dispatches. Other graph patterns keep
the packed kernel. Update arithmetic, checkpoints, precision and models are
unchanged; session state exchanges include the second output and its aliases.

## Execution and verification

All 11 native validation jobs passed, covering real dense/Qwen3.5 layers and
heads, vision, partial chunk tails, paused rows, gaps, reset and shared graph
sessions. CTest additionally compares both direct outputs bitwise against
the established packed kernel in 32 configurations: batches 1/2, heads 1/3,
key dimensions 7/128, value dimensions 5/128, active/paused rows and padded
layouts. Both pending/aborted outputs remain unreadable; recovery produces
the original values, and in-place state updates are rejected.

A serialized actual batch-2 image diagnostic executed 1,926 DeltaSplit
operations. Total graph operations fell from 48,068 to 44,216, eliminating
3,852 slice dispatches. The complete caption/prompt IDs match the frozen
native output. Per-operation diagnostics deliberately split GPU submissions
and perturb cache/dispatch behavior; their times are not serving benchmarks.

## Close native image comparison

The ABBA control is the previously selected `build-vector-slice`. Both builds
executed 24 real-image requests in total, sequentially, with identical RGB
and patch bytes, fresh vision encoding and matching complete captions. Each
row averages four warm requests per build; initialization and first-use
records are retained separately, with filesystem caches unflushed.

| Batch | Control TTFT, ms | Candidate TTFT, ms | TTFT reduction | Decode gain |
| --- | --- | --- | --- | --- |
| 1 | 616.68 | 594.17 | 3.65% | 1.77% |
| 2 | 916.42 | 853.76 | 6.84% | 3.69% |

These incremental gains support retaining direct outputs. They do not
establish a 2x advantage over vLLM-Metal.

## Fresh three-model vLLM-Metal measurements

The candidate completed another 48 text/coder requests and 24 image requests
against the unchanged isolated comparator on the same M4 16 GiB Mac mini.
Runtime processes never overlap. BF16 checkpoint files, prompts, image
geometry, greedy generation, capacity 256, batches 1/2 and caps remain frozen.
Prefix caching, encoder reuse and speculation are disabled.

| Model | Batch | Garnet tokens/s per row | vLLM tokens/s per row | Garnet TTFT, ms | vLLM TTFT, ms |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 56.77 | 74.77 | 103.05 | 49.68 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 72.56 | 89.24 | 72.00 | 40.66 |
| Qwen3-0.6B | 2 | 52.26 | 75.89 | 173.73 | 70.19 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 70.28 | 88.77 | 100.41 | 46.91 |
| Qwen3.5-0.8B, image | 1 | 46.77 | 48.77 | 596.34 | 201.29 |
| Qwen3.5-0.8B, image | 2 | 43.04 | 47.41 | 852.04 | 329.56 |

All six profiles remain below the 2x decode target and above the half-TTFT
target. Text/coder and image batch 1 preserve exact complete comparator
outputs. Image batch 2 retains the existing different/capped comparator
caption: native finishes 28 tokens, comparator reaches the common 32-token
cap. Its row is a matched-input measurement with a quality/length caveat.
Identical BF16 weights do not imply identical internal execution: native
FP32 vision/activations/KV versus comparator BF16 remains explicit.

The full goal stays active and unmet. Full commands/exits, model/input/source
and library hashes, numerical logs, actual token IDs, runtime guards, first-use
records and warm summaries are saved here. Raw RGB/patch arrays and compiled
binaries stay private on the Mac. `candidate-source/` freezes the changed
implementation and reproduction helpers. Protected GPT-OSS, Pi and original
Mac checkouts remain untouched.
