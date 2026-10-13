# Validated wide prefill tiles

The selected native candidate is `build-prefill-wide`, library SHA-256
`01ac8c4f04a91f20fcb3108ba451c3e54fc77daf3c6bc0f6103aa803848ab741`.
It uses a 16x64 output/K64 tile for 16-row prefill projections and a 32x64
output/K32 tile for larger row counts through 32. Small row/output shapes keep
the established kernel; unsupported optional pipelines also fall back. Decode
still uses the original BF16 matrix-vector kernel. Weights remain exact BF16,
with FP32 inputs/accumulation/output. No quantization or model change is involved.

## Selection and verification

`prefill-tile-gpu-20261013` compared five shared-memory geometries against the
unchanged 16x32/K32 kernel using real first-layer matrices from all three models.
All 840 checks were bit-exact, including row counts 3/5/16/17/31/32, irregular
matrix tails and padded input/output strides. The 280 timed matrix profiles
use two ABBA rounds and 12 operations per window. Larger K-only tiles mostly
regressed; wider output tiles gave modest component gains. Isolated timings
were used to choose a candidate, not to claim an inference improvement.

The production candidate then passed all 11 checks: CTest, serving/reset/shared
graph behavior, mixed-image pipeline, real Qwen3.5 linear/full-attention chunks,
both real dense model references, all three complete heads, and small/pretrained
vision. Partial tails, paused rows and chunk sizes 2/4/8/16 are covered by the
Qwen3.5 tests. The actual generated outputs are validated below.

## Close native controls

The text/coder ABBA comparison executed 48 requests, and the native real-frame
ABBA comparison executed another 24. Every generated token ID, prompt ID and
completed-output flag matches the control. Each profile has two fresh processes
per build, one first-use plus two warm requests per process. This table averages
the four warm requests; full first-use and initialization records remain saved.

| Model | Batch | Control TTFT, ms | Candidate TTFT, ms | Reduction |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 105.54 | 103.63 | 1.81% |
| Qwen3-0.6B | 2 | 190.74 | 173.81 | 8.88% |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 75.28 | 71.87 | 4.54% |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 112.31 | 100.70 | 10.33% |
| Qwen3.5-0.8B, actual image | 1 | 697.09 | 692.04 | 0.72% |
| Qwen3.5-0.8B, actual image | 2 | 1186.92 | 1133.62 | 4.49% |

Decode ratios against the controls are approximately 0.995–1.000. There is no
decode-speed gain demonstrated by this prefill-only change. The smallest
batch-1 TTFT changes are modest; the batch-2 reductions justify keeping the
candidate. The image clocks include the common CPU frontend, fresh native
vision encoding and worker protocol. Native frame controls use the same
RGB/patch bytes and complete the same 28-token captions in both batch sizes.

## Updated actual vLLM comparisons

The same candidate completed 48 matched text/coder requests and 24 matched
image requests against the unchanged isolated comparator. Processes ran
strictly sequentially. Prefix reuse/speculation stayed disabled; all checkpoint
files, prompts, image geometry and caps match the frozen profiles. Each runtime
again has separate first-use and warm records, with initialization separate
and filesystem caches unflushed. All workers shut down cleanly.

| Model | Batch | Garnet tokens/s per row | vLLM tokens/s per row | Garnet TTFT, ms | vLLM TTFT, ms |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 56.42 | 74.55 | 103.82 | 50.15 |
| Qwen3-0.6B | 2 | 52.31 | 76.37 | 173.78 | 64.09 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 72.44 | 92.45 | 71.88 | 34.86 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 70.03 | 92.67 | 100.07 | 38.16 |
| Qwen3.5-0.8B, actual image | 1 | 43.59 | 48.72 | 692.52 | 202.55 |
| Qwen3.5-0.8B, actual image | 2 | 37.44 | 47.62 | 1139.68 | 327.35 |

Text/coder and VL batch 1 preserve exact complete outputs against vLLM. VL
batch 2 retains the existing different/capped comparator-caption limitation:
native completes 28 tokens, comparator reaches the common 32-token cap. Its
results are matched-input measurements with that output-quality/length caveat.
Native FP32 vision/activations/KV versus comparator BF16 arithmetic remains
disclosed; unchanged checkpoints are not bit-identical internal execution.
All six profiles fail both the 2x-throughput and half-TTFT targets. This is a
validated incremental improvement, not completion of the active objective.

## Evidence and reproduction

The directories retain all actual commands/exits, hardware/model/library/input
hashes, process guards, complete token/caption outputs, numerical logs and
summaries. `prefill-wide-source-20261013` snapshots the candidate C++ source and
all runners. The earlier comparator runner version is saved separately because
native-frame ABBA support was added after the matched vLLM jobs started. The
new runner records its own source digest and selects the appropriate native
build for both frame backends. The native-frame summarizer validates real
execution, complete equal outputs and input hashes before reporting ratios.

Raw checkpoint-derived matrix/patch/RGB arrays and compiled binaries are
excluded from the evidence checkout and retained privately on the Mac or
reproducible from pinned inputs. No original Mac repo, GPT-OSS checkout or Pi
checkout was changed. The full three-model objective remains active and unmet.
