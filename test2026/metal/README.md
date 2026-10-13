# Garnet native Metal/MPS: Qwen vision, text and coder

Actual text and image-plus-text inference work on the Apple M4 Mac mini
(16 GiB, macOS 15.6.1), using Garnet's native xlang3 package, captured xModel
graphs, Metal kernels and MPS matrix multiplication. Learned projections,
normalization, attention and full vision-tower computation execute on the
Apple GPU. Tokenization, image preprocessing, embedding lookup,
and request control use the host CPU. The default decoder retains attention
state and intermediate hidden values on the GPU across layers, and selects
answer tokens on the GPU. Executed adapters also include Qwen3-0.6B and
Qwen2.5-Coder-0.5B-Instruct; the latest three-model measurements are below.

Most learned projections use FP32 MPS matrices. Vocabulary heads retain the
original BF16 checkpoint values and accumulate in FP32, without additional
quantization. These remain experimental adapters with measured M4 profiles;
remaining performance and serving limits are described below.

## Isolation and checkpoint

Windows source: `D:/CantorAI/Garnet-metal`, branch `feature/qwen35-metal`.
Mac source/build: `/Users/zsdx/garnet-metal-investigation/{source,build}`.
The existing Garnet checkouts and shared xlang3 SDK/runtime are reused read-only.
`feature/gpt-oss-120b` and the Pi investigation branch were not edited.
The existing Mac checkout remained clean.

Qwen/Qwen3.5-0.8B, pinned Hugging Face revision
`2fc06364715b967f1860aea9cf38778875588b17`, BF16 checkpoint:
1,746,942,600 bytes, SHA256
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
The Mac downloaded the exact file directly from the official ModelScope mirror;
full file and config/tokenizer/preprocessor hashes were checked before inference.
The saved verified SSH route was retained; no credentials or host-key bypasses
were used. Background jobs preserve command, output and exit status despite
intermittent network connections.

## Actual measurements, 2026-10-11 Pacific (2026-10-12 UTC logs)

The table reflects the chunk-4 prefill build. Each process loads the
model once and runs two independent requests, resetting
state between them. The table uses the second request. Output limits apply
per batch row. Timings exclude CPU image preprocessing, which was performed
beforehand, and network transfer. First-token time includes vision and decoder
prefill. Decode throughput excludes the initial token produced by prefill.

| Workload | Batch | Output | Loaded first token, s | Loaded result, s | Decode tokens/s per row | Combined tokens/s | Process peak RSS, GiB |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Text: local-model benefits, chunk 4 | 1 | 32-token limit | 0.810 | 2.986 | 14.25 | 14.25 | 5.04 |
| Text: local-model benefits, chunk 4 | 2 | 32-token limit each | 1.012 | 3.617 | 11.90 | 23.81 | 5.24 |
| Image, chunk 4 | 2 | 28 tokens including stop each | 4.428 | 6.927 | 10.80 | 21.61 | 5.64 |
| Image, complete caption, chunk 4 | 1 | 28 tokens including stop | 3.293 | 5.391 | 12.87 | 12.87 | 5.43 |

Peak RSS is measured for the whole native process in macOS bytes, converted
to GiB, and is not isolated GPU allocation. It varies across runs and MPS
allocation profiles. The Mac had other workloads: a final system snapshot
showed 3,741 MiB swap in use, so these are development-machine observations,
not isolated laboratory results or percentile latency guarantees.

Text initialization was about 2.85 s for the short Paris test; the first
request including startup reached its first token in 4.77 s. Those short-test values are from the earlier build. The allocation-reuse build
had longer-text startup plus first token at 3.36 s. With lazy chunk-4 compilation,
text initialization is 2.07 s and first token including startup is 4.21 s;
image initialization is 2.91 s and first token including startup is 7.37 s.
The first chunked request includes extra graph compilation, while subsequent
requests reuse those graphs. The faster warm result must not be presented as
a cold-start timing.
The longer text test reached its token cap rather than EOS; the final captions
finish normally at 28 tokens per row. Batch 2 repeats the same input on both rows;
independent row controls are tested separately in the integration fixture.

The Paris prompt generated `Paris` and the normal stop token. The real-image
caption was:

> A large statue of a man with a golden headband stands on a balcony overlooking a densely populated city nestled in a mountainous landscape.

The source photo was inspected and these main objects match. This is a
qualitative sanity check, not a VLM accuracy evaluation. Source image SHA256:
`eba5d9b6124fbd3b2a809e1c664886c29d842406134bb4c1b281414d0c7101b9`.
Original 1818 x 1372 pixels; resized to 288 x 192, patch grid 18 x 12,
54 merged visual tokens and 79 total prompt tokens. Repeated-frame tests
re-encode this same image each time with weights resident; they do not measure
moving video, camera capture or temporal understanding.

## Correctness and optimization

- Native primitive CTest: 1/1 pass after final build, 0.59 s.
- Full-size pretrained linear layer 0 and attention layer 3: batches 1/2,
  four steps each, all hidden/recurrent/KV outputs compared with independent
  FP32 NumPy references, including paused/resumed rows and invalid-slot rejection.
- All 12 pretrained vision blocks and merger: batches 1/2 against an independent
  reference; final max absolute errors 2.15e-6 / 2.63e-6. Original tolerance
  remains `5e-6 + 5e-5 * abs(reference)`; worst tolerance ratios below 0.40.
- Synthetic mixed vision/linear/full-attention/head fixture passes batches 1/2
  after final build, two requests and 14 steps per request, retaining weights,
  resetting state and pausing/resuming batch rows.
  Final logs: `pretrained-final-20261012` and `pipeline-final-20261012`.

Real pretrained weights exposed non-finite values in the native tanh GELU
at large arguments. Clamping the argument to [-10, 10], where the FP32
result already saturates, fixes that failure without relaxing tolerance.
Temporary stage-selection diagnostics were removed.

Vision LayerNorm originally repeated the same row sums for every output
column. Computing row statistics once preserves the reference result and
reduces redundant work. On the same real image, second-request vision time
fell from 0.553 to 0.351 s (36.5% lower); loaded first-token latency fell from
6.256 to 5.649 s (9.7% lower). Generated tokens remained identical. These are
single before/after runs; prefill and system pressure also varied. Batch-2
vision time fell from 1.033 to 0.709 s, while total first-token time remained
about 8.8 s because decoder prefill varied.

Large reference tensors now use binary FP32 fixtures. Sampling the original
full-size test showed minutes spent parsing numeric JSON inside xlang3 before
GPU dispatch. This improves test execution, not reported model throughput.

## Decoder allocation reuse, 2026-10-11 Pacific

Graph input buffers now persist across token steps. Host tensors copy directly
into these existing allocations rather than into a temporary vector followed
by a newly allocated GPU buffer. Small position/slot/active controls still
undergo pre-dispatch validation. Host updates reject wrong dimensions, pending
outputs, a foreign device and an open command batch.

MPS matrix-multiplication objects are cached per graph context by result rows,
result columns, inner width and transpose mode. Weights, matrix strides and
operands remain explicitly bound on every encode. Captured graph math and
checkpoint precision are unchanged. No GPU-resident recurrent state claim is made by this allocation optimization;
the subsequent chunked-prefill implementation is described below.

A fresh pre-change caption baseline was run immediately before rebuilding:

| Batch-1 warm complete caption | Before | After | Change |
| --- | ---: | ---: | ---: |
| Decoder prefill, s | 5.251 | 4.685 | 10.8% lower |
| First token, s | 5.612 | 5.037 | 10.3% lower |
| Complete caption, s | 7.908 | 7.101 | 10.2% lower |
| Decode tokens/s | 11.76 | 13.08 | 11.2% higher |
| Process peak RSS, GiB | 8.17 | 4.81 | 41.2% lower |

Peak RSS comparisons are sampled process high-water marks, including MPS
resources, rather than isolated weight/GPU allocation. Memory pressure and
other workloads can affect them; no fixed resident-memory budget is promised.
The first allocation-reuse stage already reached 7.168 s for the warm caption;
removing the temporary host copies additionally reached 7.101 s. Their separate
contributions have not been isolated with repeated factorial benchmarks.

Final primitive CTest passes 1/1 (0.52 s), including padded/repeated host
updates and invalid updates. Full-size pretrained decoder and the full vision
tower pass for batches 1/2 at unchanged tolerances. The four real workload
benchmarks run sequentially through `benchmark_optimization.py`. Both requests
in every workload produce exactly the earlier token IDs and finish status:
batch-1 caption finishes at 28 tokens; capped text and batch-2 captions remain
explicitly unfinished. Batch-2 image first-token latency improves from the
previous 8.763 s observation to 6.372 s; that comparison uses the earlier
saved run, unlike the freshly paired batch-1 baseline.

All commands, native outputs, timings and successful exit status are saved in
`results/20261012/allocation-reuse/`, with decoded records in `benchmarks.json`.
Vision reference tensors also use binary files now, retaining the original
reference and tolerance while avoiding the numeric JSON parsing bottleneck.

## Native chunked prefill, 2026-10-11 Pacific

Metal lowering now supports stride-aware reshape, chunk token extraction,
chunk masks and sequence concatenation. Prompt normalization and MLP projections
run across a chunk; causal convolution, GatedDelta and full-attention state
updates remain in token order within one GPU submission per layer/chunk.
Position, slot and active controls extracted from chunks are checked on the
host before GPU dispatch. No learned operation falls back to the CPU.

`MetalDecoder.enable_prefill(4)` enables the measured profile; call
`prefill_sequence(hidden_tokens, positions)` for an entire prompt, then use
ordinary `step`/`step_hidden` for generation. `hidden_tokens` is a list of
batch-by-hidden native tensors, and positions are batch-by-token-by-THW lists.
Chunks 2 and 4 are supported. The existing sequential path remains available.
Graphs compile lazily on first use, remain cached across request resets, and
are released when changing chunk size or closing the decoder. A partial final
chunk is zero-padded and masked; only actual prompt tokens advance the slots.

Fresh sequential/chunked measurements use the same checkpoint, 79-token prompt,
288 x 192 image, 32-token output limit and two requests per process. The image
caption finishes at 28 tokens including EOS. All caption token IDs and finish
status match exactly between sequential, chunk 2 and chunk 4.

| Warm batch-1 image request | Sequential | Chunk 2 | Chunk 4 |
| --- | ---: | ---: | ---: |
| Prefill, s | 4.682 | 3.658 | 2.936 |
| First token, s | 5.037 | 4.011 | 3.293 |
| Complete caption, s | 7.101 | 6.102 | 5.391 |
| Process peak RSS, GiB | 4.81 | 5.44 | 5.43 |

Chunk 4 lowers warm prefill by 37.3%, first-token wait by 34.6%, and complete
caption time by 24.1% versus the freshly paired sequential run. It does not
increase decode throughput: 12.87 versus 13.08 tokens/s in these runs. The
cached prefill graphs increase measured peak process memory by about 0.62 GiB.
The first chunk-4 request spends 1.1 s compiling its 24 prefill graphs; its
loaded first-token wait is 4.46 s, versus 5.13 s for sequential. For the shorter
text prompt, lazy graph compilation makes the first chunked request slower
(loaded first token 2.14 s), followed by the 0.81 s warm result. Text warm
comparisons use the earlier saved allocation-reuse run, not a fresh text pair.

Batch-2 complete image captions improve from a paired sequential 6.282 s first
token / 8.756 s complete result to 4.428 s / 6.927 s with chunk 4. Both rows
match exactly and finish at 28 tokens. Text chunk-4 outputs match the earlier
sequential 32-token capped outputs exactly for batches 1/2 and both requests.

Correctness evidence on the M4:

- Full-size pretrained linear layer 0 and full-attention layer 3, batches 1/2
  and chunks 2/4: all chunk hidden outputs and final recurrent/KV states match
  independent sequential FP32 NumPy references at the unchanged tolerance.
  Each profile runs two chunks, including paused/resumed rows and partial tails.
- Invalid active flags, out-of-capacity slots and non-finite positions are
  rejected before dispatch; subsequent valid executions still pass.
- Serving-level small BF16 fixture, batches 1/2: chunks 2/4/2 agree with sequential
  predictions and final states; profile changes, request reset and partial tails
  pass. This fixture also checks padded widths through native token stacking.
- Primitive CTest passes 1/1 after the final chunk build, 0.61 s.

Raw command/exit/output evidence and decoded timing records are in
`results/20261012/chunk-prefill/`. `benchmark_chunk.py` runs the workloads
sequentially through background mode `chunk-suite`; `image-chunk:1:4` and
`text-chunk:1:4` run individual image/text jobs. Chunks are opt-in, and benchmark
scripts record `prefill_chunk` and first-request compile time explicitly.

## Batched attention input projections — 2026-10-12

Chunk lowering now projects the whole normalized input chunk once per attention
weight, then extracts each token's projection on the GPU. Convolution, gated
DeltaNet recurrence and causal KV updates retain their sequential order. This
uses the existing native Garnet Metal/MPS backend and FP32 checkpoint weights.

Actual M4 runs with the same image, 79 prompt tokens, chunk 4 and 28-token
completed caption:

| Warm batch-1 image request | Previous build repeat | Batched projections repeat |
| --- | ---: | ---: |
| Prompt processing | 2.953 s | 2.406 s |
| First token, including vision | 3.307 s | 2.759 s |
| Complete caption | 5.419 s | 4.867 s |
| Answer generation | 12.78 tokens/s | 12.81 tokens/s |

The first new run measured 2.786 s first token / 4.911 s complete, confirming
this repeat. First-token latency improves about 17%, prompt processing about
19%; generation speed is essentially unchanged. The initial old-build run was
slower (3.793 s / 5.921 s), so it is not used for the conservative percentage.
Batch 2 measures 3.906 s first token / 6.446 s for both complete captions,
versus the earlier saved chunk-4 4.428 s / 6.927 s; that is a historical
comparison rather than a fresh batch-2 pair. Both rows and both requests retain
exactly the same token IDs and finish normally.

Text batch 1 measures 0.658 s warm first token / 2.830 s for the 32-token
capped output; it matches the earlier chunk-4 output tokens. It remains capped,
not a completed answer. First image request with new graphs takes 4.056 s to
first token and 6.198 s to completion after 2.949 s model initialization:
approximately 7.005 s / 9.147 s from process-side model initialization.
Warm numbers exclude initialization and cached-graph compilation. New batch-1
image peak native RSS is 5.45–5.46 GiB; the previous-build repeat is 5.47 GiB.
RSS is process memory, not an isolated GPU allocation measurement; workloads
and macOS caching can affect timing and memory.

Full-size pretrained layer references pass batches 1/2 and chunks 2/4 at the
unchanged tolerance for all hidden outputs and final states, including paused
rows, partial tails and invalid controls. Serving lifecycle tests pass reset,
chunks 2/4/2 and profile replacement for batches 1/2. Native primitive CTest
passes 1/1. Raw commands, exit codes and outputs from eight successful jobs
are saved in `results/20261012/batched-projections/`, with parsed
`benchmarks.json`. The previous-build runs use commit `0bceee4`; other jobs
use this change. The optimized source and library are restored on the Mac.

## Eight-token prefill profile — 2026-10-12

`MetalDecoder.enable_prefill(8)` now supports an opt-in larger chunk with the
same native Metal/MPS computation. Compared with chunk 4, the 79-token image
prompt needs 10 rather than 20 chunk submissions per layer, reducing state
transfers and using larger batched projections. Causal state updates remain
sequential within each chunk. Default `enable_prefill()` remains chunk 4.

| Warm image request, batch 1 | Fresh chunk-4 control | Chunk 8 | Chunk-8 repeat |
| --- | ---: | ---: | ---: |
| Prompt processing | 2.414 s | 1.968 s | 1.955 s |
| First token including vision | 2.770 s | 2.319 s | 2.307 s |
| Complete 28-token caption | 4.871 s | 4.430 s | 4.398 s |
| Answer generation | 12.86 tokens/s | 12.79 tokens/s | 12.92 tokens/s |

First-token latency improves another 16–17% against the fresh chunk-4 control;
generation speed is unchanged. All image token IDs and finish flags match the
previous build for both requests and batches 1/2. Batch 2 takes 3.420 s to first
token and 5.982 s for both completed captions (10.54 tokens/s per row). This
batch-2 comparison uses the prior saved chunk-4 run, not a fresh paired control.
Text batch 1 takes 0.531 s warm first token / 2.705 s for the unchanged 32-token
capped output; this is not a completed answer.

Cold image graph use takes 3.544–3.608 s first token / 5.670–5.729 s complete
after 2.925–2.990 s model initialization. Process initialization through first
token is therefore approximately 6.47–6.60 s. Chunk-8 batch-1 peak native RSS
is 5.44–5.45 GiB versus the fresh chunk-4 5.44 GiB; this does not establish an
allocation reduction. Timing varies with other Mac workloads and caching.

Full-size pretrained linear/full-attention references pass batches 1/2 with
chunks 2/4/8, including two successive chunks, partial tails, paused/resumed
rows, invalid controls, all hidden outputs and final states at unchanged
tolerance. Serving lifecycle tests pass chunks 2/4/8/2 and profile replacement.
Seven successful jobs and exact commands/exits are saved under
`results/20261012/chunk8/`, with parsed `benchmarks.json`.
Reproduce with background modes `image-chunk:1:8`, `image-chunk:2:8` and
`text-chunk:1:8`. Chunk 8 is the fastest executed image profile on this M4;
this section measures through chunk 8; the following section measures chunk 16. Other Apple devices remain unmeasured.

## Sixteen-token prefill profile — 2026-10-12

`MetalDecoder.enable_prefill(16)` supports processing up to 16 prompt tokens
per chunk. This is prompt length per GPU graph execution, independent of
request batch size and answer-token limits. The 79-token image prompt needs
5 chunks per layer rather than 10 with chunk 8. State transitions remain
causal and sequential. The default profile remains chunk 4.

| Warm batch-1 image request | Fresh chunk-8 control | Chunk 16 |
| --- | ---: | ---: |
| Prompt processing | 1.956 s | 1.746 s |
| First token including vision | 2.316 s | 2.102 s |
| Complete 28-token caption | 4.418 s | 4.208 s |
| Answer generation | 12.84 tokens/s | 12.82 tokens/s |

This is another 9% first-token improvement with identical output tokens.
The repeat's warm result is 2.080 s / 4.198 s, but its cold request briefly
overlapped the batch-2 job, so the repeat is excluded from the primary paired
comparison. The original batch-2 job is also excluded from performance claims.
Its correctness outputs are retained. An isolated rerun of batch 2 measures
2.951 s first token / 5.518 s for both complete captions, 10.52 tokens/s per
row. Compared with the prior saved chunk-8 3.420 s / 5.982 s, that is a
historical comparison, not a fresh batch-2 pair.

The uncontended first batch-1 request takes 3.420 s to first token / 5.535 s
complete after 2.935 s initialization, approximately 6.354 s / 8.470 s including
initialization. Peak native RSS is 5.47 GiB for batch 1 and 5.70 GiB for the
isolated batch-2 run; these are process memory measurements. Answer generation
speed is unchanged. Text-only chunk 16 has not been benchmarked.

Full-size pretrained layer tests pass chunk 16 for batches 1/2, two successive
chunks, paused/resumed rows, partial tails, invalid controls, all hidden
outputs and final states at unchanged tolerance. Serving tests pass profile
switching 2/4/8/16/2 and partial padding. The reference test now uses capacity
32 to accommodate two 16-token chunks, and retains profiles 2/4/8/16 by default;
the focused saved reference job executed only chunk 16. Evidence for the
smaller profiles is retained in the preceding rounds. Raw commands, exit codes
and parsed timing data for seven successful jobs are in
`results/20261012/chunk16/`. Reproduce with `image-chunk:1:16` or
`image-chunk:2:16`. Larger chunks remain unmeasured.

## Removing unused attention-state packing — 2026-10-12

Metal graph lowering now aliases complete segments of packed attention results
back to their existing buffers and traces graph dependencies to remove unused
packing/unpacking operations. Replacing the hidden result packs the final
hidden/state buffers directly. Unused GPU allocations are released after
lowering. Caller-owned states, masking, validation and FP32 math are preserved.
Graphs expose `operation_count` for diagnostics; image logs include per-layer
counts. This reduces GPU copies inside a graph; it does not yet keep states
on the GPU across separate layer calls.

| Warm batch-1 image, chunk 16 | Fresh previous build | New build | Repeat |
| --- | ---: | ---: | ---: |
| Prompt processing | 1.728 s | 1.555 s | 1.558 s |
| First token including vision | 2.084 s | 1.908 s | 1.908 s |
| Complete 28-token caption | 4.198 s | 3.954 s | 3.969 s |
| Answer generation | 12.78 tokens/s | 13.19 tokens/s | 13.10 tokens/s |

Warm first-token latency improves about 8.5%; generation improves about 3% in
these runs, so this is a small measured decoding gain rather than a large one.
All image and text output IDs match exactly. Batch 2 measures 2.599 s first
token / 5.041 s for both captions and 11.05 tokens/s per row; compare with the
prior saved isolated 2.951 s / 5.518 s as a historical comparison. Text batch 1
is 0.506 s first token / 2.649 s for the same 32-token capped output, not a
completed answer. Model-loaded first image request takes 3.276 s first token /
5.363 s complete after 2.995 s initialization. Batch-1 process peak RSS stays
around 5.45 GiB, so no process memory reduction is established.

Full-size pretrained layers pass batches 1/2 and chunks 2/4/8/16, including
paused rows, partial tails, invalid controls, all hidden outputs and final
states at unchanged tolerance. Serving reset/profile replacement tests and
native primitive CTest pass. Seven successful job command/exit/output records
and parsed timing data are in `results/20261012/state-copy/`.
The previous build is commit `31571af`; new runs use this lowering change.
The following section records the dedicated Qwen text and coder adapters requested next by the user.

## Dedicated Qwen text and coder adapters — 2026-10-12

`xModel/qwen/metal_dense/decoder.py` now provides `DenseMetalDecoder` for
native Garnet Metal/MPS text generation. Executed checkpoints are
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) and
[Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct).
The adapter shares prefill scheduling, request reset and explicit KV serving
with the Qwen3.5 Metal adapter. It adds plain RMS weights, ungated attention,
Qwen3 Q/K normalization, standard text RoPE, Qwen2.5 QKV biases and configurable
GPU vocabulary projection. Token embedding lookup, state copies and greedy
argmax remain on the CPU; all learned projections and attention run on GPU.

Actual M4 warm measurements with chunk 16 and the model resident:

| Model / prompt | Batch | Completed tokens per row | First token | Complete answer | Decode tokens/s per row | Peak process RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B: capital of France, 22 input tokens | 1 | 8 including stop | 0.608 s | 1.108 s | 13.99 | 4.16 GiB |
| Qwen3-0.6B: same prompt on both rows | 2 | 8 including stop | 0.727 s | 1.305 s | 12.12 | 4.28 GiB |
| Qwen2.5-Coder: Python add function | 1 | 17 including stop | 0.347 s | 1.132 s | 20.38 | 3.34 GiB |
| Qwen2.5-Coder: same prompt on both rows | 2 | 17 including stop | 0.390 s | 1.217 s | 19.34 | 3.39 GiB |

The text answer is “The capital of France is Paris.” The coder returns a fenced
`def add(a, b): return a + b` function. Its AST contains only the expected
function, names, addition and return; the generated function passes four cases
covering positive/negative integers, zero and floating-point addition.
This is a functional coding smoke test, not a coding benchmark suite.
Sequential and chunk-16 generation match token IDs exactly for both requests;
batch-2 rows also match. The requests reset KV state and reprocess the prompt.
Sequential warm first-token latency is 1.292 s for text and 1.042 s for coder.
Sequential peak RSS is lower: 3.23 GiB / 2.71 GiB respectively.

Lazy compilation costs matter for short requests: first model-loaded chunked
text/coder requests take 2.012 s / 1.451 s to first token, after 1.563 s /
1.254 s initialization. Their first complete answers including initialization
are approximately 4.098 s / 3.510 s. Sequential first requests are faster than
these cold chunked requests. Warm timings and decode rates are for these short
answers, not a long-output throughput claim. Peak RSS is native process memory,
not isolated GPU memory. No other Apple hardware or larger checkpoint is claimed.

Independent references execute real first and last pretrained layers for each
model, batches 1/2, chunks 1/16 and two successive chunks (capacity 32). They
check paused/resumed rows, partial tails, invalid flags/slots/positions, every
hidden vector and final KV state. Dense reference dot products and RMS
reductions accumulate in FP64 and round values back to FP32; runtime inference
remains FP32. Hidden vectors and each cached value vector are checked in both
Euclidean and infinity norms with `5e-6 + 5e-5 * reference_norm`; rotated key
pairs use the same Euclidean norm bound. These are **vector-based criteria**,
not the older Qwen3.5 component-relative criterion. Cancellation-sensitive
component tests failed on random final-layer inputs before this change. The
raw initial failure and wider-reference failure are retained. Final vector
checks pass, and logs report maximum component error as well as normalized
error ratios. Qwen3 final-layer maximum component error reaches 0.00818 on
large-valued activations; hidden vector relative error remains below 5e-6 in
the executed cases. This is not a Transformers end-to-end numerical parity claim.
Qwen3.5's existing component reference criteria remain unchanged and pass.

Download manifests in `dense_models/` pin all four files by size and SHA-256
from the official Qwen ModelScope repositories. Both Mac downloads verify every
file before publishing its final name; raw download logs retain the hashes.
Use `download_dense_checkpoint.py manifest.json model_directory` to reproduce.
The full checkpoints are downloaded inputs and are not committed.

Run Mac Python with `benchmark_dense_text.py runtime build source model_directory
batch output_limit prompt chunk`. The background modes are
`dense-text:Qwen3-0.6B:1:16`,
`dense-text:Qwen2.5-Coder-0.5B-Instruct:1:16` and
`dense-reference:model_name`; substitute batch 2 or chunk 0 for the executed
alternatives. The model API is `DenseMetalDecoder(checkpoint,config,batch,capacity)`,
`enable_prefill(16)`, `prefill_sequence(...)`, `step(...)`, `reset()` and `close()`.
This is an experimental native adapter rather than a paged/continuous-batching
service. Only the two named checkpoints, fixed batches 1/2 and short contexts
have been executed. Scaled RoPE, sliding-window attention and non-SwiGLU MLP
profiles are rejected. Quantized weights and long-context serving remain future work.

Raw commands, exit/output records, parsed benchmarks and generated-code
validation are in `results/20261012/dense-text-coder/`. The final shared library
also repeats Qwen3.5 image inference with the same 28 token IDs, 1.928 s warm
first token / 3.996 s completion. Full-size Qwen3.5 component references and
native primitive CTest pass after the shared kernel changes.
After sharing the prefill scheduler, additional text/coder batch-2 requests
retain identical tokens, the serving fixture passes reset/profile changes
2/4/8/16/2, and the repeated image caption is unchanged at 1.919 s / 3.993 s.
The saved folder contains 18 job records, including two earlier diagnostic
failures; final verification jobs exit successfully.


## Shared optimization for all three models — 2026-10-12

The default `MetalDecoder` and `DenseMetalDecoder` now use a native
`garnet.decoder_session`. Each session owns private GPU recurrent/KV state;
all decoder layers, final normalization, vocabulary projection and greedy
selection execute in one command submission per generated token. A prompt
chunk similarly submits all layers together. Intermediate hidden values and
states stay on the GPU. Request flags/positions and embedding lookup remain
on the host, and inference remains synchronous at the request-step boundary.

The shared changes are:

- GPU state and cross-layer hidden buffers, replacing per-layer host copies
  and waits; resident lowering also removes unused packed-output operations.
- BF16 vocabulary weight storage with a native SIMD matvec and FP32
  accumulation. Checkpoint values are preserved; F32 fixtures retain their
  original F32/MPS path.
- Parallel RMS normalization and GPU argmax. Argmax checks nonfinite logits
  and preserves the lowest token index on a tie; only token IDs and validity
  flags return to the host during normal generation.
- Constant-time tracking of pending GPU outputs, replacing repeated linear
  scans during large chunk submissions.
- Prefill graphs share immutable decoder weights through `build_shared`,
  checking checkpoint metadata, file size and modification time. Their buffer
  ownership survives the source graph closing.
- Immutable Metal shader pipelines are shared by device through a weak cache;
  queues and request state stay separate, and closing the last context releases
  the cached pipeline set.
- The default Metal prefill profile is now 16 tokens. Explicit profiles
  2/4/8/16 remain supported, independently of request batch size.

Fresh original-build controls were executed before edits, using `6a5b565`.
The final optimized requests and repeats ran separately on the same M4, using
identical checkpoints, photo/prompts and output limits. Second-request, loaded
batch-1 measurements are:

| Model / completed output | First token, old → new | Complete answer, old → new | Decode tokens/s, old → new | Peak process RSS GiB, old → new |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B image, 28 tokens | 1.931 → 1.713 s | 3.999 → 3.054 s | 13.06 → 20.15 | 5.45 → 3.38 |
| Qwen3-0.6B Paris, 8 tokens | 0.607 → 0.501 s | 1.109 → 0.811 s | 13.95 → 22.61 | 4.16 → 2.35 |
| Qwen2.5-Coder add function, 17 tokens | 0.359 → 0.274 s | 1.151 → 0.771 s | 20.23 → 32.22 | 3.35 → 1.95 |

Decode rates increase about 54%, 62% and 59% respectively. First-token latency
falls about 11%, 18% and 24%, and complete-answer time falls about 24%, 27% and
33%. Repeat optimized rates are 20.21 / 22.64 / 32.01 tokens/s, with identical
output IDs and normal EOS completion. These prompts have different lengths
and answer sizes; this table compares each model with its own control.
Timings exclude network transfer and image preprocessing as in earlier rounds.
RSS measures the entire native process, including startup and both requests,
not isolated GPU allocation or the complete system memory footprint.

The final-library suite also runs a separate explicit-state/F32-head control
alongside two optimized runs for each model and batches 1/2. This control
already includes the new parallel RMS, shader reuse and shared prefill weights.
Against that partial control, resident state/BF16 head/GPU selection improve
batch-1 decode rates from 14.69 → 20.15, 15.80 → 22.61 and 23.13 → 32.22.
Those gains must not be added to the original-build comparison above.

| Final-library model | Batch-2 control first / complete | Batch-2 optimized first / complete | Optimized decode tokens/s per row | Optimized process RSS |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B image | 2.892 / 5.071 s | 2.308 / 3.913 s | 16.82 | 3.43 GiB |
| Qwen3-0.6B text | 0.700 / 1.211 s | 0.592 / 0.946 s | 19.74 | 2.35 GiB |
| Qwen2.5-Coder | 0.364 / 1.095 s | 0.309 / 0.854 s | 29.40 | 1.95 GiB |

Batch-2 inputs repeat the same prompt/photo on both rows. Aggregate decode
throughput is twice the per-row value in these runs. All 36 benchmark requests
match their control's token IDs and finish flags exactly, including both rows;
all three optimized batch-1 outputs also match the original build.

Initialization remains distinct from warm latency. Optimized initialization
is 2.626 / 1.340 / 1.076 s for image/text/coder. The first request including
initialization takes 4.618 / 2.173 / 1.649 s to first token, and
5.975 / 2.485 / 2.148 s through the complete answer. Lazy prefill construction
is included in those first-request timings. Normal requests reuse loaded
weights and compiled profiles, but reset attention state between requests.

Longer completed outputs also pass exact token parity against final-library
controls: Qwen3 generates 55 tokens of Python readability tips, taking
0.650 s to first token / 3.650 s complete at 18.00 decode tokens/s, versus
0.746 / 4.830 s at 13.22 tokens/s. Coder generates a 37-token factorial function
in 0.285 / 1.461 s at 30.61 tokens/s, versus 0.335 / 1.938 s at 22.47 tokens/s.
Both requests finish at EOS rather than the 128-token cap. The add and factorial
outputs contain only the expected function/arithmetic/range AST nodes and pass
four functional cases each; these are coding smoke tests, not a quality suite.

One native process additionally keeps **all three models loaded together** and
interleaves two reset requests per model without closing or reloading them.
Every output ID matches the isolated control. Measured combined peak process
RSS is **5.213 GiB**. Its second-request image/text/coder first-token latencies
are 2.225 / 0.763 / 0.338 s, completion times 3.602 / 1.087 / 0.842 s and decode
rates 19.61 / 21.64 / 31.75 tokens/s. These are serialized model-switching
measurements; isolated-model latencies are lower. Concurrent GPU requests and
production scheduling remain unmeasured.

Independent pretrained references retain the earlier criteria: Qwen3.5 uses
its original component bound; dense Qwen uses the documented vector bounds.
They pass real layers, batches 1/2, paused/resumed rows, partial tails and invalid
controls. Resident results additionally equal an identical explicit-state GPU
execution exactly, including every hidden value and final state. Serving tests
cover profiles 2/4/8/16/2, private sessions sharing graphs, reset isolation,
closed graphs, failed/nonfinite requests requiring reset and default adapters.
Full pretrained vocabulary logits pass the vector-norm comparison against F32
MPS, with matching greedy predictions for batches 1/2. The BF16 child graph
still runs after its F32 weight-source graph closes. Native primitive CTest
passes BF16 matvec, parallel RMS, tie-breaking/nonfinite argmax and the existing
MPS/storage cases. No tolerance was loosened for these optimization changes.

The default constructor enables resident state and BF16 heads. For the
explicit-state/F32-head comparison use `resident=False,bf16_head=False`.
`enable_prefill()` now selects chunk 16; use an explicit argument to select a
smaller profile. Resident sessions expose `session.snapshot(layer)` for
intentional CPU state inspection; the old layer state fields become `None`.
`reset()` clears private state, and `close()` releases the session and graphs.
Individual per-layer/head timings are unavailable for a combined submission;
new logs report `decode_session_ms` and leave isolated head timing null.

Reproduce with `three_model_suite.py runtime build source phase output_folder`,
where phase is `checks`, `benchmarks`, `extended` or `resident`.
`background_test.py` accepts `three-models:phase`, `image-optimized:batch:chunk`
and `dense-optimized:model_name:batch:chunk`. Native benchmark runners accept
optimization mode 0 (explicit/F32), 1 (resident/F32) or 2 (resident/BF16), with
mode 2 now the default. `summarize_three_models.py evidence_directory` parses
raw results and asserts exact output parity for the preserved comparisons.

Evidence is saved in `results/20261012/all-three-models/`, including commands,
outputs/exits, full token IDs, parsed timings, RSS and generated-code checks.
Both eight-job check suites, 18 benchmark jobs, four extended jobs and the
combined-residency job finish successfully. Final compatibility checks also
pass image-to-mixed-decoder/head integration for batches 1/2 (two requests,
14 steps each) and Qwen3.5 text-only batch 2 with EOS on both rows. The
pipeline fixture now loads binary F32 reference tensors, avoiding slow large
JSON arrays in the interpreter; its numerical criteria are unchanged. The
`checks` phase includes this pipeline test for future runs. The text-only
smoke test overlapped the older JSON fixture parsing, so its timings are not
used as performance evidence. There are 55 saved status records
including parent suites and one earlier image-profile rejection; that rejected
trial is retained, fixed and successfully rerun. A draft shader also used the
reserved name `thread`; native CTest caught the compilation error before the
successful final builds. The final artifact has no active investigation jobs.

## Limits and next performance work

The default decoder keeps recurrent/KV state and intermediate hidden values
on the GPU, with one synchronous submission per output token or prompt chunk.
Warm isolated image first-token time is about 1.71 s, including approximately
0.35 s vision encoding; prefill still dominates this delay. Most projection
weights remain expanded FP32. The vocabulary head retains BF16 storage and
prefill graphs share decoder weights, substantially reducing process RSS.
Further work includes BF16/quantized kernels for the other projections,
parallel causal prefill attention, KV/state allocation reduction and longer
context profiling. Other Apple GPUs remain unmeasured.
Batch 2 improves aggregate throughput while increasing per-request latency;
only batches 1/2 have been executed. No batch-4, multi-client concurrency,
production service, streaming camera or other Apple device claim is made.
This still-image encoder does not implement a general video temporal profile.

## Reproduction and evidence

Build with `GARNET_METAL=ON` and `GARNET_METAL_XMODEL=ON`, setting
`XLANG3_SDK_ROOT` and `GARNET_XLANG3_RUNTIME_LIBRARY` to the existing SDK/runtime;
see `cmake/GarnetMetal.cmake`. Run the following with Mac Python, supplying
`runtime build source model` as the four arguments:

- `test2026/metal/test_pretrained_graph.py`
- `test2026/metal/test_pretrained_vision_graph.py`
- `test2026/metal/benchmark_text.py ... 1 32 'Explain three benefits of running language models locally.'`
- `test2026/metal/benchmark_image.py ... images/RealWorld-04.batch1.json 32`

`background_test.py` launches durable jobs with modes `pretrained`,
`pretrained-vision`, `pipeline`, `text-long:1`, `text-long:2`, `image:1`,
`image:2`, `image-long:1` or `optimization-suite`. The suite runs image and
text benchmarks for batches 1/2 sequentially with separate process RSS records. Its explicit arguments are
`launch log_directory runtime build source mode`.
Real checkpoint and image inputs are not committed.

Saved raw command/exit/output evidence is in `results/20261012/`;
The earlier `benchmarks.json`, `allocation-reuse/benchmarks.json` and
`chunk-prefill/benchmarks.json` retain tokens, output, per-step/layer/head timings and peak
RSS. Historical bring-up evidence is in `BRINGUP_HISTORY.md`; its earlier
pending-state notes are superseded by this report.
