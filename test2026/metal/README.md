# Garnet native Metal/MPS: Qwen3.5-0.8B

Actual text and image-plus-text inference work on the Apple M4 Mac mini
(16 GiB, macOS 15.6.1), using Garnet's native xlang3 package, captured xModel
graphs, Metal kernels and MPS matrix multiplication. All learned neural
operations, including the full vision tower and tied vocabulary projection,
execute on the Apple GPU. Tokenization, image preprocessing, embedding lookup,
request control, argmax and explicit state movement use the host CPU.

This is an experimental FP32 backend. It is functional and measured, with
remaining performance limits described below; it is not fully optimized.

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

The table reflects the final allocation-reuse build. Each process loads the
model once and runs two independent requests, resetting
state between them. The table uses the second request. Output limits apply
per batch row. Timings exclude CPU image preprocessing, which was performed
beforehand, and network transfer. First-token time includes vision and decoder
prefill. Decode throughput excludes the initial token produced by prefill.

| Workload | Batch | Output | Loaded first token, s | Loaded result, s | Decode tokens/s per row | Combined tokens/s | Process peak RSS, GiB |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Text: local-model benefits | 1 | 32-token limit | 1.245 | 3.395 | 14.42 | 14.42 | 4.40 |
| Text: local-model benefits | 2 | 32-token limit each | 1.477 | 4.031 | 12.14 | 24.28 | 4.41 |
| Image, allocation reuse | 2 | 16-token limit each | 6.372 | 7.750 | 10.88 | 21.77 | 4.84 |
| Image, complete caption | 1 | 28 tokens including stop | 5.037 | 7.101 | 13.08 | 13.08 | 4.81 |

Peak RSS is measured for the whole native process in macOS bytes, converted
to GiB, and is not isolated GPU allocation. It varies across runs and MPS
allocation profiles. The Mac had other workloads: a final system snapshot
showed 3,741 MiB swap in use, so these are development-machine observations,
not isolated laboratory results or percentile latency guarantees.

Text initialization was about 2.85 s for the short Paris test; the first
request including startup reached its first token in 4.77 s. Those short-test values are from the earlier build. In the final build,
longer-text initialization was 2.04 s with first token including startup at
3.36 s; image initialization was 2.81 s with first token including startup
at 7.96 s.
The longer text test reached its token cap rather than EOS; the caption
finished normally at 28 tokens. Batch 2 repeats the same input on both rows;
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
checkpoint precision are unchanged. No GPU-resident recurrent state or chunked
prefill claim is made by this allocation optimization.

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

## Limits and next performance work

The decoder still uses sequential prefill and synchronous per-layer graph
submission, with explicit CPU state copies. Warm image prefill is about 4.68 s
for 79 prompt tokens and dominates first-token latency. Captured weights are
expanded from BF16 into FP32, increasing memory and bandwidth requirements.
The tied vocabulary head alone occupies approximately 1 GB in FP32.
Chunked prefill, GPU-resident cross-layer state, BF16/quantized kernels,
KV/state allocation reduction and memory-pressure profiling remain future work.
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
Both the earlier `benchmarks.json` and `allocation-reuse/benchmarks.json`
retain tokens, output, per-step/layer/head timings and peak
RSS. Historical bring-up evidence is in `BRINGUP_HISTORY.md`; its earlier
pending-state notes are superseded by this report.
