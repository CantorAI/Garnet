# Garnet Continuous Batching and VLM Scheduler Design

## Status

This document defines the next serving-runtime stage for Garnet after the
verified batch-one Qwen3-VL compiled runtime. It extends the normative compiled
xmodel architecture with concrete scheduling, batching, paged-KV, multimodal,
and GPU-sampling decisions.

The canonical model remains the `.x` graph. The scheduler may select profiles,
allocate state, and launch compiled engines, but it must not recreate Qwen model
layers in Python or hardcode a second model architecture in C++.

## Purpose

Garnet must increase aggregate VLM/LLM serving throughput without sacrificing
the current low batch-one latency. The production design is:

```text
one model instance per GPU
+ one shared weight copy
+ independent vision, prefill, and decode queues
+ global paged-KV memory
+ continuous decode batching
+ GPU-side sampling
+ profile-aware TensorRT engine and CUDA Graph selection
```

Multiple HTTP workers may submit requests, but they must not each load another
copy of the model on the same GPU.

## Current Verified Baseline

The current Qwen3-VL-2B regular-resolution profile is:

```text
GPU:                         RTX 4080 16 GB
serving batch:               1
prefill token profile:       1,024
KV page size:                16 tokens
KV pages:                    80
KV token capacity:           1,280
vision patches:              3,772
merged visual tokens:        943
total benchmark prompt:      1,013 tokens
output ceiling:              100 tokens
decode throughput:           approximately 128 tokens/s
TTFT:                        approximately 129 ms
four-image total latency:    495-599 ms
```

Measured GPU-0 memory during the full benchmark:

```text
Windows idle baseline:       approximately 2.3 GiB
model/engine loaded total:   approximately 11.7 GiB
full-test peak total:        approximately 14.6 GiB
Garnet peak above baseline:  approximately 12.0 GiB
free at observed peak:       approximately 1.7 GiB
```

This memory footprint is too tight for multiple model replicas and is a gating
constraint for large serving batches.

## Required Invariants

1. All model inputs, activations, weights, KV pages, logits, and sampler state
   remain GPU-resident as `X::Tensor` values or model-owned CUDA allocations.
2. Only compact scheduling state, selected token IDs, status, and final output
   cross the CPU boundary during normal serving.
3. Requests in one physical batch remain semantically isolated.
4. Batch-one uses the same scheduler and state model as larger batches.
5. Engine cache validity includes `.x` dependencies, tensor shapes, dtypes,
   weights, compile options, TensorRT version, runtime schema, and GPU compute
   capability.
6. Generated outputs must match independent batch-one execution within the
   defined numerical and token-level tolerance.
7. Python is a test and debugging client only. It is not the serving scheduler
   or generation loop.

## Terminology

### Request batch

A set of user requests selected by the scheduler for one stage.

### Vision microbatch

A dense batch of compatible images or ROIs with the same compiled visual
profile.

### Packed prefill

Tokens from several independent requests concatenated into one physical tensor,
with offsets that preserve request boundaries.

### Continuous decode batch

The active set of requests contributing one token each to a decode iteration.
Requests may join or leave between iterations.

### Engine bucket

A compiled TensorRT profile selected for an actual workload, such as batch 3
executing in a batch-4 CUDA Graph with one inactive row.

## High-Level Architecture

```text
API and application workers
            |
            v
      request ingress
            |
            v
  +-------------------------+
  | per-GPU Garnet scheduler|
  +-------------------------+
     |          |         |
     v          v         v
  vision      prefill   decode-ready
   queue       queue       queue
     |          |         |
     v          v         v
  vision      packed     continuous
 microbatch   prefill      decode
     |          |         |
     +----------+---------+
                |
                v
       global GPU KV pool
                |
                v
          GPU sampler
                |
                v
       per-request outputs
```

Vision, prefill, and decode batches are intentionally independent. A request
may be in a vision microbatch with requests that never share its decode batch.

## Request Lifecycle

Each request moves through explicit states:

```text
RECEIVED
  -> PREPROCESSING
  -> VISION_QUEUED
  -> VISION_RUNNING
  -> PREFILL_QUEUED
  -> PREFILL_RUNNING
  -> DECODE_READY
  -> DECODING
  -> FINISHED | CANCELLED | FAILED
```

The model-owned request state contains at least:

```text
request GUID
arrival time and priority
prompt token IDs
MRoPE position state
multimodal item descriptors
vision embedding references
logical-to-physical KV block table
context length
next slot position
sampling parameters
GPU RNG state
grammar/allowed-token state
generated token count
finished/cancelled status
latency timestamps
```

Cancellation must release pending vision work, KV pages, prefix references,
sampler state, and scheduler slots.

## Vision Scheduling

### Initial decision: dense profile microbatching

The first implementation must not attempt a universal ragged visual encoder.
It groups compatible images by a profile key such as:

```text
model + dtype + resized width + resized height + patch count + temporal grid
```

Example profiles:

```text
vision-v256-b1/b2/b4
vision-v943-b1/b2/b4
vision-v2040-b1/b2
roi-224-b1/b2/b4/b8
roi-448-b1/b2/b4
```

The scheduler may wait for a short microbatch window, but it must dispatch a
batch-one request when its latency deadline would otherwise be missed.

### ROI path

One JPEG is decoded once. GPU ROI descriptors then produce compatible crops:

```text
JPEG -> nvJPEG GPU image -> CUDA ROI crop/resize -> vision microbatch
```

ROIs from different requests may share a physical microbatch. Their embeddings
remain owned by their original requests. Each descriptor records:

```text
request_id
roi_id
source image dimensions
source bbox
target visual profile
coordinate transform
```

Model-returned ROI-local coordinates are transformed back into full-image
coordinates during output processing.

### Deferred visual packing

Qwen-specific flat patch execution may be added later using patch offsets,
`image_grid_thw`, and segmented vision attention. It is not required for the
first production scheduler. The expected first win comes from dense compatible
microbatches and embedding caching.

### Vision cache

Vision embeddings are cached per multimodal item, not per complete request.
The key includes:

```text
model fingerprint
processor/profile fingerprint
content hash or trusted media UUID
ROI coordinates
output dtype
```

Cache hits skip image decoding, preprocessing, and the vision tower. GPU cache
entries are reference-counted and evicted under a separate memory budget.

## Text Prefill Batching

### First milestone: padded prefill buckets

The simplest correct implementation uses exact batch and token profiles:

```text
prefill-b1-t1024
prefill-b2-t1024
prefill-b4-t1024
prefill-b1-t2048
prefill-b2-t2048
```

Padding is permitted initially but must be reported as wasted scheduled tokens.

### Production milestone: packed prefill

Each request first constructs its own complete multimodal sequence:

```text
system text + visual placeholders/embeddings + user text
```

Independent sequences are concatenated physically:

```text
packed_tokens = [A tokens | B tokens | C tokens]
offsets       = [0, len(A), len(A)+len(B), total]
```

Required tensors include:

```text
packed_input_ids:     [total_tokens]
packed_embeddings:    [total_tokens, hidden]
sequence_offsets:     [batch + 1]
position_ids:         [3, total_tokens]
slot_mapping:         [total_tokens]
request_indices:      [batch]
page_tables:          [batch, max_logical_pages]
```

Attention is independently causal within each sequence. No request may attend
to another request's text or visual tokens.

### Chunked prefill

Long prompts are divided into scheduler-controlled chunks so they cannot block
decode latency:

```text
tick 1: A prefill[0:512]    + B/C decode
tick 2: A prefill[512:1024] + B/C decode
```

The scheduler operates on a total scheduled-token budget, not only a request
count.

## Continuous Decode Batching

### Pre-batching kernel prerequisite

Before continuous batching changes request scheduling, Garnet must replace the
current split score, global-workspace softmax, and value-accumulation decode
path with a fused paged FlashAttention-style kernel. The kernel consumes paged
K/V directly, processes the context in tiles, maintains FP32 online-softmax
state, and accumulates the weighted value output without materializing the full
attention-score row in global memory.

The interface is batch-shaped from its first implementation even while the
verified execution profile remains batch one:

```text
queries:          [batch, q_heads, head_dim]
key/value pages:  [layers, physical_pages, page_size, kv_heads, head_dim]
page tables:      [batch, max_logical_pages]
context lengths:  [batch]
slot positions:   [batch]
active mask:      [batch]
outputs:          [batch, q_heads, head_dim]
```

The first correctness gate compares batch-one output against both the existing
split implementation and the Hugging Face reference at context lengths 64,
256, 1,024, and the configured maximum. FP32 running maximum, normalization
sum, and output accumulation are required; BF16 remains the storage and engine
interface dtype. Continuous batching may begin only after this gate passes.

Every active request contributes one token row:

```text
token_ids:        [B, 1]
position_ids:     [3, B, 1]
context_lengths:  [B]
slot_positions:   [B]
page_tables:      [B, max_logical_pages]
active_mask:      [B]
```

The decode engine returns:

```text
logits:           [B, vocabulary]
```

The active set changes between iterations:

```text
iteration 1: A B C D
iteration 2: A C D E
iteration 3: A C E F
```

The batch-aware `.x` decode graph remains the model source. CUDA paged-attention
kernels consume batched page tables, context lengths, slot positions, and
active masks.

## Global Paged-KV Pool

The per-GPU physical pool is shared by all requests:

```text
key_pages:
[layers, total_physical_pages, page_size, kv_heads, head_dim]

value_pages:
[layers, total_physical_pages, page_size, kv_heads, head_dim]
```

Each request owns a logical block table:

```text
request A logical page 0 -> physical page 17
request A logical page 1 -> physical page 42
request B logical page 0 -> physical page 3
```

The page manager provides:

```text
free-page list
allocation watermark
per-request page ownership
prefix-page reference counts
copy-on-write for shared prefixes
cancellation cleanup
memory admission checks
allocation and fragmentation metrics
```

Current BF16 Qwen3-VL-2B KV cost is:

```text
112 KiB per context token per request
140 MiB for the current 1,280-token capacity
```

KV memory grows linearly with active context capacity. Weight and TensorRT
context memory currently dominate batch-one memory.

## Prefix KV Cache

Fixed system prompts and other matching prefixes use immutable shared pages.
Prefix records are identified by a GUID and content fingerprint. Each GPU has a
local residency record because GPU pointers are device-specific.

```text
prefix GUID
model/tokenizer fingerprint
token IDs and length
physical page references
last access
reference count
optional local persistent metadata
```

Requests reference shared pages until divergence, then allocate private pages.
The prefix manager must never compare only user-provided IDs; it validates the
model, tokenizer, token content, and positional contract.

## GPU Sampler

The sampler converts GPU logits directly into one selected token per active
request:

```text
logits [B, vocabulary]
  -> penalties and masks
  -> temperature
  -> top-k/top-p/min-p or greedy argmax
  -> token_ids [B]
  -> token_scores [B]
  -> finished_mask [B]
```

The current Garnet runtime implements persistent-buffer BF16/FP32 greedy
argmax for batch one. Production batching adds:

```text
batched greedy argmax
per-request temperature/top-k/top-p/min-p
per-request GPU RNG state
repetition/presence/frequency penalties
stop-token sets
grammar and allowed-token masks
finished masks
optional log probabilities
```

Sampler output remains on GPU and feeds the next decode iteration directly.
The CPU observes compact token/status data required for streaming and scheduler
decisions. A later optimization may fuse LM-head projection with sampling to
avoid materializing full logits.

## Scheduler Policy

Each scheduler tick performs:

```text
1. Process cancellation and completion events.
2. Release or reference-count KV and vision-cache resources.
3. Admit requests that fit GPU memory and profile limits.
4. Select the active decode set.
5. Select a decode engine/CUDA Graph bucket.
6. Execute one decode iteration and GPU sampling.
7. Schedule bounded prefill chunks within the remaining token budget.
8. Schedule compatible vision microbatches within the remaining compute budget.
9. Publish selected tokens and update metrics.
10. Repeat.
```

Decode is latency-sensitive and receives the highest recurring priority.
Prefill and vision work are bounded so a large image or prompt cannot create an
unbounded inter-token delay.

### Admission inputs

```text
available physical KV pages
estimated vision activation memory
engine/context workspace
request prompt and output reservation
active batch/profile limits
priority and deadline
queue wait time
```

### Fairness

The scheduler combines request priority with aging. A low-priority request must
eventually run unless it exceeds an explicit resource or policy limit.

### Preemption

The first implementation uses this order:

```text
1. Stop admitting new work near the memory watermark.
2. Delay unscheduled vision/prefill work.
3. Preempt the lowest-priority non-decoding request.
4. Release and later recompute KV when necessary.
```

CPU KV swapping is deferred. Recompute is simpler and avoids making CPU memory
part of the normal tensor plane.

## Engine Profiles and Automatic Compilation

Initial exact profiles:

```text
decode-b1/b2/b4/b8-t1280
prefill-b1/b2/b4-t1024
prefill-b1/b2-t2048
vision-b1/b2/b4-v943
roi-b1/b2/b4/b8-r224
```

Actual batch sizes select the next compatible CUDA Graph bucket:

```text
actual B3 -> B4 graph with one inactive row
actual B5 -> B8 graph with three inactive rows
```

Each compiled profile has its own cache directory or content-addressed key.
Changing batch, token capacity, patch profile, dtype, `.x`, weights, workspace,
or optimization level produces a different fingerprint and automatic rebuild.

Production startup should prewarm configured profiles. An online request must
not unexpectedly wait minutes for TensorRT compilation unless an explicit
on-demand compilation policy allows it.

Dynamic TensorRT min/opt/max profiles may replace some exact profiles after
correctness and performance are established.

## CUDA Streams and Context Ownership

One GPU worker owns model execution. Suggested streams:

```text
high priority:   decode
normal priority: text prefill
normal priority: vision
utility:         nvJPEG/preprocessing and transfers
```

Overlap is permitted only when measured to improve throughput without violating
decode latency. Concurrent threads must not mutate one TensorRT execution
context's bindings or CUDA Graph state. Contexts are owned by a scheduler lane,
not borrowed without synchronization.

## Multiple GPUs

For a model that fits on one GPU, prefer one independent serving worker per GPU:

```text
GPU 0: model replica + scheduler + KV pool
GPU 1: model replica + scheduler + KV pool
```

The load balancer routes whole requests to a GPU based on queue depth, memory,
prefix/vision-cache locality, and expected work.

For models that do not fit on one GPU, the compiled graph is partitioned for
tensor/pipeline parallel execution. That is separate from request batching and
does not change the per-request ownership contract.

## Memory Plan

Before enabling large batches on a 16 GB GPU, Garnet must:

1. Verify memory reaches a stable plateau over long repeated-request tests.
2. Attribute memory to weights, engine contexts, workspaces, CUDA Graphs,
   vision activations, KV pages, and caches.
3. Reduce duplicated prefill/decode weight residency.
4. Add admission control with a safety reserve.
5. Evaluate BF16 activation plus weight-only INT8 decoder/LM-head storage.

Initial target:

```text
batch-one Garnet peak:       7-9 GiB
serving/cache reserve:       3-5 GiB
driver/system safety margin: at least 1 GiB
```

The scheduler must use measured allocation sizes rather than assuming KV cache
is the dominant consumer.

## Public and Internal API Shape

Application-facing API remains request-oriented:

```python
request_id = server.submit(
    image=image,
    prompt=prompt,
    sampling={"max_tokens": 100, "temperature": 0.0},
    prefix_id=optional_prefix_guid,
)
```

The model still supports synchronous `forward()` for tests and embedded use.
Serving internally uses model-owned objects similar to:

```text
GpuRequestState
VisionBatch
PrefillBatch
DecodeBatch
PagedKVPool
PrefixManager
GpuSampler
EngineProfileManager
CudaGraphBucketManager
GpuScheduler
```

Debug probes remain generic:

```text
model.debug_probe(key, request_id, options)
```

They may observe compact metadata or explicitly copy selected tensors to CPU,
but are disabled from the performance path by default.

## Metrics

Required per-request metrics:

```text
queue wait
image decode/preprocess latency
vision latency
prefill latency
TTFT
inter-token latency
decode tokens/s
total latency
generated tokens
finish reason
```

Required per-GPU metrics:

```text
active and queued requests by stage
actual and graph-bucket batch sizes
scheduled and padded tokens
vision microbatch occupancy
KV pages free/used/shared
prefix and vision-cache hit rates
GPU memory by owned category
preemption and rejection counts
engine/cache hits and compilation time
CUDA Graph capture/replay counts
```

## Correctness Tests

1. Batch-one output remains equal to the current reference.
2. Batched B2/B4 outputs match independent batch-one runs.
3. Requests cannot attend to or modify another request's state.
4. Variable prompt and image lengths produce correct MRoPE positions.
5. Packed prefill matches padded prefill logits.
6. Paged-KV allocation, rollover, reuse, and cancellation are correct.
7. Shared prefix pages use reference counting and copy-on-write.
8. Finished rows do not write new KV or affect active rows.
9. Heterogeneous sampling parameters remain request-local.
10. CUDA Graph bucket padding does not change active outputs.
11. Engine profile cache invalidates on every contract-changing input.
12. Repeated requests reach a stable memory plateau.

## Performance Gates

Initial RTX 4080 gates, to be revised after implementation:

```text
batch-one decode:               >= 120 tokens/s
batch-one regular-image TTFT:   <= 150 ms
batch-one latency regression:   <= 5% from verified baseline
batch-4 aggregate decode:       >= 250 tokens/s
batch-8 aggregate decode:       >= 350 tokens/s
cancelled-request KV release:   bounded by one scheduler tick
unbounded GPU memory growth:    zero over long soak test
```

Aggregate throughput targets do not permit silent output shortening or quality
changes.

## Implementation Stages

### Fused attention checkpoint (2026-07-12)

Stage 1 now has an initial verified implementation in
`src/cuda/text_kv_attention_kernel.cu`. The TensorRT
`GarnetPagedKVDecodeBF16` plugin selects the fused path by default for Qwen's
128-element head dimension. `GARNET_PAGED_KV_FLASH=0` selects the previous
split score/softmax/value implementation for matched diagnostics.

Direct batch-one output parity passed against both the original attention
kernel and the optimized split implementation at context lengths 64, 256,
1,024, and 1,280. The maximum observed FP32 value difference after BF16 output
conversion was `0.000031`.

The unchanged regular-resolution four-image benchmark on RTX 4080, with 943
visual tokens, 1,013 prompt tokens, and exactly 100 generated tokens per image,
measured:

```text
                                  split path       fused online path
average total latency             890.71 ms        874.22 ms
average decode latency            761.83 ms        745.27 ms
average decode throughput         129.95 tok/s     132.84 tok/s
average TTFT                      128.51 ms        128.54 ms
```

The fused path improved decode throughput by approximately 2.2% and reduced
average end-to-end latency by approximately 16.5 ms (1.9%). TensorRT decoder
GEMMs remain the dominant decode cost, so this kernel optimization is useful
but is not expected by itself to produce a large end-to-end multiple.

The dedicated regression is:

```text
test2026/tests/phase_24_compiled_xmodel_runtime/test_paged_flash_attention.py
```

### Stage 1: fused paged decode attention

- Implement a batch-shaped fused paged FlashAttention-style decode kernel.
- Traverse physical K/V pages in tiles and use FP32 online softmax.
- Avoid the full score workspace on the fused path.
- Keep the existing split implementation as a parity oracle until validation
  and benchmark gates pass.
- Verify batch-one parity at short, medium, and maximum context lengths.
- Benchmark kernel time, workspace, register use, shared memory, and occupancy
  on RTX 4080.

### Stage 2: memory and state foundation

- Add categorized CUDA memory metrics.
- Verify long-run memory plateau.
- Implement per-GPU page pool and per-request block tables.
- Add request lifecycle and cancellation cleanup.

### Stage 3: batched decode

- Make `qwen_text_decode.x` batch-aware.
- Add batched paged-KV metadata and kernels.
- Add B1/B2/B4/B8 engine profiles.
- Add batched greedy GPU sampling.
- Verify output parity against independent batch one.

### Stage 4: continuous scheduler

- Implement join/leave decode batches.
- Add graph bucket selection and inactive masks.
- Add memory admission, priority, aging, and preemption.
- Add streaming output routing.

### Stage 5: prefill batching

- Add padded prefill buckets.
- Add chunked prefill mixed with decode.
- Add packed tokens, offsets, slot mappings, and segmented attention.
- Verify packed versus padded parity.

### Stage 6: multimodal scheduling

- Add compatible vision microbatches.
- Add ROI descriptors and coordinate transforms.
- Add per-item vision embedding cache.
- Add stage overlap under decode-latency budgets.

### Stage 7: production sampler and caches

- Add top-k/top-p/min-p and GPU RNG state.
- Add penalties, grammar masks, and batched stop handling.
- Add prefix-page sharing, GUID management, and copy-on-write.

### Stage 8: memory and throughput optimization

- Eliminate or reduce duplicate prefill/decode weights.
- Add graph-declared weight-only INT8 where parity permits.
- Evaluate fused LM-head sampling.
- Tune aggregate throughput and latency SLOs.

## Relationship to vLLM and SGLang

This design adopts proven serving concepts from vLLM and SGLang:

```text
continuous batching
paged KV
chunked prefill
prefix caching
encoder scheduling and caching
CUDA Graph buckets
memory-based admission
```

Garnet's distinction is that model execution comes from runtime-captured `.x`
graphs compiled into fingerprinted TensorRT engines. The scheduler selects and
feeds those engines; it does not replace the model graph.

References:

- https://docs.vllm.ai/en/latest/api/vllm/v1/core/encoder_cache_manager/
- https://docs.vllm.ai/en/latest/api/vllm/multimodal/inputs/
- https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/epd_disaggregation.md
- https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md
