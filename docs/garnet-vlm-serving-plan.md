# Garnet VLM Serving Goal And Plan

## Goal

Garnet is a cross-platform, model-agnostic VLM/LLM serving runtime built around xlang tensor expressions, CUDA code generation, and a native C++/CUDA execution engine.

The primary goal is not only chat-style image QA. Garnet should target low-latency "world-sense" VLM serving, where visual inputs may refresh continuously:

```text
camera frame / desktop screenshot / document page / game frame / robot scene
  -> quick perception, reasoning, or event verification
  -> short textual or structured answer
```

The first serious model target is Qwen3-VL, but Garnet should not become a Qwen-only runner. Qwen3-VL is the benchmark workload because it exercises the hard parts of modern VLM serving:

- expensive image/video prefill
- variable visual token counts
- multimodal token merge and multimodal position encoding
- decoder KV cache
- fresh-frame and repeated-context workloads
- contention between heavy prefill jobs and low-latency decode

The long-term goal is fast VLM/LLM serving with a runtime that can be compared against vLLM and SGLang, while remaining useful as a learning and experimentation platform for serving systems, GPU kernels, and runtime scheduling.

Garnet must work across both Windows and Linux:

```text
Windows:
  local development, desktop/agent integration, native DLL/plugin workflow

Linux:
  production serving, containers, multi-GPU nodes, NCCL-first deployment, cloud benchmarks
```

This cross-platform requirement affects the process model, build system, CUDA IPC assumptions, service hosting, file paths, profiling, and observability.

## Positioning

vLLM and SGLang already support VLMs. Garnet is not motivated by the idea that those systems cannot serve Qwen-VL models.

The Garnet thesis is more specific:

> VLM serving has bottlenecks that are different from text-only LLM serving. Garnet should make fresh-frame latency, visual-token budgeting, vision prefill scheduling, multimodal cache when reuse exists, and decode scheduling first-class runtime concepts.

Expected advantage areas:

- lower latency for fresh-frame VLM requests through visual-token budgeting and staged scheduling
- better throughput under mixed fresh-image, text-only, and decode-heavy traffic
- better reuse when visual context repeats, without making cache reuse the only product story
- better GPU utilization by scheduling vision prefill and decoder decode separately
- stronger control over memory layout, KV blocks, CUDA streams, generated kernels, and CUDA graphs
- model/compiler/runtime integration through xlang tensor graph lowering

## Latency Target

The product goal has two tracks:

```text
fresh-frame track:
  lowest practical TTFT for a newly refreshed HD frame on RTX 4080

cached-context track:
  sub-100 ms TTFT when visual processing or multimodal prefix can be reused
```

The fresh-frame track is the more important world-sense target. The cached-context track is useful, but should not dominate the design.

Likely feasible for cached-context mode:

```text
same image/document/screenshot/video segment
cached image processing
cached vision embeddings or cached multimodal decoder prefix
small Qwen3-VL-class model
short question
first decode token under 100 ms
```

Not realistic as an early fresh-frame target:

```text
fresh raw 1920x1080 image
full image preprocessing
full vision encoder
full decoder prefill
no cache
complete answer under 100 ms
```

For fresh HD inputs, Garnet should focus on reducing or prioritizing visual work:

```text
fast mode:
  lower visual token budget, resize/crop/tile selectively, short answer

accurate mode:
  larger visual token budget, slower TTFT

event mode:
  verify a specific event with constrained prompt and output schema

stream mode:
  keep the latest frame fresh; skip stale frames if needed
```

Open question: the first hard benchmark should probably be fresh-frame latency, not repeated-image QA. We need decide the exact workload: webcam frame, desktop screenshot, game frame, document image, or event-verification frame.

## Goal Metrics

Garnet goals should be measured with explicit latency, throughput, quality, and resource metrics. These numbers are initial targets for RTX 4080-class local development and should be revised after the first benchmark harness is running.

### Primary Fresh-Frame Metrics

Fresh-frame workloads measure a newly arrived visual input with no vision-cache reuse.

```text
fresh_frame_ttft_ms:
  time from frame accepted by Garnet to first output token

fresh_frame_e2e_ms:
  time from frame accepted by Garnet to completed short answer

frame_age_at_answer_ms:
  age of the visual frame when the answer is emitted

fresh_frame_tokens_per_sec:
  decode speed after fresh-frame prefill

visual_tokens:
  number of visual tokens produced for the frame

prefill_tokens_per_sec:
  text + visual prefill throughput
```

Initial target bands:

```text
fresh HD frame, fast mode:
  TTFT:        < 500 ms first target, < 250 ms stretch
  short answer: < 800 ms first target, < 400 ms stretch
  frame age:   < 1000 ms first target, < 500 ms stretch

fresh reduced-budget frame:
  TTFT:        < 250 ms first target, < 150 ms stretch
  short answer: < 500 ms first target, < 300 ms stretch

fresh event verification:
  TTFT:        < 200 ms first target, < 100 ms stretch
  yes/no JSON: < 350 ms first target, < 200 ms stretch
```

The sub-100 ms target belongs to constrained or cached modes until measurement proves otherwise.

### Cached Or Reuse Metrics

Cached-context workloads measure repeated or partially reused visual context.

```text
cache_hit_rate:
  percentage of requests reusing processed patches, vision embeddings, or KV prefix

cached_ttft_ms:
  time from request accepted to first output token when visual reuse is available

cache_lookup_overhead_ms:
  time spent hashing, finding, and validating cached visual/prefix state

cache_memory_bytes:
  GPU and CPU memory used by reusable visual or KV state
```

Initial target bands:

```text
same visual context + new short prompt:
  TTFT: < 100 ms target
  cache lookup overhead: < 5 ms target

near-duplicate frame reuse:
  useful only if accuracy stays acceptable for the chosen workload
```

### Scheduler Metrics

These measure whether Garnet is actually a serving runtime, not just a single-request runner.

```text
request_queue_wait_ms:
  time from request arrival to scheduler admission

decode_step_latency_ms:
  latency of one decode scheduler iteration

prefill_queue_wait_ms:
  time spent waiting for vision/text prefill

active_sequences:
  number of concurrent sequences in decode

batched_tokens_per_step:
  number of text + visual tokens processed per scheduler step

dropped_frames:
  number of skipped frames in stream mode

latest_frame_ratio:
  percentage of answers produced for the newest available frame instead of stale frames
```

Initial target bands:

```text
stream mode:
  latest_frame_ratio: > 90%
  stale frame answers: avoid by dropping frames when behind

mixed text + VLM traffic:
  text-only decode should not be blocked behind long vision prefill
```

### GPU And Memory Metrics

```text
gpu_memory_used_bytes:
  total GPU memory consumed

kv_cache_blocks_used:
  active KV blocks allocated

kv_cache_fragmentation:
  wasted KV block capacity

gpu_utilization:
  SM utilization during prefill and decode

kernel_launch_overhead_ms:
  CPU/GPU overhead from many small kernels

h2d_d2h_copy_ms:
  host-device and device-host transfer time
```

Initial target bands:

```text
single RTX 4080:
  avoid OOM for Qwen3-VL small dense model
  report memory split: weights / activations / KV cache / visual cache
  reduce kernel launch overhead with fusion and CUDA graphs after correctness
```

### Quality Metrics

Latency is not useful if the fast path destroys the answer. Each benchmark should include a lightweight quality gate.

```text
event_verification_accuracy:
  accuracy for yes/no or JSON event prompts

ocr_field_accuracy:
  field extraction accuracy for document/screenshot tasks

grounding_error:
  coordinate or bounding-box error when grounding is used

answer_consistency:
  agreement with vLLM/Transformers baseline for the same prompt and image
```

Fast mode must report both speed and quality loss. A result like "2x faster but unusable OCR" is not a win.

## Serving Pipeline

VLM requests should be decomposed into explicit stages:

```text
HTTP/OpenAI request
  -> message parse
  -> text tokenization
  -> image/video decode
  -> resize/crop/frame sampling
  -> patch embedding / vision encoder
  -> visual token projection
  -> multimodal merge
  -> decoder prefill
  -> decode loop with KV cache
  -> sampling
  -> streaming detokenization
```

For world-sense serving, Garnet also needs a streaming input mode:

```text
frame source
  -> frame admission / drop policy
  -> image preprocessing
  -> VLM request construction
  -> short decode or structured output
  -> optional state update
```

The runtime should be allowed to drop or skip frames when the model is behind. Freshness can be more valuable than answering every frame.

## Runtime Architecture

```text
API / Server Layer
  OpenAI-compatible HTTP, local C++ API, streaming, request lifecycle

Scheduler Layer
  continuous batching, chunked prefill, frame admission, decode scheduling

Runtime Layer
  graph execution, memory planner, KV cache manager, CUDA graph capture

Kernel Layer
  CUDA C++ / NVRTC / CUTLASS / cuBLASLt kernels

Model Adapter Layer
  Qwen3-VL, Qwen text, Llama, DeepSeek, future model families

Platform Layer
  Windows/Linux build, process, IPC, logging, profiling abstractions
```

The model adapter describes what to run. The runtime decides how to run it fast.

## Tensor Expression And Backend Handlers

Garnet's important design idea is that the model is written once as an xlang tensor expression, while different tensor implementations can interpret or lower that expression differently.

```text
xlang model expression
  -> TensorExpression
  -> TensorGraph / TensorRunItems
  -> backend handler
       JIT CUDA handler
       TensorRT builder handler
       debug/workbench handler
       future profiling/shape-only handler
```

The original Garnet path uses `GarnetTensor` as the tensor operation handler. It registers tensor ops such as:

```text
add
minus
mul
matmul
permute
gather
convert
header / trailer
branchBegin / branchEnd
```

In JIT mode, these handlers generate CUDA C++ code strings. `TensorGraph` schedules the expression, the `GarnetTensor` handlers emit code, and `CudaJitCompiler` compiles that code through NVRTC.

The TensorRT branch introduces another implementation concept:

```text
same xlang expression
  -> same TensorGraph
  -> TensorRT-aware handler/context
  -> HandleBinaryOp / HandleUnaryOp
  -> TensorRT network layers, plugins, or custom backend ops
```

This is the right abstraction. The model source should not need to become a different language just because the backend changes.

Example xlang frontend:

```python
output = a * T.binary_op("trt_matmul") * weights["W"]
```

The frontend records a generic open operation:

```text
binary_op("trt_matmul")
```

The backend decides what it means:

```text
TensorRT backend:
  add TensorRT matrix multiply layer

JIT CUDA backend:
  emit or call CUDA matmul kernel

Debug backend:
  emit graph node only, collect shape and metadata

Workbench backend:
  visualize as an unresolved or mapped op
```

This means Workbench should model backend lowering explicitly. Each graph node should show:

```text
frontend op:
  binary_op("trt_matmul")

selected backend:
  TensorRT

lowering status:
  mapped / unsupported / plugin / custom CUDA / debug-only

runtime data:
  tensor shapes, memory, timing, snapshots
```

Current branch status:

```text
Implemented skeleton:
  ITRTContext
  TRTBuilder
  HandleBinaryOp / HandleUnaryOp
  T.binary_op / T.unary_op registration
  TensorRT source files included in build

Still incomplete:
  real backend state from T.set_backend
  TensorRT layer creation in HandleBinaryOp / HandleUnaryOp
  mapping Garnet tensors to TensorRT ITensor
  TRTEngine wrapper and execution context
  model script convention for exposing a forward function
```

For the Workbench, this gives a very useful view:

```text
Architecture Graph:
  Qwen-VL logical blocks

Tensor Graph:
  expanded tensor ops and control flow

Backend Graph:
  JIT CUDA domain vs TensorRT domain vs Garnet custom kernel domain

Debug Trace:
  actual runtime shapes, timing, snapshots, KV cache state
```

## Model Adapter Interface

Each model family should plug into Garnet through an adapter:

```cpp
class ModelAdapter {
public:
    ModelConfig LoadConfig(const std::string& path);
    WeightMap LoadWeights(const std::string& path);
    TokenizerSpec GetTokenizerSpec() const;
    ProcessorSpec GetProcessorSpec() const;
    Graph BuildPrefillGraph() const;
    Graph BuildDecodeGraph() const;
    CacheSpec GetKVCacheSpec() const;
    AttentionSpec GetAttentionSpec() const;
};
```

For Qwen3-VL, the adapter owns:

- config parsing
- weight-name mapping
- tokenizer and special token handling
- image/video processor metadata
- visual token merge
- MRoPE / multimodal position IDs
- decoder architecture description
- KV cache shape and dtype

## Scheduler Design

Garnet should not use one generic queue for all work. VLM serving benefits from separate queues:

```text
frame_ingest_queue
vision_preprocess_queue
vision_encoder_queue
decoder_prefill_queue
decode_queue
finished_queue
```

Each engine iteration should:

```text
1. accept new requests or frames
2. estimate visual tokens and KV blocks
3. decide process / skip / downscale / tile
4. admit requests if memory budget allows
5. schedule decode batch first when latency-sensitive requests exist
6. schedule chunked prefill without starving decode
7. run GPU work
8. sample next tokens
9. update KV block tables
10. stream output
11. release or cache memory
```

Important policies:

- continuous batching
- chunked prefill
- visual-token admission control
- frame freshness and frame dropping for stream workloads
- max batched tokens per step
- max active sequences
- fairness between small text requests and heavy visual requests
- cancellation and timeout handling

Fresh-frame scheduling should answer:

```text
Should this frame be processed, skipped, downscaled, tiled, or delayed?
Should decode traffic be protected from heavy vision prefill?
Should we spend budget on more image detail or more output tokens?
```

## KV Cache Design

Garnet should implement paged KV cache early.

Do not allocate one contiguous max-length KV tensor per request. Use fixed-size blocks:

```cpp
struct KVCacheBlock {
    int block_id;
    int device_id;
    int layer_id;
    int token_capacity;
    void* key_ptr;
    void* value_ptr;
    int ref_count;
    BlockState state;
};
```

Each sequence owns logical block tables:

```cpp
struct SequenceState {
    RequestId request_id;
    int current_length;
    std::vector<int> token_ids;
    std::vector<std::vector<int>> block_table_by_layer;
};
```

Benefits:

- less fragmentation
- higher concurrency
- prefix sharing
- copy-on-write branching
- easier eviction
- long-context support

## Multimodal Cache Design

Cache is useful, but it is not the main assumption for world-sense. The design should support all three:

```text
fresh image every request
near-duplicate frames in a stream
same visual context with multiple prompts
```

Recommended cache hierarchy:

```text
raw_image_hash
  -> processed_patch_cache
  -> vision_embedding_cache
  -> multimodal_decoder_kv_prefix_cache
```

Cache key components:

- model ID and model revision
- image/video content hash
- processor config
- resize/crop policy
- frame sampling policy
- text tokens
- visual token layout
- MRoPE metadata
- dtype and quantization mode

This enables high-value workloads:

```text
same screenshot + many questions
same document page + many extraction prompts
same camera frame + multiple agent decisions
same few-shot visual examples + many requests
nearby video frames + temporal reuse
```

Open question: for many cases, putting all questions into one prompt may be better than asking multiple prompts over the same image. Cache reuse is still useful for agent loops and event verification, but Garnet should not depend on repeated prompts as the main workload.

## System Prompt KV Reuse

Fixed system prompts, tool schemas, and output-format instructions are common in serving. In normal OpenAI-style chat layout, the system message usually appears before the user image:

```text
[system prompt tokens][user header tokens][image placeholder / visual tokens][question tokens]
```

For this common case, vLLM automatic prefix caching can already reuse the decoder KV blocks for the fixed system prompt across requests, even when each request has a different fresh image. Different images should not reuse the image-dependent visual region because the multimodal cache key includes media identity.

Example:

```text
request 1:
  system S + image A + question Q
  computes KV for S and visual region for image A

request 2:
  system S + image B + question Q
  can reuse KV for S if still cached
  recomputes image-dependent visual region for image B
```

This means Garnet should not claim fixed system-prompt KV reuse as a unique advantage. It should be treated as baseline serving behavior.

Garnet can still improve the developer and runtime control around this:

- expose the final prompt/media layout
- report which token blocks are cacheable text prefix, image-dependent visual span, and query suffix
- measure actual system-prefix cache hit rate and saved prefill time
- support optional pinning of fixed system/tool-schema prefix blocks
- keep optimizing the fresh visual path after the reusable system prefix

Potential report:

```text
cacheable_prefix:
  blocks 0-7: system prompt + tool schema

image_dependent_span:
  blocks 8-64: visual tokens for current frame

query_suffix:
  blocks 65-66: event verification question
```

This visibility is important because prefix reuse depends on final token/media order after the model adapter applies the chat template and multimodal placeholder expansion.

## Attention Backend

Expose a logical attention operation:

```cpp
AttentionOutput Attention(
    Tensor q,
    Tensor k,
    Tensor v,
    AttentionMask mask,
    PositionEncoding position,
    KVCacheView cache,
    AttentionParams params);
```

Backend progression:

```text
1. naive attention
   correctness and debugging

2. fused CUDA attention
   fewer kernel launches

3. paged attention
   read K/V through block tables

4. flash attention
   tiled online softmax, no full attention matrix

5. flash paged attention
   serving-grade path
```

No Triton is required. Garnet should favor:

- CUDA C++
- NVRTC
- CUTLASS/CuTe
- cuBLASLt
- custom CUDA kernels
- NCCL for multi-GPU communication

## GPU Execution

Start simple:

```text
single process
single GPU
one compute stream
```

Then add:

```text
stream_h2d
stream_d2h
stream_vision
stream_decode
stream_kv_copy
CUDA events for dependencies
```

CUDA Graphs should be added after correctness:

```text
uncaptured path:
  flexible shapes, debugging

captured decode graphs:
  common batch sizes, e.g. 1, 2, 4, 8, 16, 32

piecewise graphs:
  capture stable subgraphs while leaving dynamic scheduler work outside
```

Cross-platform notes:

```text
Windows:
  support MSVC + CUDA, local DLL/plugin workflow, desktop app integration

Linux:
  support CMake/Ninja + CUDA, server daemon, containers, NCCL, production profiling

Shared:
  keep core runtime C++ portable, isolate OS-specific process/IPC/server code
```

## CUDA IPC And Multi-GPU

CUDA IPC is useful but should not be first.

Use CUDA IPC when:

- same-node processes need to share GPU memory
- prefill worker and decode worker share KV cache
- vision worker and decoder worker exchange tensors without CPU copy
- a memory owner process exposes blocks to worker processes

CUDA IPC is platform-sensitive. Garnet should first design a same-process/single-GPU runtime that works on both Windows and Linux, then add IPC behind an abstraction.

First multi-GPU steps should use simpler deployment modes:

```text
1. data parallel model replicas
2. tensor parallel with NCCL
3. pipeline parallel
4. vision/decoder split
5. expert parallel for MoE
6. CUDA IPC for same-node memory sharing
7. disaggregated prefill/decode
```

## Required Ops For Qwen3-VL Dense

Text decoder:

- embedding
- linear / matmul
- rms_norm
- silu / swiglu
- reshape / view
- transpose / permute
- rope / mrope
- attention
- paged KV write
- paged attention read
- sampling

Vision path:

- image normalization
- patch embedding
- vision attention
- vision MLP
- visual token projector / merger
- multimodal position metadata

Later MoE support:

- topk
- grouped GEMM
- expert routing
- scatter/gather
- all-to-all

## Milestones

### Milestone 1: Correct Single-GPU VLM

Target:

```text
Qwen3-VL small dense model
single RTX 4080
batch size 1
BF16 or FP16
correct output compared with Transformers/vLLM
```

Deliverables:

- model config loader
- safetensors loader/converter
- Qwen3-VL adapter skeleton
- image processor metadata path
- basic tensor graph
- naive attention
- simple KV cache
- shape/logit comparison tests

### Milestone 1.5: Fresh-Frame Benchmark Harness

Target:

```text
measure fresh image -> first token on RTX 4080
compare Garnet path vs vLLM baseline
separate image preprocessing, vision encoder, decoder prefill, and decode time
```

Deliverables:

- benchmark runner for fresh images
- benchmark runner for frame streams
- visual token count reporting
- configurable resize/token budget
- Windows and Linux command lines
- first target workload decision record

### Milestone 2: Serving Core

Target:

```text
OpenAI-compatible server
continuous batching
paged KV cache
streaming decode
basic metrics
```

Deliverables:

- request state machine
- scheduler loop
- paged KV allocator
- decode queue
- prefill queue
- SSE streaming
- TTFT and tokens/sec metrics

### Milestone 3: VLM-Specific Gains

Target:

```text
reduce fresh-frame TTFT and prove where time is spent
sub-100 ms TTFT only for cached or reduced-budget modes if realistic
```

Deliverables:

- visual token admission control
- chunked visual/text prefill
- fresh-frame benchmark suite
- frame dropping / freshness policy prototype
- visual token budget modes
- raw image hash cache
- processed patch cache
- vision embedding cache
- multimodal KV prefix cache
- cache-aware benchmark suite

### Milestone 4: Performance Kernels

Target:

```text
approach practical vLLM baseline on Qwen3-VL-2B
```

Deliverables:

- cuBLASLt/CUTLASS matmul path
- fused rmsnorm
- fused rope/mrope + KV write
- paged attention
- flash attention
- CUDA graph decode
- optimized sampling

### Milestone 5: Scale

Target:

```text
multi-GPU serving and larger VLM/MoE models
```

Deliverables:

- data parallel replicas
- NCCL tensor parallel
- pipeline parallel
- vision/decoder split
- CUDA IPC memory sharing
- disaggregated prefill/decode prototype
- radix-style prefix cache

## Benchmark Plan

Baseline:

```text
vLLM + Qwen/Qwen3-VL-2B-Instruct
BF16
max_model_len 4096
single RTX 4080 if supported, otherwise nearest available GPU
```

Metrics:

- TTFT
- decode tokens/sec
- prefill tokens/sec
- end-to-end latency
- GPU memory used
- KV blocks used
- prefix cache hit rate
- image preprocessing time
- vision encoder time
- decoder prefill time
- max concurrent image requests
- dropped frame count for stream workloads
- latest-frame age at response time

Test cases:

```text
fresh small image + short question
fresh HD image + short question
fresh HD image + event verification prompt
fresh desktop screenshot + short structured answer
fresh frame stream at 1/5/10 FPS input
same HD image + many questions
same document screenshot + extraction prompts
multi-image prompt
video frame prompt
mixed text-only and VLM traffic
```

## Interview Summary

A concise explanation:

> Garnet is my cross-platform C++/CUDA VLM/LLM serving runtime. vLLM and SGLang already serve VLMs, so I use them as baselines. Garnet focuses on the serving problems that become sharp in world-sense VLM workloads: fresh image/frame latency, visual-token budgeting, vision prefill scheduling, paged KV cache, multimodal cache when reuse exists, flash/paged attention, CUDA graph decode, and GPU memory planning. The first target is Qwen3-VL, but the runtime is designed to be model-agnostic and to run on both Windows development machines and Linux serving nodes.

## Open Design Questions

These should be decided before implementation goes too far:

1. What is the first world-sense benchmark?

```text
desktop screenshot
webcam frame
robot/camera scene
game frame
document/OCR page
event verification image
```

2. What is the first latency target on RTX 4080?

```text
fresh image -> first token
fresh image -> structured yes/no answer
fresh image stream -> latest-frame answer
cached visual context -> first token
```

3. What output shape matters first?

```text
short text
yes/no event verification
JSON object
bounding box / coordinates
long answer
```

4. How much accuracy can fast mode trade for latency?

```text
resize whole image
center crop
tile only selected regions
two-stage small detector then VLM
temporal reuse across frames
```

5. What is Garnet's first comparison baseline?

```text
vLLM Qwen3-VL-2B-Instruct
Transformers Qwen3-VL-2B-Instruct
SGLang Qwen3-VL
```
