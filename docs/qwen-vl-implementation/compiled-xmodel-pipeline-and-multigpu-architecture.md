# Garnet Compiled xmodel Pipeline and Multi-GPU Architecture

## Status

This document is the normative architecture contract for Garnet VLM/LLM model
loading, compilation, execution, debugging, scheduling, and multi-GPU scaling.

When this document conflicts with older implementation notes, this document
wins. Existing hardcoded Qwen runners and Python-assembled subgraph tests are
reference and parity scaffolding only. They are not the production runtime
architecture.

## Goals

Garnet must provide this public flow:

```python
import xlang

garnet = xlang.importModule("garnet")

model = garnet.load_model(
    xmodel="xModel/qwen3/vl_2b_instruct/qwen_vl_model.py",
    weights="models/Qwen3-VL-2B-Instruct",
    compile={
        "dtype": "bfloat16",
        "profiles": {
            "prefill": {
                "min_total_tokens": 1,
                "opt_total_tokens": 512,
                "max_total_tokens": 4096,
                "max_batch_size": 4,
            },
            "decode": {
                "min_batch_size": 1,
                "opt_batch_size": 4,
                "max_batch_size": 16,
            },
            "vision": {
                "min_total_patches": 4,
                "opt_total_patches": 256,
                "max_total_patches": 4096,
                "max_batch_size": 4,
            },
        },
    },
    runtime={
        "max_model_len": 4096,
        "max_num_sequences": 8,
        "max_num_batched_tokens": 4096,
        "kv_page_size": 16,
        "kv_cache_memory_fraction": 0.60,
        "prefix_store_path": "cache/prefixes",
    },
)

result = model.forward({
    "prompt": "Detect the visible objects.",
    "image": {"path": "frame_0.jpg"},
    "max_new_tokens": 256,
})

print(result["text"])
```

For a fixed event-verification prefix, the public flow is:

```python
event_prefix_id = model.prepare_prefix(
    messages=[{"role": "system", "content": EVENT_RULES}],
    boundary="user_content",
    gpu_residency="pinned",
    placement="all_replicas",
)

result = model.forward({
    "prefix_id": event_prefix_id,
    "content": [
        {"type": "image", "path": "frame_0.jpg"},
        {"type": "text", "text": "Answer YES, NO, or UNKNOWN."},
    ],
    "max_new_tokens": 3,
})
```

`prepare_prefix` returns a persistent GUID string. The GUID is model-scoped and
resolves through Garnet's C++ `PrefixManager`; callers never construct or receive
GPU page IDs directly.

Python is a client of the model. Python must not build the model architecture,
construct layer bundles, allocate KV pages, drive the decode loop, or call
low-level DLL exports for normal inference.

## Non-Negotiable Invariants

1. The complete model graph originates from executing the selected `.x` model.
2. C++ implements generic graph lowering, execution, memory, and scheduling.
3. C++ must not encode the Qwen vision blocks, decoder layer order, or layer count.
4. Python calls `load_model`, optional probe configuration, and `forward`.
5. Model weights and intermediate tensors are GPU-backed `X::Tensor` values.
6. KV contents remain on GPU for the lifetime of a sequence.
7. JPEG and prompt preprocessing are outside the captured model graph.
8. The captured model graph begins with model-ready tensor inputs.
9. The runtime may use multiple TensorRT engines and custom CUDA nodes.
10. Existing compiled artifacts are reused when their compatibility key matches.
11. Debug probes are opt-in and must not impose production overhead when disabled.
12. Pipeline stages run concurrently across requests, subject to dependencies and
    per-GPU scheduling budgets.
13. Multi-GPU placement is fixed when the runtime graph is compiled; request-time
    scheduling executes that placement and does not repartition the model.
14. Garnet uses one serving process to control local GPUs. GPU IPC is not part of
    the normal local execution path.
15. Matching requests on one model replica share one physical set of immutable
    prefix KV pages; they must not receive copied per-request prefix pages.
16. `.x` functions express generic compilation intent through parameterized
    `T.fusion(...)` annotations. Garnet must not infer Qwen-specific engine
    boundaries from C++ model names or fixed layer counts.

## Architectural Boundaries

Garnet has three major data-plane stages:

```text
Raw request
    |
    v
InputProcessor
    prompt/image -> ModelInputs
    |
    v
RuntimeGraphRunner
    ModelInputs -> ModelOutputs
    |
    v
GenerationRuntime / OutputProcessor
    ModelOutputs -> final output
```

### InputProcessor

The input processor owns:

- native tokenization and chat-template expansion
- nvJPEG decode for encoded JPEG inputs
- raw CPU or GPU image input adaptation
- ROI and crop operations
- resize, padding, and normalization
- patch packing and model-specific image layout
- `image_grid_thw` and multimodal token metadata

It produces model-ready inputs:

```text
ModelInputs
  input_ids: GPU X::Tensor<int64>
  attention_mask: GPU X::Tensor<int64/bool>
  mm_token_type_ids: GPU X::Tensor<int64>
  pixel_values: GPU X::Tensor<fp32/bf16>
  image_grid_thw: GPU X::Tensor<int64>
  generation_options
```

The processor must support converging source forms:

```text
JPEG file
JPEG bytes
decoded CPU image
GPU RGB image tensor
GPU video frame
precomputed pixel_values
```

JPEG decode is not part of the captured model graph. Dense tensor operations after decode,
such as resize and normalization, may use custom CUDA/NPP or a separately cached
preprocessing TensorRT engine. Such an engine remains an InputProcessor detail.

### RuntimeGraphRunner

The plan runner owns only compiled model computation originating from `.x`:

- vision encoder
- visual embedding merger
- multimodal position computation represented as model tensor operations
- text prefill
- text decode
- model-side KV writes and paged attention
- final normalization and LM head
- model tensor memory planning
- graph-defined probe points

KV page allocation policy is not a model graph operation. The serving scheduler
allocates pages and passes page tables and slot mappings to model execution.

### GenerationRuntime and OutputProcessor

The generation runtime owns:

- request state transitions
- repeated decode scheduling
- sampling policy
- stop-token and stop-string handling
- incremental and final detokenization
- output validation when requested
- final small CPU result construction

Sampling may execute on GPU and may be captured with the decode CUDA Graph, but
sampling policy and generation-loop ownership remain runtime concerns rather
than xmodel architecture.

## xmodel Compilation

### Symbolic Execution

On a cache miss, `load_model` executes the `.x` forward function with symbolic
`X::Tensor` values and model configuration:

```text
qwen_vl_model.py
  -> imports vision_encoder.py, vl_adapter.py, qwen_llm.py
  -> symbolic execution
  -> complete xlang TensorGraph
```

Symbolic execution must:

- preserve stable tensor and node names
- expand fixed-depth loops from model configuration
- resolve compile-time configuration branches
- represent runtime branches explicitly or create specialized variants
- record tensor shapes, dtypes, devices, and weight references
- record custom/open operations without losing graph dependencies

For Qwen3-VL-2B, the 24 vision blocks and 28 text layers come from the `.x`
loops and configuration. They must not come from C++ loops or Python bundles.

### Generic Lowering

The backend lowers graph nodes through an operation registry:

```text
TensorGraph node
  -> HandleBinaryOp / HandleUnaryOp / structured-op handler
  -> TensorRT layer, custom CUDA node, or unsupported-op error
```

`HandleBinaryOp` and `HandleUnaryOp` must return real backend graph values. An
empty return is a compilation failure, not a successful build.

The lowering layer maps symbolic tensor identity to backend values:

```text
X::Tensor graph value -> TensorRT ITensor or custom-plan tensor binding
```

The compiler must report unsupported operations with the full xmodel node path,
shape, dtype, and requested backend. It must not silently switch to a hardcoded
model runner.

### Graph Partitioning

Not every operation must be inside TensorRT. The compiler partitions the graph:

```text
TensorRT region
    -> custom CUDA operation
    -> TensorRT region
    -> custom CUDA operation
```

Likely custom nodes include:

- paged KV write
- paged attention or selected flash-attention backend
- dynamic MRoPE helpers
- multimodal scatter/gather where TensorRT is unsuitable
- GPU sampling when attached by the generation runtime
- tensor probe reduction/snapshot nodes in debug variants

The partitions and their dependencies are compiler outputs, not hand-authored
Qwen execution order.

### Parameterized `T.fusion` Contract

Garnet extends the existing `@T.fusion(...)` decorator instead of introducing a
second `T.stage` construct. A fusion annotation names a captured graph region
and supplies constraints and hints to the generic partition planner. It is
compile-time metadata; it does not execute an operation, move a tensor, or
force a CPU synchronization.

```xlang
@T.fusion(
    name="vision",
    role="encoder",
    boundary="preferred"
)
def Qwen3VisionEncoder(...):
    ...


@T.fusion(
    name="text_prefill",
    role="transformer_prefill",
    boundary="required"
)
def Qwen3TextPrefill(...):
    ...


@T.fusion(
    name="text_decode",
    role="transformer_decode",
    boundary="required",
    cuda_graph=True
)
def Qwen3TextDecode(...):
    ...


@T.fusion(
    role="decoder_layer",
    atomic=True
)
def DecodeLayer(...):
    ...
```

Supported metadata has the following meaning:

| Parameter | Meaning |
| --- | --- |
| `name` | Stable region identity used by diagnostics, profiling, cache manifests, and generated engine names. |
| `role` | Generic workload classification such as `encoder`, `transformer_prefill`, `transformer_decode`, `decoder_layer`, `moe_router`, or `moe_expert`. It selects generic cost-model rules, never model-specific C++ code. |
| `boundary="required"` | The planner must end one backend partition and begin another at this function boundary. |
| `boundary="preferred"` | The planner should consider this boundary, but may merge adjacent compatible regions when memory and runtime costs favor one engine. |
| `boundary="none"` | No partition preference. This is the default. |
| `atomic=True` | The planner must not split inside the expanded function region. It may place a boundary immediately before or after it. |
| `cuda_graph=True` | The runtime may place executions of this region in a CUDA Graph bucket when its bindings and launch topology satisfy capture requirements. |

`T.fusion` means that operations inside the region are candidates for backend
fusion; it does not mean that every annotated function must become exactly one
TensorRT engine. The planner may merge adjacent `preferred` regions or group
multiple repeated atomic layers into one engine. It may not cross a `required`
boundary or split an `atomic` region.

Nested and repeated calls retain their expanded scope identities. For example,
an atomic `DecodeLayer` called by a 28-iteration `.x` loop produces 28 atomic
instances. The planner may create layer groups such as `[0, 7)`, `[7, 14)`,
`[14, 21)`, and `[21, 28)` according to measured build memory, runtime memory,
launch overhead, and device placement. Neither the annotation nor C++ embeds
the number 28 or those group sizes.

The compiler validates annotations before lowering:

- region names must be stable and unique after scope expansion
- `required` boundaries must have representable GPU tensor contracts
- an `atomic` region cannot contain a conflicting required inner boundary
- unknown roles remain valid metadata but receive only the default cost model
- unsupported parameter names or values are compilation errors

The existing `GarnetTensor::Fusion` API already receives positional and keyword
arguments. The implementation must copy those arguments into `Fusionist` and
then into the captured `TensorGraph` region metadata. Merely accepting
`params`/`kwParams` and ignoring them does not implement this contract.

### Partition Planner Inputs

Annotations constrain and guide the planner; they do not replace planning. The
generic planner combines:

- `T.fusion` region metadata and expanded function/loop topology
- backend operator support and custom CUDA operation boundaries
- min/opt/max input profiles
- weight, activation, workspace, and engine-build memory estimates
- expected invocation frequency for vision, prefill, and decode regions
- launch, synchronization, and cross-device transfer costs
- compile configuration, including target devices and build-memory limits

The planner emits an internal execution manifest containing partitions, engine
fingerprints, GPU bindings, cross-partition tensors, streams, events, custom
operations, and eligible CUDA Graph buckets. This manifest is a disposable
cache artifact. The `.x` files remain the only model-structure source of truth.

### Compile-Time VLM Flow Partitioning

The compiler partitions a VLM by execution phase as well as by supported
backend operations:

```text
captured .x TensorGraph
  -> vision.engine
  -> multimodal_prefill.engine
  -> text_decode.engine
```

These engine boundaries and their device assignments are static compiler
outputs. For example:

```text
GPU0: vision.engine
GPU1: multimodal_prefill.engine + text_decode.engine

vision.engine
  -> DeviceTransferNode(projected_visual_embeddings)
  -> multimodal_prefill.engine
  -> text_decode.engine
```

The partition cost must account for phase invocation frequency:

```text
cost = compute_time
     + engine_launch_overhead
     + boundary_bytes / measured_link_bandwidth
     + synchronization_cost
     + peak_memory_cost
     + repeated_execution_cost
```

Vision is normally invoked once per image, multimodal prefill once per request,
and decode once per generated token. A cross-device boundary in decode is
therefore penalized by expected output length, while a visual-embedding transfer
is normally paid only once. The compiler must strongly avoid decode-loop
communication unless tensor parallelism is required for model capacity.

The compiler may place the visual projection on either side of the vision/text
boundary. It compares the transfer shapes, projection compute, weight memory,
and device balance:

```text
vision tower + projection -> transfer [visual_tokens, text_hidden]
vision tower              -> transfer [visual_tokens, vision_hidden] -> projection
```

The runtime may pipeline different requests through these fixed engines, but it
must not dynamically move an engine or change the compiled partition.

### Multi-Engine GPU Execution

A non-monolithic compiled graph does not require tensor data to return to the
CPU and does not require a model-specific persistent GPU scheduler. Partition
outputs remain GPU-backed `X::Tensor` values and become bindings of downstream
engines or custom CUDA operations.

The C++ serving scheduler owns request admission, continuous-batch membership,
KV page allocation, and dependency submission. CUDA streams and events enforce
GPU data dependencies. Stable launch sequences may be captured as CUDA Graphs
to reduce per-iteration CPU launch overhead:

```text
C++ request/continuous-batch scheduler
  -> enqueue partition A on stream A
  -> record CUDA event A-ready
  -> stream B waits for A-ready
  -> enqueue partition B or custom CUDA node
  -> launch captured decode CUDA Graph when eligible
```

The CPU observes only compact scheduling state and selected output-token/status
metadata. It must not read intermediate activations, copy partition tensors, or
call `cudaDeviceSynchronize` in the normal path. A future persistent GPU
scheduler is an optional measured optimization, not a requirement created by
partitioning.

For partitions submitted to the same CUDA stream, stream ordering is the
dependency mechanism and no CUDA event is needed. An event is emitted only for
a dependency crossing streams or devices. This keeps the single-stream decode
path minimal while preserving an explicit DAG for concurrent vision, prefill,
transfer, and decode work.

## Runtime Graph and Compiled Cache

The `.x` source is the only authoritative description of model structure.
`load_model` first validates the recursive `.x` dependency hash. It executes the
root `.x` and captures a fresh `TensorGraph` only when the source, configuration,
compiler ABI, or cached graph has changed. An unchanged model loads its cached
captured graph without executing `.x`.

Compiled backend artifacts are cached independently:

```text
cache/<compatibility-key>/
  manifest.json
  runtime_graph.cache
  partitions/
    <partition-fingerprint>.engine
  optional_debug_variants/
```

`runtime_graph.cache` is an automatically generated, versioned serialization of
the graph previously captured by executing `.x`. It retains nodes, dependencies,
engine/custom-op bindings, device placement, memory assignments, stream
dependencies, KV interfaces, and probe names. It is a disposable startup cache,
not a second source of truth. Garnet must never load it unless all compatibility
fields match. Deleting it only causes `.x` to execute and recapture the graph.

### Cache Validation

Before executing `.x`, Garnet reads the manifest and validates its recorded
dependencies. The cache keys must include:

- recursive hash of the root `.x` file and all imported `.x` dependencies
- expanded graph/compiler ABI version
- model configuration hash
- precision and quantization configuration
- TensorRT optimization profiles
- TensorRT, CUDA, and custom-operation ABI versions
- GPU compute capability
- weight layout and, when embedded, weight content hash
- probe configuration when probes require extra engine outputs
- distributed placement, TP/PP degree, rank, and partition boundary contracts

Cache-hit flow:

```text
load_model
  -> read manifest and hash recorded root/imported .x dependencies
  -> validate configuration, compiler, backend, GPU, and weight compatibility
  -> deserialize runtime_graph.cache without executing .x
  -> deserialize matching .engine files
  -> bind graph regions and custom CUDA operations
  -> load/bind weights into GPU X::Tensor values
  -> create runtime state
  -> ready
```

Cache-miss or stale flow:

```text
load_model
  -> detect missing, changed, or incompatible cache
  -> run .x symbolically and capture TensorGraph
  -> lower and partition runtime graph
  -> reuse unchanged partition engines
  -> compile changed or missing TensorRT regions
  -> write manifest, runtime_graph.cache, and engines atomically
  -> bind the in-memory graph
  -> load/bind weights
  -> ready
```

Changing an `.x` file therefore never requires users to generate another graph
artifact manually. Reloading the model captures the new graph and recompiles
only backend partitions whose fingerprints changed. With no changes,
`load_model` neither executes `.x` nor recompiles engines.

Concurrent processes must use cache locking and atomic rename so no process can
observe partially written artifacts.

### Engine Recompilation Rules

Engine reuse is decided per TensorRT partition, not for the model as a whole.
The semantic partition fingerprint includes:

- operations, topology, and operation attributes
- input/output contracts and static dimensions
- dynamic-shape optimization profiles
- precision, quantization, calibration, and builder flags
- embedded weight content and layout
- TensorRT, CUDA, plugin, and custom-operation ABI compatibility
- target GPU compatibility
- distributed rank, sharding, and engine boundary contracts
- additional outputs required by a debug probe variant

Changing any of these fields recompiles that partition. Unchanged partition
fingerprints reuse their existing engines even when another `.x` region changed.
For the initial production implementation, weights are embedded in TensorRT
engines; changing a weight shard therefore recompiles its partition. TensorRT
refit support is a later feature for frequent weight or adapter switching.

The following runtime values do not recompile engines while they remain within
compiled profiles and memory limits:

- prompt, image, and token contents
- batch, token, and visual-patch counts within profile bounds
- `max_new_tokens`, sampling settings, and stop conditions
- KV occupancy, page allocation, queue state, and request composition
- tokenizer, JPEG decoder, and preprocessing implementation changes
- CUDA stream assignment that preserves the engine binding contract

## Compile and Runtime Limits

TensorRT dynamic dimensions require bounded min/opt/max profiles. Garnet should
compile around packed work dimensions rather than padded maximum shapes.

### Prefill

```text
hidden_states: [total_packed_tokens, hidden_size]
```

The compile profile bounds total packed tokens and participating requests.

### Decode

```text
hidden_states: [active_sequences, hidden_size]
```

Pure decode normally processes one token per active sequence. Speculative decode
may use more than one query token per sequence.

### Vision

```text
pixel/patch tensor: [total_packed_vision_patches, patch_feature_size]
```

Profiles should primarily bound packed patch count, not exact image width and
height. Image preprocessing converts source dimensions into this model contract.

### Runtime-Only Limits

`max_new_tokens` is a request option and does not invalidate TensorRT engines.
`max_model_len`, active sequence count, and KV memory limits belong to runtime
capacity planning unless they change compiled tensor profiles.

Requests outside every compiled profile follow an explicit policy:

- reject with a clear shape/profile error (production default)
- select another cached profile
- compile a new profile during controlled warmup

Request-time compilation is disabled by default.

## Parallel Pipeline Runtime

Pipeline stages run concurrently across different requests:

```text
Request A: text decode
Request B: text prefill
Request C: vision encoder
Request D: JPEG decode and preprocessing
Request E: output detokenization
```

The stages communicate through bounded queues and typed bundles. A GPU tensor
bundle contains:

```text
TensorBundle
  X::Tensor values
  shape/dtype/device metadata
  ownership/lifetime references
  producer CUDA event
```

The consumer waits through CUDA stream/event dependencies. It must not copy the
tensor to CPU or call `cudaDeviceSynchronize` between stages.

### Preprocess Instances

Multiple InputProcessor instances are allowed:

```text
GPU0 preprocess stream 0
GPU0 preprocess stream 1
GPU1 preprocess stream 0
CPU tokenizer worker pool
```

They share device-owned resource pools. The per-GPU scheduler applies a vision
and preprocess budget so frontend work cannot indefinitely delay decode.

### Persistent GPU Resources

Each GPU worker owns long-lived resources:

- CUDA context and persistent stream pool
- high-priority decode stream
- prefill, vision, and preprocessing streams
- TensorRT engines and execution contexts
- cuBLAS/cuBLASLt handles
- CUDA Graph executions
- activation/workspace pools
- model weight tensors
- KV page pools
- pinned metadata staging buffers

Streams, handles, engines, and large buffers are never created per operator or
per token.

## Serving Scheduler

### Scheduler Layers

Garnet has two scheduling levels:

```text
GlobalRouter
  -> choose model replica or distributed model group

PerModelGroupScheduler
  -> admission, continuous batching, token budgets, encoder budgets, KV pages
```

Each physical GPU has one device worker that serializes ownership decisions for
that device while launching independent work asynchronously.

### Request State

```text
Queued
Preprocessing
VisionReady
PrefillReady
Prefilling
Decoding
Finished
Failed
Cancelled
```

### Continuous Batching

At every model iteration, the scheduler assigns a token budget:

```text
Request A: 1 decode token
Request B: 1 decode token
Request C: 256 prompt tokens
Request D: 128 prompt tokens
```

The scheduler packs selected tokens subject to:

- `max_num_batched_tokens`
- decode batch profile
- prefill profile
- encoder/vision token budget
- available KV pages
- priority and latency policy

Long prefills are chunked so active decode requests continue making progress.

### CPU/GPU Iteration Boundary

The initial production design uses one C++ scheduling iteration per decode token
per active sequence:

```text
C++ selects active batch
  -> update small persistent GPU metadata
  -> launch decode CUDA Graph
  -> GPU runs model, KV update, LM head, and sampling
  -> collect token IDs and finished flags
  -> update request states
```

C++ does not schedule every transformer operator individually. The compiled plan
and CUDA Graph contain those operations. Tensor values and KV contents stay on
GPU. Only small token/status data crosses the iteration boundary when needed.

Future optimizations may execute multiple stable decode steps per CPU decision
or use speculative decoding, but must preserve continuous-batching semantics.

### CUDA Graph Buckets

Decode plans should capture stable batch buckets:

```text
batch 1, 2, 4, 8, 16, ...
```

A three-request batch can use a batch-four graph with one inactive slot. Batch
metadata and active masks use fixed GPU addresses so replay does not require
rebinding the complete graph.

## Paged KV Cache

KV storage is a persistent GPU page pool:

```text
key_pages[layer]: GPU X::Tensor
value_pages[layer]: GPU X::Tensor
page_tables: GPU X::Tensor<int32>
sequence_lengths: GPU X::Tensor<int32>
slot_mapping: GPU X::Tensor<int32>
```

The scheduler owns logical allocation:

- allocate pages during admission/prefill
- append pages at page boundaries
- release pages on completion/cancellation
- track request affinity to a model replica
- support immutable prefix block sharing and copy-on-write
- apply preemption or admission rejection when capacity is exhausted

The attention kernel receives page tables and sequence metadata. It never
allocates pages itself. KV data never moves through CPU memory during normal
inference.

For Qwen3-VL-2B BF16, approximate KV bytes per token are:

```text
28 layers * 2 (K,V) * 8 KV heads * 128 head_dim * 2 bytes
= 114,688 bytes/token, approximately 112 KiB/token
```

One full 4096-token sequence therefore needs approximately 448 MiB before page
fragmentation and metadata.

### Prefix KV Cache

Prefix caching is generic exact-prefix reuse, not a special system-prompt-only
execution path. A long fixed system prompt benefits naturally because its token
sequence appears before the per-request image and user content.

The cache index is a radix tree or chained block-hash table over complete KV
pages. For page `i`:

```text
block_key[i] = Hash(
    block_key[i - 1],
    model_and_adapter_identity,
    KV_layout_and_dtype,
    token_ids_for_block,
    position_ids_for_block,
    multimodal_content_signature
)
```

Each entry references the corresponding K/V pages for every layer and required
distributed rank. Cached pages are immutable and reference-counted:

```text
PrefixCacheEntry
  block_key
  token_count
  per_rank_layer_pages
  reference_count
  pin_count
  last_access
```

Within one GPU/model replica, every matching request page table references the
same physical prefix page IDs:

```text
prepared prefix, page size 16, 1,000 tokens:
  shared complete pages = P10..P71       # 62 pages, 992 tokens
  unmatched tail        = 8 tokens

request A block table:
  [P10..P71, A_private_tail, A_image_pages, A_decode_pages]

request B block table:
  [P10..P71, B_private_tail, B_image_pages, B_decode_pages]
```

`P10..P71` exist once in that GPU's KV pool regardless of the number of active
requests. Paged attention reads those shared pages through each sequence's block
table. KV writes are permitted only to request-private pages.

Request admission performs:

```text
tokenize and apply the exact chat template
  -> find the longest matching sequence of complete cached blocks
  -> retain shared page references
  -> prefill only the unmatched suffix
  -> use private pages for new KV
```

A partially filled shared page is never modified. The initial implementation
reuses only complete pages and recomputes at most `kv_page_size - 1` tail tokens
into one private page per request. Copy-on-write cloning of a partial page is a
later optimization.

Configured fixed system prompts use the same mechanism. `model.prepare_prefix`
tokenizes and prefills the prompt, stores it through C++ `PrefixManager`, and
returns a persistent GUID. `model.forward` accepts that GUID as `prefix_id`.
This does not create a separate fixed-prompt KV format. Unconfigured repeated
prefixes are discovered through the same exact block matching and admitted
according to frequency/recency policy.

For VLM requests, identical image placeholder token IDs do not imply identical
KV. Matching must stop before the first multimodal token unless the key includes
the exact media-content digest, preprocessing configuration, image grid,
multimodal position IDs, and embedding contract. Fresh-frame event verification
therefore reuses the system/rule prefix but always recomputes image and
frame-specific KV.

The preferred event-verification sequence is:

```text
fixed system/rules       -> shared cached KV
fresh image placeholders -> new vision embeddings and KV
short frame instruction  -> new KV
short decision output    -> ordinary paged decode KV
```

The cache is physically scoped per GPU/model replica. `placement="all_replicas"`
creates one local physical prefix copy on each replica under one logical handle;
it does not copy pages between GPUs per request. In TP/PP mode, every rank owns
its local KV shard pages, and a logical prefix hit is committed only if all
required ranks have those pages. Requests assigned to the same replica/group
share its rank-local physical pages.

Eviction releases a physical page only when cache ownership, pin ownership, and
all active request references reach zero. Completing one request removes only
its page-table references and cannot release pages still used by another
request.

For Qwen3-VL-2B BF16, sharing a 1,000-token system prefix avoids approximately
109 MiB of duplicate KV per additional concurrent request. Fifty requests would
otherwise consume approximately 5.45 GiB for that prefix alone; shared paging
keeps one approximately 109 MiB copy per GPU/model replica plus request-private
tail, image, and decode pages.

### Persistent Local Prefix Store

Every explicitly prepared prefix is a persistent C++-managed resource. Garnet
generates a UUID/GUID when `model.prepare_prefix` succeeds and returns its string
form to the caller:

```text
prefix_id = "6f4e94c1-7d61-4ca5-b7af-bbd9ec6a93f2"
```

`PrefixManager` owns the mapping from this stable ID to local snapshot metadata
and currently resident per-GPU page sets:

```text
PrefixManager
  prepare(messages, boundary, placement, residency) -> GUID
  resolve(GUID, target_replica) -> shared page references
  acquire(GUID, request_id)
  release(GUID, request_id)
  evict_gpu(GUID, replica)
  delete_prefix(GUID)
```

The local store layout is:

```text
<prefix_store_path>/<model-store-identity>/<prefix-guid>/
  manifest.json
  rank-0.kv
  rank-1.kv              # only for distributed layouts
```

The packed KV files contain tensor bytes and logical ordering, never CUDA
pointers or physical page IDs. On restore, `PrefixManager` allocates new local
GPU pages, loads the packed rank-local K/V tensors, and constructs new block
tables referencing those pages.

The manifest contains:

- GUID, creation time, boundary, token count, and exact token IDs
- model, weight, adapter, tokenizer, and chat-template identities
- position/MRoPE metadata
- KV dtype, page size, layer/head dimensions, and layout ABI
- captured graph/engine compatibility fingerprint
- TP/PP layout and rank count
- placement and requested GPU-residency policy
- per-file sizes and checksums

Store writes use a temporary directory plus atomic rename. On process restart,
the manager can resolve a known GUID without recomputing the prefix when all
manifest compatibility fields and checksums match. An incompatible or corrupt
entry is reported as stale and is never attached to a request.

Local persistence and GPU residency are separate:

```text
local snapshot:       remains until delete_prefix(GUID)
GPU pinned residency: restored at model startup and not LRU-evicted
GPU cached residency: loaded on first use and may be evicted under pressure
active request lease: cannot be evicted until that request releases it
```

Model unload releases GPU pages but does not delete the local prefix. Deleting a
prefix removes its local snapshot only after active request leases reach zero.
Automatically discovered prefixes remain ordinary GPU LRU entries unless they
are explicitly promoted through `prepare_prefix` and receive a GUID.

### Prefix Length and Context Cost

Prefix sharing does not remove context limits. Every request must satisfy:

```text
prefix_tokens
+ visual_tokens
+ per_request_text_tokens
+ generated_tokens
<= min(model_context_limit, compiled_max_model_len, runtime_page_capacity)
```

A cached prefix avoids running prefill for the prefix tokens themselves. New
visual/text suffix queries and every decode query must still attend to the
cached K/V. For Qwen3-VL-2B BF16, the approximate shared storage and K/V read
traffic per generated token are:

```text
KV_bytes(context_length) = context_length * 112 KiB
```

| Cached/total context | Shared KV size | Approximate KV read per decode token |
| ---: | ---: | ---: |
| 1,000 tokens | 109 MiB | 109 MiB |
| 4,000 tokens | 438 MiB | 438 MiB |
| 8,000 tokens | 875 MiB | 875 MiB |
| 16,000 tokens | 1.75 GiB | 1.75 GiB |
| 32,000 tokens | 3.5 GiB | 3.5 GiB |

Physical page sharing removes duplicate capacity across requests, but ordinary
paged attention still performs attention against the prefix for each request.
Expected optimized batch-one decode ranges on RTX 4080 are planning estimates:

| Total context | Estimated decode throughput |
| ---: | ---: |
| 1K-2K | 30-50 tokens/s |
| 4K | 25-45 tokens/s |
| 8K | 20-35 tokens/s |
| 16K | 14-28 tokens/s |
| 32K | 10-20 tokens/s |

These estimates require measurement after the compiled paged-attention path is
implemented. A future shared-prefix attention kernel may load one shared K/V
tile for multiple batched queries, but that optimization is not required for
initial correctness.

The event-verification profile should normally keep system/rule prefixes around
500-1,500 tokens. With a 1,000-token prefix, a balanced 943-visual-token request
uses approximately 2K context; a full-HD 2,040-visual-token request uses
approximately 3K. Both fit a 4,096-token profile with a short instruction and
decision output. Longer rules require a larger compiled profile and consume
more KV capacity and attention bandwidth.

## Multi-GPU Architecture

### Small Models: Replication

When a complete model fits one GPU, scale with independent replicas:

```text
GPU0: complete model + KV pool
GPU1: complete model + KV pool
GPU2: complete model + KV pool
GPU3: complete model + KV pool
```

The global router assigns a new request using:

- queue depth
- active token load
- free KV capacity
- vision workload
- usable prefix-cache locality

Sequence affinity remains on one replica after prefill. Migrating an active
sequence requires KV transfer and is not a normal balancing operation.

Within one GPU, Garnet normally uses one model weight copy and continuous
batching. Creating multiple duplicate model instances on one GPU wastes memory
and fragments scheduling unless explicitly required for isolation.

### Large Models: Compiler-Generated Graph Partitioning

When a model does not fit one GPU, the compiler partitions the runtime graph.
The partition is derived from `.x`, not hardcoded model-specific C++.

The logical graph is captured once and transformed into rank-local graphs:

```text
logical .x TensorGraph
  -> distributed partition pass
  -> rank-local engine/custom-op graphs
  -> explicit transfer and collective nodes
```

Each rank owns only its weight shards, persistent buffers, execution contexts,
and KV pages. A rank-local weight is represented by a GPU-backed `X::Tensor`;
Garnet must not load a complete model onto one GPU before sharding it.

The checkpoint loader maps source ranges directly to destination ranks using a
bounded host staging buffer, or a direct-storage backend when available:

```text
checkpoint index
  -> tensor sharding rule
  -> bounded read/transform
  -> destination GPU X::Tensor
```

No complete CPU weight copy remains after initialization.

#### Tensor Parallelism

Each participating GPU owns weight shards for the same layers. The distributed
plan inserts communication nodes such as:

- all-reduce
- all-gather
- reduce-scatter
- point-to-point transfer

Tensor parallelism favors latency but depends on interconnect bandwidth.

Transformer defaults are:

```text
QKV projection       column parallel
attention output     row parallel
MLP gate/up          column parallel
MLP down             row parallel
embedding/LM head    vocabulary parallel
```

GQA requires head-aware sharding. If the KV-head count is smaller than the TP
degree, the compiler must use grouped ranks, replicate selected KV heads, or
reject that TP degree.

#### Pipeline Parallelism

Different device stages own different graph regions or layer ranges:

```text
GPU0: vision and early text graph partition
GPU1: middle text graph partition
GPU2: later text graph partition
GPU3: final text partition and LM head
```

Packed microbatches flow through stages. Pipeline parallelism reduces some
collective communication but introduces bubbles and requires careful decode
microbatch scheduling.

#### Combined Parallelism

A distributed group may combine tensor and pipeline dimensions. The top-level
plan records:

- device mesh and ranks
- per-device subplans and engines
- tensor placement and sharding
- communication nodes
- cross-device dependencies
- per-rank workspace and KV placement

For example, eight GPUs may form two pipeline stages with four-way tensor
parallelism per stage, or two independent TP4 serving replicas. Placement is a
compile configuration and is included in graph and engine cache compatibility.

### Distributed KV Ownership

With tensor parallelism, each rank stores KV for its local KV heads. With
pipeline parallelism, each stage stores KV for its assigned decoder layers.
Ranks share logical sequence and page IDs but map them to local physical pages:

```text
logical page 42
  rank 0 -> local page 117
  rank 1 -> local page 103
```

Page allocation is committed across all required ranks. If any rank cannot
reserve its local pages, the request waits, is preempted, or fails admission;
other ranks must not retain a partial allocation.

### Vision Placement

Vision placement is part of compile-time flow partitioning, not a dynamic worker
choice. Supported static layouts include:

```text
colocated:       GPU0 = vision + text
modality split:  GPU0 = vision, GPU1..N = text
distributed:     GPU0..A = vision TP, GPUA..N = text TP
replicated:      each serving replica owns its vision and text engines
```

For a modality split, image tensors remain on the vision device and only the
selected graph-boundary tensor is transferred. The request scheduler may overlap
vision for request B with text decode for request A, but both follow the fixed
compiled placement.

### Cross-Platform Device Transport

Garnet controls all local GPUs from one process. A CUDA pointer is therefore an
ordinary process-local device pointer; CUDA IPC and cross-process memory handles
are not required by the serving data path.

The runtime graph expresses backend-neutral communication nodes:

```text
DeviceTransferNode
AllReduceNode
AllGatherNode
ReduceScatterNode
BroadcastNode
```

For point-to-point transfer on Windows and Linux:

```text
same device                -> tensor alias, no copy
peer access available      -> cudaMemcpyPeerAsync
peer access unavailable    -> pinned-host staged asynchronous D2H/H2D
```

Garnet queries every device pair with `cudaDeviceCanAccessPeer`, enables peer
access where supported, and measures bandwidth/latency for the compile-time
partition cost model. It uses CUDA copy APIs rather than custom copy kernels.
The staged fallback is allowed only when hardware cannot provide P2P; tensor
contents must never be inspected or transformed by the CPU.

For collective operations:

```text
Linux    -> NCCL backend
Windows  -> Garnet CUDA collective backend when TP is required
```

The Windows backend implements collective algorithms using CUDA P2P transfers
and reduction kernels; it does not reimplement memory copy. Until that backend
exists, Windows compilation must reject a distributed layout requiring an
unsupported collective instead of silently moving tensors through Python.

### GPU Synchronization

Memory transfer and readiness notification are separate concerns. Dependencies
are represented with persistent CUDA streams and events:

```text
producer engine stream
  -> cudaEventRecord(producer_ready)
transfer stream
  -> cudaStreamWaitEvent(producer_ready)
  -> asynchronous peer or staged copy
  -> cudaEventRecord(transfer_ready)
consumer engine stream
  -> cudaStreamWaitEvent(transfer_ready)
  -> consumer engine
```

NCCL collectives are enqueued on CUDA streams and participate in the same event
dependency graph. The CPU scheduler submits work and updates compact request
state, but does not copy tensor contents, poll tensor memory, or synchronize
after every operation. `cudaDeviceSynchronize` is forbidden in the normal
request path.

## Backpressure and Memory Lifetime

All stage queues are bounded. Limits include:

- pending preprocessing requests
- ready vision patches/tokens
- active sequences
- batched prompt/decode tokens
- available KV pages
- pending output work

When the model stage is saturated, preprocessing admission slows instead of
accumulating unconsumed image tensors in GPU memory.

Default lifetimes:

- decoded image tensor: released after preprocessing/patch production
- pixel values: released after vision execution
- visual embeddings/deepstack features: request-local until prefill completes
- temporary model activations: execution-plan memory pool
- KV pages: retained until sequence completion/cancellation
- weights, engines, workspaces, and graph executions: model lifetime

## Probe and Correctness ABI

Probes attach to stable names generated by the expanded `.x` graph:

```python
model.set_probes([
    {
        "tensor": "vision.blocks.0.attn.qkv.output",
        "capture": "stats",
    },
    {
        "tensor": "language_model.layers.3.output",
        "capture": "tensor",
        "tokens": "last",
    },
    {
        "tensor": "kv.layers.3.key",
        "capture": "slice",
        "head": 0,
        "token_range": [0, 8],
    },
])

result = model.forward(request)
snapshots = model.probe_results()
```

Capture modes:

- `metadata`: shape, dtype, device, and bytes
- `stats`: min, max, mean, norm, NaN/Inf count, checksum
- `sample`: selected elements
- `slice`: selected tokens, heads, or channels
- `tensor`: full GPU snapshot for debug/parity only

Stats are reduced on GPU and only their compact result is returned. Full tensor
snapshots remain GPU `X::Tensor` values until the test explicitly calls
`garnet.tensor_to_cpu`. Only CPU tensors can be converted to NumPy.

If a requested probe is internal to a TensorRT region, Garnet compiles a debug
engine variant that marks the selected tensor as an additional output. Probe
configuration is then part of that debug variant's cache key. Probes disabled
must have no production execution cost.

HF comparison tests may use Python as a correctness oracle, but Python must not
control Garnet model execution. The same request is independently run through HF
and through one Garnet `model.forward` call, then selected outputs are compared.

## Metrics

Low-overhead request metrics include:

- tokenize time
- JPEG decode and image preprocessing time
- queue/admission time
- vision execution time
- text prefill time
- time to first token
- per-token and aggregate decode time
- generated tokens per second
- active sequences and batched tokens
- KV pages used/free and prefix-cache hits
- GPU workspace and peak memory
- per-link transfer bytes, latency, and achieved bandwidth
- collective time by operation and distributed rank
- P2P versus pinned-host-staged transfer counts
- per-stage queue depth and dropped/cancelled requests

Detailed node timings are debug/profiling mode and must not require full tensor
readback.

## RTX 4080 Event Verification Targets

These values are engineering estimates for acceptance planning, not completed
Garnet measurements. They must be replaced or annotated with measured values
after the compiled runtime is implemented.

The target workload is:

```text
model:                Qwen3-VL-2B-Instruct, BF16
GPU:                  one RTX 4080
model state:          loaded, engines warm
prefix:               long system/rule prompt already in shared KV
per-frame text:       fewer than 20 new text tokens
output:               fewer than 10 constrained tokens
batch latency:        batch 1 unless otherwise stated
```

The existing non-production runner establishes this baseline for the 65,536
pixel/60-visual-token test path:

```text
steady vision:        299-316 ms in the latest multi-image run
earlier best vision:  80.9-82.6 ms
steady text prefill:  394-422 ms in the latest multi-image run
decode average:       63.37 ms/token, 15.8 tokens/s
```

The compiled BF16 target for decode is:

```text
batch 1: 20-33 ms/iteration, 30-50 tokens/s
batch 4: 31-50 ms/iteration, 80-130 aggregate tokens/s
batch 8: 47-73 ms/iteration, 110-170 aggregate tokens/s
```

Continuous batching targets aggregate throughput. It does not promise the same
per-request token rate at every batch size.

The 60-token test image is too small to represent normal event-verification
quality. Two target profiles are:

```text
balanced quality:
  approximately 1312x736, 3,772 vision patches, 943 merged visual tokens

full-HD quality:
  approximately 1920x1088 after alignment, 8,160 vision patches,
  2,040 merged visual tokens
```

Estimated warm latency with a prefix-KV hit:

| Stage | Balanced, 943 visual tokens | Full HD, 2,040 visual tokens |
| --- | ---: | ---: |
| nvJPEG and preprocessing | 5-15 ms | 8-20 ms |
| vision engine | 180-350 ms | 300-650 ms |
| unmatched visual/short-text prefill | 120-250 ms | 250-500 ms |
| first decision token | 20-33 ms | 20-33 ms |
| five to nine additional tokens | 100-300 ms | 100-300 ms |
| complete result, under ten tokens | 0.43-0.95 s | 0.68-1.50 s |

Initial product targets are:

```text
balanced image-to-first-decision-token:  < 600 ms
balanced one-to-five-token result:       < 700 ms
balanced stretch result:                 < 400 ms
full-HD under-ten-token result:          < 1.5 s
batch-4 aggregate decode:                > 100 tokens/s
```

Sub-100 ms is not a target for a fresh regular-resolution frame because vision
and visual-token prefill must run for every new image. It is valid only for a
cached image/visual prefix or a much smaller visual-token budget.

The benchmark must report image-to-first-token and image-to-complete-result
separately. Complete latency is approximately:

```text
total = preprocess + vision + unmatched_prefill
      + first_token + (generated_tokens - 1) * decode_iteration
```

For yes/no event decisions, prefer one constrained decision token plus a GPU
confidence/reason-code result. Formatting fixed JSON field names and punctuation
outside model generation avoids spending decode iterations on deterministic
syntax. If the model must generate JSON, token count remains part of the latency
contract.

## Acceptance Tests

An end-to-end Garnet model test is architecturally valid only when:

1. Python imports Garnet through xlang.
2. Python calls `load_model` with the root xmodel and model directory.
3. Garnet validates or compiles the complete `.x` runtime graph.
4. Python calls one `model.forward(request)`.
5. Garnet owns preprocessing, model execution, KV, scheduling, and generation.
6. Python reads the final result and optional probe outputs.
7. No Python-created per-layer/subgraph model bundles are used.
8. No normal inference stage calls low-level DLL exports through `ctypes`.
9. The C++ runtime contains no Qwen-specific layer loop or operation ordering.
10. A cache-hit test proves `.x` is not executed and TensorRT compilation does
    not occur.
11. A modified `.x` dependency invalidates the captured-graph cache and
    recompiles only partitions whose semantic fingerprints changed.
12. Probe-enabled and probe-disabled correctness match at non-probed outputs.
13. A fixed vision/text placement test proves each engine executes on its
    compiled device and transfers the declared boundary tensor only.
14. P2P and forced pinned-staging transport tests produce identical tensors.
15. A synchronization test proves no normal request-path
    `cudaDeviceSynchronize` is issued.
16. A distributed-load test proves each rank owns only its declared weight and
    KV shards.
17. Windows compilation rejects unsupported collective layouts explicitly;
    Linux NCCL and Windows CUDA collective backends pass the same collective
    correctness contract when available.
18. Repeated identical system prefixes share the same physical immutable KV
    page IDs within a replica, do not increase physical prefix-page count, and
    prefill only the unmatched request suffix.
19. A one-token prefix difference misses at the affected block, and a shared
    partial block is never modified by an active request.
20. Two different images with identical placeholder token IDs never share image
    KV unless their complete multimodal content signatures match.
21. The event-verification benchmark reports warm prefix-hit TTFT, complete
    latency, visual tokens, output tokens, and batch throughput for both target
    image profiles.
22. `placement="all_replicas"` creates exactly one physical prefix copy per
    replica, and releasing one request cannot free pages referenced by another.
23. A prepared prefix GUID survives process/model restart, restores compatible
    rank-local KV snapshots into new physical GPU pages, and produces the same
    continuation logits as a fresh prefix prefill.
24. Model, tokenizer, adapter, KV-layout, distributed-layout, and checksum
    mismatches reject a persisted prefix as stale without attaching its pages.
25. `delete_prefix` is atomic with active leases: new acquisitions stop, local
    files remain until active requests release them, and then storage is removed.

## Forbidden Production Patterns

The following patterns are explicitly non-production:

- Python loading one model object per norm, projection, attention, or MLP
- Python assembling 24 vision or 28 text layer bundles
- C++ `QwenTextRunner` or `QwenVisionRunner` encoding model architecture
- direct `ctypes` calls for end-to-end preprocessing, KV, or model execution
- `HandleBinaryOp`/`HandleUnaryOp` returning empty while compilation reports success
- silent fallback from unsupported graph lowering to hardcoded Qwen execution
- CPU NumPy tensors between model subgraphs
- per-token stream, engine, handle, or large-buffer construction
- synchronization after every operator
- `cudaDeviceSynchronize` in the normal request path
- custom CUDA kernels used only to replace `cudaMemcpyPeerAsync`
- GPU IPC in the ordinary same-process local-GPU execution path
- runtime movement of engines between devices after compile-time placement
- duplicate full model copies on one GPU for ordinary continuous batching

Existing tests may retain these mechanisms temporarily as parity scaffolding,
but they must be labeled accordingly and cannot satisfy production milestones.

## Implementation Order

1. Freeze the `load_model`, `forward`, and probe ABI.
2. Add an acceptance test that loads only `qwen_vl_model.py` and currently fails.
3. Make generic TensorGraph lowering fail loudly on unsupported operations.
4. Implement backend tensor identity mapping and binary/unary operation registry.
5. Compile and execute a complete small `.x` graph without subgraph assembly.
6. Add graph partitioning, partition fingerprints, and engine cache validation.
7. Lower Qwen vision, multimodal, prefill, and decode graph regions incrementally.
8. Integrate runtime-owned KV pages, exact block-prefix sharing, C++
   `PrefixManager` with GUID/local persistence, and continuous batching.
9. Add persistent buffers and CUDA Graph decode buckets.
10. Add compile-time VLM flow partitioning and cross-platform CUDA P2P/staged
    transport nodes.
11. Add multi-replica routing, distributed weight/KV ownership, and
    compiler-generated TP/PP partitions.
12. Add Linux NCCL collectives, followed by the Windows CUDA collective backend.
13. Enable stable graph probes and HF intermediate parity.
14. Optimize kernels only after execution is proven to originate from `.x`.
