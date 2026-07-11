# Compiled xmodel Serving Runtime Implementation Plan

## Status

This is the execution plan for implementing the normative architecture in
`compiled-xmodel-pipeline-and-multigpu-architecture.md`.

It replaces implementation sequences that optimize Python-assembled TensorRT
subgraphs, direct DLL orchestration, or hardcoded Qwen C++ runners as the
production path. Existing runners remain correctness and performance references
until this plan reaches end-to-end parity.

No stage is complete because a smoke test returns text. A stage is complete only
when its ownership, graph-source, device-residency, correctness, and performance
gates pass.

## Branch Strategy

Use a new branch for implementation:

```text
feature/xmodel-compiled-serving-runtime
```

The current `feature/qwen-vl-trt-architecture` branch and checkpoint `b909047`
retain the working Qwen-VL runner baseline. Before creating the new branch:

1. Review and checkpoint the normative architecture and this plan explicitly.
2. Keep unrelated IDE, result, cache, and temporary files out of the checkpoint.
3. Record the existing correctness output and RTX 4080 benchmark artifacts.
4. Branch from that reviewed checkpoint.

Do not silently move the current dirty working tree to a new branch or commit it
without explicit approval.

## Final Product Contract

The public runtime flow is:

```python
model = garnet.load_model(
    xmodel="qwen_vl/xmodel/qwen_vl_model.x",
    weights="models/Qwen3-VL-2B-Instruct",
    compile=compile_config,
    runtime=runtime_config,
)

prefix_id = model.prepare_prefix(
    messages=event_system_messages,
    boundary="user_content",
    gpu_residency="pinned",
    placement="all_replicas",
)

result = model.forward({
    "prefix_id": prefix_id,
    "content": [
        {"type": "image", "gpu_tensor": frame},
        {"type": "text", "text": "Answer YES, NO, or UNKNOWN."},
    ],
    "max_new_tokens": 3,
})
```

Python/xlang invokes this API and may inspect probes. It does not build layers,
drive prefill/decode, allocate KV, schedule GPUs, or call internal DLL exports.

## Global Guardrails

Every stage must preserve these rules:

1. The root `.x` execution or its validated captured-graph cache is the only
   model-structure authority.
2. C++ lowering and scheduling are generic; Qwen layer order is not hardcoded.
3. Model weights, activations, visual embeddings, logits, and KV are GPU
   `X::Tensor` values in the production path.
4. CPU code submits work and manages compact metadata; it does not process model
   tensor contents.
5. Unsupported lowering fails with node path, shape, dtype, and backend reason.
6. No production fallback may call `QwenTextRunner`, `QwenVisionRunner`, Python
   subgraph bundles, NumPy intermediates, or `ctypes` orchestration.
7. Existing runners may only be called by explicitly named reference/parity
   tests.
8. Performance work begins only after the corresponding production ownership
   and correctness tests pass.

## Stage 0: Freeze Baseline and Failure Tests

### Deliverables

- Preserve current four-image output, logits/probe parity, and RTX 4080 metrics.
- Add a production-path test that loads only `qwen_vl_model.x`.
- Add forbidden-path counters for Python bundles, hardcoded runners, CPU tensor
  conversions, and direct internal exports.
- Mark the new test expected-fail until generic graph execution exists.

### Exit Tests

- Existing baseline remains reproducible.
- The new test fails at a precise unsupported graph/lowering point.
- It cannot pass through a hardcoded or Python fallback.

## Stage 1: Freeze Public ABI and Runtime Ownership

### Deliverables

- C++/xlang objects for `Model`, `Request`, `Result`, `ProbeConfig`, and
  `PrefixManager`.
- `load_model`, `forward`, `prepare_prefix`, `delete_prefix`, and probe APIs.
- Structured compile/runtime configuration validation.
- One C++ model owner for tokenizer, image processor, graph runner, scheduler,
  KV manager, prefix manager, and output processor.

### Exit Tests

- Python performs one `load_model` and one `forward` call.
- Invalid configuration fails before GPU allocation.
- Object lifetimes and GPU ownership survive repeated create/destroy cycles.
- ABI tests pass through xlang without direct `ctypes` calls.

## Stage 2: Captured Graph and Cache Authority

### Deliverables

- Execute the root `.x` with symbolic `X::Tensor` values.
- Capture stable nodes, tensors, loops, branches, source paths, and bindings.
- Recursively record root/imported `.x` dependency hashes.
- Serialize `runtime_graph.cache` as disposable derived data.
- Validate graph/compiler/config compatibility before cache reuse.

### Exit Tests

- First load executes `.x` and captures the complete test graph.
- Unchanged reload does not execute `.x`.
- Changing one imported `.x` file recaptures the graph.
- Deleting/corrupting the cache safely recaptures it.
- Cache data can never override a changed `.x` dependency.

## Stage 3: Generic Backend Lowering

### Deliverables

- Backend-value identity map from graph `X::Tensor` to TensorRT/custom values.
- Real binary, unary, structured-op, control-flow, and function-call handlers.
- Explicit operation registry and unsupported-operation diagnostics.
- Tensor metadata propagation and weight binding independent of Qwen names.

### Exit Tests

- Function, five-iteration loop, and static-branch graphs compile and execute.
- A complete small `.x` model executes without subgraph assembly.
- `HandleBinaryOp` and `HandleUnaryOp` never report success with empty values.
- CPU and TensorRT references pass numerical tolerance gates.

## Stage 4: Engine Partitioning and Incremental Cache

### Deliverables

- Partition captured graphs into TensorRT regions and custom CUDA nodes.
- Calculate semantic fingerprints per partition.
- Compile and atomically store rank/device-specific engines.
- Reuse unchanged partitions after a local `.x` change.
- Select dynamic profiles for vision, prefill, and decode.

### Exit Tests

- Changing one partition rebuilds only that engine.
- Prompt/image contents and runtime batch values within profiles do not rebuild.
- Shape/profile, precision, weight, plugin, or engine-boundary changes rebuild
  the affected engine.
- Concurrent cache loaders never observe partial artifacts.

## Stage 5: Compile-Time VLM Flow Placement

### Deliverables

- Identify vision, multimodal prefill, and decode invocation regions from graph
  dependencies and compiler annotations, not model-specific C++ ordering.
- Estimate compute, memory, transfer, synchronization, and invocation-frequency
  costs.
- Emit fixed engine/device placement and explicit transfer nodes.
- Keep decode-loop communication strongly penalized.

### Exit Tests

- Colocated and split vision/text layouts produce matching results.
- Each engine runs only on its compiled device.
- Only declared boundary tensors cross devices.
- Runtime cannot dynamically move engines or change the partition.

## Stage 6: Full Qwen-VL Graph Correctness

### Deliverables

- Lower vision tower, visual projection/merge, MRoPE, text prefill, text decode,
  final norm, LM head, and GPU sampling from the root `.x` graph.
- Bind real Qwen3-VL-2B BF16 weights.
- Provide stable graph probes for HF intermediate parity.
- Remove hardcoded runner reachability from the production model object.

### Exit Tests

- Vision block, visual merge, selected text layers, logits, and token probes
  match HF/reference tolerances.
- Dataset images produce semantically correct outputs.
- Production counters prove zero hardcoded-runner/Python-subgraph use.
- All intermediate model tensors remain GPU-backed.

## Stage 7: Paged KV and Persistent PrefixManager

### Deliverables

- Per-GPU persistent K/V page pools, block tables, slot mappings, and lengths.
- Immutable complete-page sharing with request reference counts.
- Recompute at most one partial prefix tail page per request initially.
- C++ `PrefixManager` generating persistent GUID IDs.
- Atomic local store containing manifests and packed rank-local KV snapshots.
- Per-replica physical page materialization and TP/PP rank-local shards.
- GPU pinned/cached residency, leases, eviction, and explicit deletion.
- Exact block keys including model, adapter, token, position, KV layout, and
  multimodal identity.

### Exit Tests

- Fifty requests referencing one prefix use one physical complete-page set per
  GPU/replica, not fifty copies.
- Shared pages are never written; suffix/image/decode pages are private.
- One-token prefix differences miss at the correct block.
- Different images with identical placeholder IDs cannot share image KV.
- A GUID survives process restart and restores logits identical to fresh prefill.
- Incompatible/corrupt snapshots fail as stale.
- Prefix deletion waits for active leases and then removes local storage.

## Stage 8: Generation Runtime and Continuous Batching

### Deliverables

- Separate chunked prefill and one-token decode queues.
- One batched decode iteration produces one token per active sequence.
- GPU sampling, stop-token state, and compact completion status.
- Admission, preemption, cancellation, and KV-capacity policy.
- Persistent buffers and CUDA Graph buckets for common batch sizes.

### Exit Tests

- Requests join and leave the decode batch without corrupting page tables.
- Batch 1/4/8 results match independent generation.
- No logits or intermediate activations move to CPU.
- No normal request issues `cudaDeviceSynchronize`.
- Cancellation releases only request-private pages and references.

## Stage 9: Native Fresh-Frame Pipeline

### Deliverables

- nvJPEG decode, ROI/crop, resize, normalize, and patch pack on GPU.
- Raw GPU image tensor and precomputed pixel-value input paths.
- Per-request pipeline overlap across preprocess, vision, prefill, and decode.
- Bounded queues and GPU-memory backpressure.

### Exit Tests

- JPEG and raw-GPU inputs match processor contracts.
- No decoded image or visual tensor round-trips through CPU.
- Four-image test runs after one warmup with per-stage metrics.
- Queue saturation cannot grow GPU image memory without bound.

## Stage 10: Cross-Platform Multi-GPU Transport

### Deliverables

- Device topology table and measured pair bandwidth/latency.
- `DeviceTransferNode` using alias, `cudaMemcpyPeerAsync`, or pinned asynchronous
  staging fallback.
- Persistent transfer streams and event dependencies.
- Linux NCCL collective backend.
- Explicit Windows rejection for unsupported collective layouts, followed by a
  CUDA P2P/reduction collective backend.

### Exit Tests

- Direct-P2P and forced-staging copies are bit-identical.
- Producer, transfer, and consumer event ordering passes race stress tests.
- CPU never reads transfer tensor contents.
- Linux collectives pass rank parity; Windows never silently falls back through
  Python or CPU tensor operations.

## Stage 11: Distributed Weights and KV

### Deliverables

- Direct checkpoint-range loading into rank-local GPU `X::Tensor` shards.
- Column/row/vocabulary parallel rules and GQA-aware KV-head placement.
- Pipeline layer ownership and combined TP/PP device meshes.
- Coordinated distributed KV page allocation and prefix restoration.

### Exit Tests

- No rank temporarily owns the complete model.
- Rank-local weight checksums match expected checkpoint slices.
- Distributed logits/tokens match the single-GPU reference.
- Partial distributed KV/prefix allocation rolls back atomically.

## Stage 12: Performance Closure

### Deliverables

- BF16 fused engine regions and selected custom CUDA attention kernels.
- Persistent TensorRT contexts/workspaces and allocation-free steady decode.
- Paged/flash attention and optional shared-prefix attention optimization.
- RTX 4080 event-verification benchmark and quality gates.

### Target Gates

For a warm Qwen3-VL-2B model with a cached 500-1,500-token system prefix:

```text
batch-1 decode:                         30-50 tokens/s at 1K-2K context
batch-4 aggregate decode:               > 100 tokens/s
balanced image first decision token:    < 600 ms
balanced one-to-five-token result:      < 700 ms
balanced stretch result:                < 400 ms
full-HD under-ten-token result:         < 1.5 s
```

Targets are not completion claims. The report must include visual/context/output
token counts, prefix hit status, TTFT, complete latency, percentiles, GPU memory,
and quality result. Any missed gate must be explained by measured stage/kernel
timings rather than hidden by a smaller image or shorter output.

## Required Test Matrix

Every release candidate runs:

```text
OS:             Windows, Linux
GPU mode:       single GPU, replica, P2P split, staged-copy fallback
graph cache:    miss, hit, stale dependency, corrupt artifact
prefix:         miss, auto hit, GUID restore, pinned, evicted/reloaded, deleted
image profile:  60-token debug, 943-token balanced, 2,040-token full HD
generation:     1, 5, 10, 100 tokens
batch:          1, 4, 8 with dynamic membership
correctness:    HF probes, single-GPU/distributed parity, semantic output
```

## Definition of Done

The redesign is complete only when a clean process can load the root `.x`, reuse
or compile its graph/engines, restore a GUID prefix, accept a fresh JPEG/GPU
frame, execute GPU-resident vision/prefill/paged decode with continuous batching,
and return a correct short decision through one `model.forward` call. The same
test must prove no production dependency on Python graph assembly, direct
internal exports, hardcoded Qwen runners, or CPU model tensors.
