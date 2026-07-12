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

### Implementation Progress (2026-07-11)

The new implementation branch now has a production-path foundation with these
verified properties:

- Parameterized `@T.fusion(...)` annotations now accept and validate `name`,
  `role`, `boundary`, `atomic`, and `cuda_graph` metadata directly from xlang
  decorator expression objects. Nested calls and repeated loops retain parent,
  depth, and invocation identity during compiled graph capture.
- Graph cache V2 persists a generated execution-plan sidecar. Cache hits restore
  fusion regions and partition candidates without executing `.x`. Runtime
  status explicitly reports `cpu_control_gpu_execution`, current CUDA stream
  ordering, planned cross-partition CUDA events, and no intermediate host-copy
  or device-wide-sync policy.
- The real Qwen text-decode expression captures one required CUDA-Graph-eligible
  decode region and 28 distinct atomic decoder-layer instances. Its paged-KV
  regression remains numerically valid.
- Required nested fusion boundaries now produce physical TensorRT engines. A
  two-partition fixture records the operation DAG, derives each function's
  operation slice from its tensor input/output contract, compiles two engine
  files, and passes the producer's GPU `X::Tensor` allocation directly to the
  consumer engine on the same CUDA stream. Numerical parity and cache-hit reuse
  pass with no intermediate host tensor or device-wide synchronization.
- Preferred boundaries now use inclusive operation cost, while repeated atomic
  regions can be grouped with `max_atomic_regions_per_partition`. Required and
  selected preferred regions produce stable, topologically ordered physical
  engine partitions. Incremental per-partition rebuilds and optional
  multi-stream CUDA event edges remain Stage 4 work. Same-stream engine chains
  require no event; stream ordering already enforces their GPU dependency.

- `runtime_mode="compiled_xmodel"` loads the selected root `.x` and does not
  invoke Python subgraph assembly, direct internal exports, or hardcoded Qwen
  runners.
- Qwen model expressions use `garnet.tensor()` as their tensor-expression
  implementation and select the TensorRT backend.
- A decorated root function can execute with symbolic `X::Tensor` inputs and
  return an xlang `TensorGraph` without entering the old NVRTC launch path.
- Local imported `.x` files, explicit `# garnet-dependency:` files, entry
  function, input shapes, runtime schema, and weight-location metadata
  contribute to the graph compatibility fingerprint.
- A validated graph/engine cache hit performs zero root `.x` executions.
  Missing, stale, or corrupt graph cache data recaptures and atomically replaces
  derived artifacts.
- Generic TensorRT lowering maps graph tensor identity across an
  `add -> multiply -> subtract` chain, nested functions, a five-iteration
  xmodel loop, static branches, matrix multiplication, Qwen-layout linear
  projection, and activation unary ops.
- FP32 and BF16 symbolic input contracts compile to matching TensorRT bindings.
  Execution returns GPU `X::Tensor` outputs; BF16-to-FP32 conversion occurs only
  when the test explicitly calls `tensor_to_cpu` as an observation boundary.
- Unsupported operators fail with the operation name and do not publish an
  engine artifact.
- The complete Qwen3-VL-2B root expression captures from the real 625-tensor,
  4,255,064,064-byte checkpoint without a native xlang assertion. Model
  weights are resolved by canonical safetensors name through a read-only mapped
  file; graph metadata never owns temporary weight tensor wrappers.
- The checkpoint names in `.x` now exactly use `model.visual.*` and
  `model.language_model.*`. Embedding, flattened Conv3D patch projection,
  learned-position interpolation, vision LayerNorm/MLP/RoPE/attention,
  DeepStack and final patch mergers, multimodal scatter, text RMSNorm/QKV/
  interleaved MRoPE/causal GQA, DeepStack injection, and tied LM head all lower
  from the captured graph.
- Grid-dependent vision metadata and multimodal text position metadata are
  explicit GPU request tensors. The native image/request frontend owns their
  construction because the reference implementation also precomputes these
  data-dependent loops before graph compilation.
- A cacheless fixed-shape Qwen3-VL-2B prefill graph serialized successfully on
  the RTX 4080. The observed monolithic plan was 9,757,430,436 bytes and
  TensorRT compilation peaked at approximately 21.56 GB host working set. This
  proves complete lowering, but the plan is not a production artifact: it was
  removed after measurement and must be replaced by partitioned and/or
  stripped-refittable plans.
- Stripped, individually refittable plans are now implemented and verified with
  a numerical named-weight fixture. Its plan is 18,724 bytes and refits directly
  from the mapped safetensors range. Rebuilding the full Qwen graph with the
  same policy reduced the plan to 13,172,492 bytes (about 741x smaller than the
  initial 9.76 GB plan).
- Cached engine deserialization now uses TensorRT `IStreamReaderV2` instead of
  first copying the complete plan into a host `std::vector`.
- The stripped full engine refits all 625 checkpoint tensors and executes to a
  finite GPU logits tensor of shape `[1, 16, 151936]`. Measured smoke timings on
  the RTX 4080 were approximately 3,592 ms for cold refit plus first forward and
  23.05 ms for a warm forward including explicit logits observation.
- A real `frame_0.jpg` smoke request now traverses native tokenizer, nvJPEG,
  CUDA resize/normalize/patch layout, asynchronous GPU FP32-to-BF16 conversion,
  native GPU vision interpolation/position metadata, native multimodal text
  MRoPE metadata, the compiled engine, and logits. For the intentionally tiny
  32x32 profile (four patches, one merged visual token), the latest cold
  frontend-to-logits run was 327.17 ms and steady-state was 29.37 ms. These are
  plumbing metrics, not semantic or HD performance claims: the profile is tiny
  and still performs no decode loop.
- `frontend="qwen3_vl"` now installs a model-owned compiled frontend. A caller
  can pass only `image_path`, `prompt`, and image-profile limits to
  `model.forward`; C++ constructs all eleven root graph inputs as GPU
  `X::Tensor` values and executes the cached engine. The one-call output is
  bit-identical to the expanded debug-input path.
- Greedy top-1 sampling now supports FP32 and BF16 logits directly on GPU and
  reads back only the selected token ID/value. On the same tiny warm profile,
  one-call JPEG plus prompt to the first GPU-sampled token measured 25.36 ms.
  The full-logits parity path measured 59.26 ms because it explicitly copied
  and compared the entire `[1,16,151936]` tensor.
- Native BF16 paged-KV write and grouped-query attention kernels now operate on
  GPU page arenas and an INT32 logical-to-physical page table. A model-scoped
  `debug_probe("paged_kv_bf16", ...)` fixture validates a five-token write/read
  against a NumPy numerical reference while keeping QKV, K pages, V pages, and
  attention output as GPU `X::Tensor` values. This is the backend primitive;
  the production decode graph still needs explicit cache, block-table, and
  context-length operands plus prefill/decode partition execution.
- Captured TensorRT networks are now strongly typed, preserving the dtype of
  each graph `X::Tensor` and preventing implicit FP32 conversion at custom
  boundaries. A serializable `GarnetPagedKVDecodeBF16` plugin consumes explicit
  QKV, key-page, value-page, page-table, context-length, and slot-position
  tensor edges from `.x`; no opaque cache handle appears in the graph.
- The actual 28-layer Qwen text-decode `.x` entry compiles against all real 2B
  weights into a 5,820,116-byte stripped/refittable engine. Cache-hit load was
  76.78 ms, first refit/forward was 1,785.73 ms, warm GPU-sampled decode was
  7.52 ms, and a persistent 16-token page averaged 8.11 ms/token with 8.51 ms
  p95. This is a batch-1, one-page decode profile; it does not include visual
  prefill or continuous batching and must not be presented as final serving
  throughput.
- `GarnetPagedKVPrefillWriteBF16` is now a separate serializable TensorRT
  plugin. It accepts explicit QKV, layer-page, page-table, and device
  start-position tensors, writes every prompt K/V row, and passes QKV onward
  to the ordinary full causal attention graph. A five-token fixture crosses a
  page boundary with a non-identity physical page table; a separately cached
  decode engine consumes all six logical positions with numerical parity.
- `qwen_text_prefill.x` and `qwen_vl_prefill.x` now define real paged prefill
  graphs. The latter contains vision encoding, visual embedding replacement,
  DeepStack injection, all 28 text layers, per-layer BF16 page writes, and the
  tied LM head. Its two-page test writes 16 multimodal tokens and hands the
  same GPU `X::Tensor` pages to token-17 decode. Measured warm timings were
  12.03 ms for VLM prefill and 8.87 ms for decode; cold refit was about 3.29 s
  and 2.00 s respectively.
- A 15-input `frontend="qwen3_vl"` profile now allocates model-owned BF16 KV
  pages on CUDA and initializes an internal compiled text-decode runtime. One
  public `model.forward({image_path, prompt, max_new_tokens})` call performs
  nvJPEG/CUDA preprocessing, native Qwen tokenization, VLM prefill, GPU greedy
  sampling, paged decode, stop-token handling, and native detokenization. On
  the one-visual-token plumbing profile, a warm JPEG plus 14-token prompt plus
  four generated tokens took 48.44 ms total after the packed-cache alias fix.
  The output was `This image shows a`; its first three token decisions exactly
  matched cacheless full-sequence execution. This proves ownership, page
  persistence, and execution-flow parity, not semantic image quality at a
  useful visual resolution.
- Packed per-layer cache selection is metadata, not a TensorRT slice. The
  prefill/decode plugins offset directly into the model-owned
  `[layers,pages,page_size,kv_heads,head_dim]` allocation. The earlier slice
  lowering mutated temporary TensorRT buffers and produced repetitive decode;
  a cacheless-token regression now prevents that failure from returning.
- Vision learned-position interpolation now reduces the four bilinear
  neighbors on axis 1, producing `[patches, hidden]`. The old axis-0 reduction
  accidentally worked only when `patches == 4` and failed at realistic grids.
- The 60-visual-token reference profile now compiles and runs with a 192x320
  resize, 240 vision patches, a 96-token text profile, and ten generated tokens.
  Four distinct JPEGs measured 122.08-123.72 ms end to end after warmup. For
  `frame_0.jpg`, Garnet returned
  `A person is weaving a basket in a traditional,`, consistent with the visible
  basket-weaving scene. These are batch-1 RTX 4080 measurements; formal HF
  logits parity and larger visual profiles remain required quality gates.
- The native single-image chat template now matches the official Qwen template
  token for token: user text directly follows `<|vision_end|>`, and
  `<|im_end|>` directly follows user text. The 60-visual-token sentence prompt
  is 77 tokens in both Garnet and HF. HF chose
  `A man sits in a rustic, bamboo-flo`; Garnet chose
  `A person is weaving a basket in a traditional setting`. Both are consistent
  with the frame, but token-level generation diverges after the shared first
  token. GPU preprocessing currently measures 0.0384 mean absolute pixel error
  versus the HF processor, so image preprocessing and full-logit numerical
  parity remain explicit follow-up work.
- The same 60-visual-token Qwen VLM graph now selects the preferred vision
  boundary and physically compiles a multi-engine plan from the captured `.x`
  operation DAG. Tensor-valued keyword operands are first-class DAG
  dependencies, which is required for `input_ids`, attention metadata, and
  other non-positional graph edges to cross partitions correctly. A focused
  fixture guards this contract.
- The schema-clean partitioned Qwen compile took 142.70 s on the RTX 4080. A clean
  process then loaded the persisted graph and engines in 229.31 ms, registered
  Garnet paged-KV plugin creators before partition deserialization, and ran the
  warm JPEG-to-ten-token path in 120.97-123.06 ms across four frames. The first
  post-load request remained a 5.16 s refit/cold-execution cost; eliminating or
  amortizing that cost remains production work.
- Serving readiness now eagerly deserializes, refits, and creates execution
  contexts for every prefill partition and the decode engine. On a clean cached
  process this moved refit into explicit model startup. Readiness also loads the
  native tokenizer instead of parsing it on the first prompt. Per-engine load
  locks allow independent prefill partitions to prepare concurrently. Decode
  initialization remains on the xlang calling thread so cache misses can safely
  capture its graph. A clean 60-token cached process loaded in 4.64 s, including
  4.30 s engine preparation and 178 ms frontend preparation. The first JPEG
  request fell from 5.16 s to 160.87 ms, versus approximately 122 ms warm.
  Runtime status reports `engines_prepared`, `engine_prepare_ms`,
  `frontend_prepared`, and `frontend_prepare_ms`; the real VLM test rejects a
  ready model that performs lazy engine preparation during its first request.
- The balanced production profile now has a dedicated regression using the
  original 1920x1080 JPEG, Qwen smart resize to 1312x736, 3,772 vision patches,
  943 merged visual tokens, a 1,024-token text profile, and 80 KV pages. The
  1,280-token KV capacity covers the 960-token prompt plus a 100-token output
  cap. Captured
  graph compilation accepts a fingerprinted `builder_workspace_mb` option; this
  profile uses 4 GB because its vision-merger tactic requires more than 1 GB,
  and the independent one-token decode compiler inherits the same explicit
  workspace limit unless `GARNET_DECODE_WORKSPACE_MB` overrides it.
- On the RTX 4080, the cached balanced image-to-first-token request took
  302.63 ms. Ten generated tokens took 945.89-1,014.89 ms across four images.
  Profiled prefill partitions were 206.36 ms for vision, 65.88 ms for the main
  text region, and about 5.65 ms for the remaining boundaries. The result misses
  the current sub-700-ms balanced target because decode at roughly 960-token
  context is about 73 ms/token, not because JPEG/vision prefill exceeds target.
  Example output for `frame_0.jpg` was
  `A man sits in a dimly lit, rustic` (truncated at ten tokens).
- Matched post-warmup four-image runs separated one-token and ten-token costs.
  Mean image-to-first-token latency was 305.58 ms and mean ten-token latency was
  955.44 ms. The vision partition averaged 205.54 ms, text prefill averaged
  73.32 ms, and the additional nine decode iterations averaged 649.86 ms total
  (72.21 ms/token). Therefore image/vision dominates first-token latency, but
  text decode is about 68% of the ten-token request and is the primary next
  optimization target.
- The BF16 paged-attention decode kernel now computes every Q-K score once per
  layer and keeps its softmax weights in shared memory. The prior kernel
  recomputed each score for max, sum, and every one of 128 output dimensions.
  On the same RTX 4080 balanced profile this raised sustained decode from about
  13.8 tokens/s to roughly 50 tokens/s. The four-image performance regression
  uses `ignore_eos=1` and generates exactly 100 tokens so different EOS points
  cannot bias throughput. Its measured totals were 2,270-2,543 ms, with
  327-338 ms TTFT and 44.9-51.1 tokens/s decode; averages were 2,345 ms total,
  333 ms TTFT, and 2,012 ms for the remaining 99 iterations. Normal serving
  keeps `ignore_eos` disabled and stops naturally; the same images completed
  meaningful 38-56-token descriptions in about 0.97-1.32 s during the earlier
  semantic run. The runtime now reports `time_to_first_token_ms`, `decode_ms`,
  `decode_tokens_per_second`, and `total_ms`, and benchmark output always
  includes the actual generated count.
- Persistent GPU metadata tensors remove per-token allocation of token,
  position, context-length, and slot-position inputs. This cleans up the serving
  loop but did not materially change the 943-token benchmark. BF16-pair
  vectorization, a two-block-per-head attention split, and TensorRT decode CUDA
  Graph replay were each measured and rejected because they were neutral or
  slower on this profile. Future optimization should profile TensorRT decoder
  GEMMs and use a proper split-K paged-attention workspace before revisiting
  those approaches.
- Vision attention now lowers to TensorRT 10 `IAttention` in BF16 instead of
  casting Q/K/V to FP32 and materializing a `[16, 3772, 3772]` score tensor in
  every vision block. Garnet applies the Qwen scale `1/sqrt(head_dim)` to Q as
  strongly typed BF16 weights before fused attention. On the balanced profile,
  the vision partition fell from approximately 209 ms to 50-53 ms. The exact
  100-token four-image run measured 2,002-2,026 ms total, 129-132 ms TTFT, and
  52.2-52.9 decode tokens/s; averages were approximately 2,015 ms total and
  130 ms TTFT. Semantic descriptions remained correct across all four images.
- Regular-resolution grounding exposed and fixed a vision metadata layout bug.
  The CUDA producer wrote bilinear interpolation indices and weights in
  `[4, patch_count]` order while the captured TensorRT graph consumed
  `[patch_count, 4]`. Four-patch fixtures could not detect the mismatch because
  both dimensions were four. The producer now emits interleaved per-patch
  metadata matching the graph profile. At 3,772 patches, the canonical `.x`
  vision pooler reaches 0.9962 cosine similarity with HF, and person boxes on
  four 1920x1080 frames visually align with the intended subjects. With EOS
  enabled and a 100-token ceiling, those requests measured 1.04-1.36 s total,
  131-152 ms TTFT, and 49.4-51.5 decode tokens/s.
- The optimized paged-KV decode path now uses TensorRT plugin workspace for
  split-K score evaluation and split-value aggregation. Score tiles raise Q-K
  parallelism from 16 blocks to roughly 128-144 blocks at this profile. Value
  tiles use 128 context positions per partial, followed by a small GPU
  reduction, instead of assigning the complete context to only 16 blocks.
  Persistent GPU logits and sampling buffers remove per-token allocation, and
  per-binding CUDA Graph replay starts after one warm decode and automatically
  recaptures when request GPU addresses change. The graph remains generated by
  the canonical `.x` model and executed by the compiled TensorRT runtime.
- On the RTX 4080, the unchanged regular-resolution person-detection prompt
  with 943 visual tokens and 1,013 total prompt tokens now sustains
  127.8-128.2 decode tokens/s. Four EOS-terminated JSON responses generated
  51, 61, 48, and 48 tokens in 520.49, 599.17, 496.76, and 495.10 ms total.
  TTFT was 127.94-129.20 ms. Descriptions and person boxes remained correct.
  The matched legacy-attention A/B on the same engine measured about
  51.6-51.9 tokens/s; split-K alone reached about 88 tokens/s, CUDA Graph replay
  reached about 92-93 tokens/s, and split-value aggregation produced the final
  128-token/s result. `GARNET_PAGED_KV_SPLIT_K=0`,
  `GARNET_PAGED_KV_SPLIT_VALUE=0`, and `GARNET_DECODE_CUDA_GRAPH=0` provide
  explicit diagnostic opt-outs. `GARNET_PROFILE_DECODE_LAYERS=<path>` writes a
  one-shot TensorRT layer profile without changing normal serving behavior.
- TensorRT builder optimization level is now a fingerprinted compile option
  (`builder_optimization_level`, range 0-5), with an independent
  `GARNET_DECODE_OPTIMIZATION_LEVEL` override. Level 5 took about 6.4 minutes to
  build and was performance-neutral at roughly 51.7 tokens/s on the legacy
  attention A/B, so production keeps TensorRT's level-3 default.
- A matched Hugging Face Transformers 5.13 / PyTorch 2.11 BF16-SDPA run on the
  same RTX 4080, images, 960-token prompt, and forced 100-token output averaged
  2,890 ms total, 214 ms TTFT, and 37.0 decode tokens/s. The current Garnet path
  is therefore about 1.43x faster end to end, 1.65x faster to first token, and
  1.42x faster in decode than that plain HF baseline.
- TensorRT fused causal text attention was evaluated and rejected. Both native
  unequal-head GQA and explicit Qwen KV-head repetition passed small fixtures
  but produced incorrect balanced-profile descriptions. Production keeps the
  semantically verified FP32 masked text-prefill lowering, currently about
  66-73 ms, until a padded-profile logits-parity test supports a replacement.
- Decode graph capture must execute on the xlang calling thread when its cache
  is missing. A prior attempt to initialize it asynchronously deadlocked on a
  new 64-page profile because xlang runtime capture is not a background-thread
  operation. Concurrency is limited to already-built TensorRT engine
  deserialization/refit, where per-engine locks are safe.
- The runtime cache schema now includes the strongly typed/paged-KV compiler
  ABI. A clean rebuild of the complete 625-weight VLM root succeeded under that
  schema and produced an 11,393,228-byte stripped engine. The rebuilt one-call
  tiny-profile JPEG path returned its first GPU-sampled token in 16.77 ms;
  explicit full-logits observation took about 120 ms and is a debug boundary,
  not the serving path.
- The original `qwen_vl_model.x` remains the cacheless correctness graph.
  Production generation uses the explicit paged `qwen_vl_prefill.x` plus
  `qwen_text_decode.x` partition; neither graph uses opaque cache handles.
- Multi-result model operations are represented as explicit single-output
  expressions or packed tensor state. Tensor expressions are never indexed as
  dictionaries; this is enforced by the root-capture regression test.
- The cache-hit model executes the same engine and result without executing the
  root `.x` again.
- `QwenTextRunner`, `QwenVisionRunner`, and the unused alternate
  `Model::BuildTRTEngine -> TRTBuilder::BuildEngine` path have been removed from
  this branch.

Verified tests:

```text
Release build: build_scripts/windows/build_release.bat garnet
Compiled path: test2026/tests/phase_24_compiled_xmodel_runtime/test.py
Qwen root: test2026/tests/phase_24_compiled_xmodel_runtime/test_qwen_root_capture.py
Qwen text prefill: test2026/tests/phase_24_compiled_xmodel_runtime/test_qwen_prefill.py
Qwen VLM prefill/decode: test2026/tests/phase_24_compiled_xmodel_runtime/test_qwen_vl_prefill.py
Native one-call generation: test2026/tests/phase_24_compiled_xmodel_runtime/test_qwen_vl_generate.py
Fusion/partition planner: test2026/tests/phase_24_compiled_xmodel_runtime/test_fusion_annotations.py
Partitioned 60-token VLM: test2026/tests/phase_24_compiled_xmodel_runtime/test_qwen_vl_generate_60_visual.py
Balanced 943-token VLM: test2026/tests/phase_24_compiled_xmodel_runtime/test_qwen_vl_generate_943_visual.py
Reference path: RUN_GARNET_TRT_PREFLIGHT=1 phase_00_trt_expression_preflight/test.py
```

This completes the core Stage 2 gates, most fixed-shape Stage 3 lowering
mechanics, explicit batch-1 paged prefill/decode, and the first model-owned
one-call Stage 9 path. It does not complete dynamic profiles, semantic VLM
parity, prefix persistence, continuous batching, or production-resolution
image gates. Next work is HF intermediate parity at useful visual profiles,
persistent decode buffers/CUDA Graphs, and scheduler-owned page allocation.

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

- Extend `@T.fusion(...)` capture so `name`, `role`, `boundary`, `atomic`, and
  `cuda_graph` keyword arguments are validated and retained on expanded graph
  regions. Do not add a separate `T.stage` language construct.
- Preserve nested-function and loop-instance scope identity so repeated atomic
  layers are individually visible to the partition planner.
- Reject conflicting annotations, including a required boundary inside an
  atomic parent region.
- Partition captured graphs into TensorRT regions and custom CUDA nodes.
- Combine annotation constraints with operator support, shape profiles, build
  memory, runtime memory, launch cost, invocation frequency, and placement cost.
- Group repeated atomic layers automatically; do not hardcode Qwen model names,
  layer counts, or layer-group sizes in C++.
- Calculate semantic fingerprints per partition.
- Compile and atomically store rank/device-specific engines.
- Reuse unchanged partitions after a local `.x` change.
- Select dynamic profiles for vision, prefill, and decode.
- Emit a generated execution manifest with GPU tensor bindings, streams,
  events, custom operations, and CUDA Graph eligibility.

### Exit Tests

- `T.fusion` keyword metadata survives capture, graph-cache serialization, and
  graph-cache reload without executing `.x` on the cache-hit path.
- Required boundaries always partition; preferred boundaries may merge under a
  deterministic cost configuration; atomic regions are never split.
- A five-layer loop annotated with an atomic layer function partitions only
  between complete layer instances.
- Invalid and conflicting fusion parameters fail with source path and expanded
  graph-region diagnostics.
- Changing one partition rebuilds only that engine.
- Prompt/image contents and runtime batch values within profiles do not rebuild.
- Shape/profile, precision, weight, plugin, or engine-boundary changes rebuild
  the affected engine.
- Concurrent cache loaders never observe partial artifacts.
- A two-engine fixture passes an intermediate GPU `X::Tensor` directly between
  engines using stream/event dependencies, with no host tensor copy or device-
  wide synchronization.

## Stage 5: Compile-Time VLM Flow Placement

### Deliverables

- Identify vision, multimodal prefill, and decode invocation regions from graph
  dependencies and compiler annotations, not model-specific C++ ordering.
- Estimate compute, memory, transfer, synchronization, and invocation-frequency
  costs.
- Emit fixed engine/device placement and explicit transfer nodes.
- Keep decode-loop communication strongly penalized.
- Build runtime launch DAGs from the generated execution manifest. The C++
  scheduler submits request and continuous-batch work; CUDA streams/events and
  optional CUDA Graph buckets execute dependencies without CPU tensor access.

### Exit Tests

- Colocated and split vision/text layouts produce matching results.
- Each engine runs only on its compiled device.
- Only declared boundary tensors cross devices.
- Runtime cannot dynamically move engines or change the partition.
- Multi-engine execution performs zero intermediate D2H copies and zero
  `cudaDeviceSynchronize` calls in the production request path.

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
