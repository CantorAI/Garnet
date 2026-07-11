# Qwen3-VL Production One-Call Generate and KV Cache Plan

## Goal

Garnet serving must expose one production request boundary:

```text
JPEG/raw image + prompt -> Garnet generate -> text output
```

Callers must not manually call tokenizer, image preprocess, visual tower, text prefill, or decode subgraphs. Those are internal stages of one request.

## Public API Shape

Initial C++/xlang API:

```text
garnet.qwen_vl_generate(
    model,
    image_path="...",
    prompt="Describe this picture.",
    max_new_tokens=64,
    min_pixels=65536,
    max_pixels=65536,
    request_id=optional,
    cache_policy="request"
)
```

The return value should include:

```text
{
  text,
  token_ids,
  finish_reason,
  timings,
  source_image_size,
  resized_image_size,
  image_grid_thw,
  prompt_tokens,
  generated_tokens
}
```

## Internal Request Pipeline

One request owns these stages:

1. Decode JPEG by nvJPEG or accept caller-provided GPU/raw image.
2. Resize, normalize, and patch-pack with CUDA/NPP kernels.
3. Tokenize prompt with native Qwen tokenizer.
4. Build multimodal sequence with image placeholders and `mm_token_type_ids`.
5. Run vision tower once.
6. Replace visual placeholders with visual embeddings.
7. Run text prefill once and write KV pages.
8. Decode one token at a time using KV cache.
9. Detokenize generated ids.

The test harness now caches visual tokens once. Production C++ must move that ownership into the request object.

## Tensor Ownership Rule

`X::Tensor` is the only tensor object used by the production graph. GPU memory is represented by the tensor descriptor:

```text
X::Tensor
  shape
  dtype
  optional CPU/debug storage
  TensorDescriptor.gpuMemory = CUDA device address
```

After image preprocessing and weight loading, all layer inputs, intermediate activations, QKV, RoPE outputs, attention outputs, MLP outputs, logits, and KV page arenas must stay as GPU-backed `X::Tensor` values. CPU readback is allowed only at explicit boundaries:

- final sampled token ids / decoded text
- debug snapshots
- parity tests

The C ABI helpers are preflight/test compatibility APIs. They may accept raw host pointers because Python tests use them, but the production xlang/Garnet flow must use `X::Tensor` inputs and outputs. No new parallel "device tensor handle" abstraction should be introduced.

Current bridge work:

- `TRTBuilder` now checks `TensorDescriptor.gpuMemory` for inputs and attaches output GPU memory back to returned `X::Tensor` values.
- Device-binding coverage now includes matmul, RMSNorm, LayerNorm, QKV projection, QKV head norm, RoPE, text attention, vision attention, linear transpose (`o_proj`/LM head), linear+bias transpose, TextMLP, and VisionMLP.
- `LoadModelFromFile` moves loaded tensors to GPU immediately so model weights are GPU-resident before forward.
- `qwen_vl_preprocess_jpeg_file` uses nvJPEG decode -> CUDA resize/normalize/patch-pack -> GPU-backed `pixel_values` `X::Tensor`.
- Garnet exposes xlang-callable `device_paged_kv_write(handle, qkv_tensor, token_count, start_position)` and `device_paged_kv_attention(handle, q_or_qkv_tensor, sequence_length)` so KV write/read can consume GPU-backed `X::Tensor` values directly.
- `GARNET_TRT_SYNC_CPU_OUTPUTS=0` disables debug CPU synchronization for TRT outputs. Existing Python parity harnesses leave it enabled because they call `numpy()` between subgraphs.

## Cache Objects

### Request Context

```text
QwenVLRequestContext
  request_id
  model_id
  prompt_text
  input_ids
  mm_token_type_ids
  image_grid_thw
  pixel_values
  visual_embeddings
  kv_sequence
  generated_ids
  timings
```

`visual_embeddings` are request-local for normal image QA. They can become shared later only when the same image embedding is reused across prompts.

### KV Cache Manager

KV cache must be owned by a long-lived model/server object, not by the Python test loop.

```text
PagedKVCacheManager
  page_size_tokens
  num_layers
  num_kv_heads
  head_dim
  dtype
  device_id
  free_pages
  active_sequences
```

Each active sequence stores:

```text
KVSequence
  sequence_id
  pages_per_layer
  logical_length
  max_length
  last_access_time
  ref_count
```

## Prefill and Decode Contract

Prefill:

```text
prefill(input_embeddings, position_ids, mm_token_type_ids, kv_sequence)
```

Writes K/V for all prompt tokens, including visual tokens after visual tower replacement. Returns logits for the final prompt token.

Decode:

```text
decode(next_token_id, position_id, kv_sequence)
```

Reads previous K/V pages and appends the new token K/V. It must not rerun:

- JPEG decode
- image preprocessing
- vision tower
- prompt tokenization
- full prompt text layers

## Scheduler Requirements

The scheduler should batch requests at two levels:

1. Prefill batch: heterogeneous prompt/image sizes, less frequent.
2. Decode batch: one token per active request, frequent and latency-sensitive.

Required scheduler metadata:

```text
RequestState = WaitingPrefill | Prefilling | Decoding | Finished | Failed
priority
arrival_time
deadline_ms
max_new_tokens
kv_pages_reserved
```

For the MVP, a single-GPU FIFO scheduler is enough. The interface should still keep request state explicit so we can later add continuous batching.

## Memory Policy

Default policy:

- Visual embeddings: request-local, freed after request.
- Pixel values: freed immediately after vision tower finishes.
- KV pages: held until request finish, then returned to free list.
- Tokenizer cache: process-level per model directory.
- TRT engines/kernels: process-level per model, shape bucket, and subgraph.

Future reuse policy:

- System prompt prefix KV can be shared when the exact prefix tokens and position policy match.
- Visual embeddings can be reused when the exact image preprocessing fingerprint matches.
- Full multimodal prefix KV can be reused only when prompt prefix, image grid, visual embeddings, and positions match.

## Metrics Required

Every generate call must report:

```text
jpeg_decode_ms
image_preprocess_ms
tokenize_ms
vision_prefill_ms
text_prefill_ms
first_token_ms
decode_total_ms
decode_avg_ms
total_ms
kv_pages_used
peak_gpu_memory_bytes
```

Current known state:

- Native tokenizer hot Qwen-VL prompt build: about 2 ms.
- Native JPEG/CUDA preprocess is validated and no longer the major cost:
  - `frame_0.jpg` 1920x1080 -> 320x192, `pixel_values [240,1536]`: mean 4.529 ms, min 3.986 ms, max 6.080 ms, 50 iterations, 5 warmup discarded.
  - `frame_0.jpg` 1920x1080 -> 1312x736, `pixel_values [3772,1536]`: mean 5.095 ms, min 4.479 ms, max 6.928 ms, 50 iterations, 5 warmup discarded.
  - Command shape: `GARNET_TRT_SYNC_CPU_OUTPUTS=0 out/build/x64-Debug/bin/garnet_qwen_vl_gpu_path_perf.exe <jpg> 50 65536 1003520 <garnet.dll> 5`.
- Native one-call frontend is now validated without Python runtime:
  - `GarnetQwenVLPrepareJpegPromptDevice` performs native cached tokenizer + nvJPEG/CUDA preprocess and returns GPU `pixel_values` device memory plus prompt IDs/mm token types.
- `GarnetQwenVLPrepareJpegPromptDeviceTensors` returns GPU device memory for `input_ids`, `mm_token_type_ids`, and `pixel_values`, so the next Garnet model stage can consume the prepared request without a new CPU tensor boundary.
- `GarnetCreateQwenVLDeviceRequest` / `GarnetGetQwenVLDeviceRequestInfo` / `GarnetDestroyQwenVLDeviceRequest` move those GPU buffers under a Garnet-owned request handle. This is request lifetime ownership, not a parallel tensor abstraction; the next model runner should consume this request and eventually replace the C ABI handle with the xlang `QwenVLRequestContext` object.
- `GarnetAllocateQwenVLDeviceRequestKV`, `GarnetGetQwenVLDeviceRequestKVInfo`, and `GarnetGetQwenVLDeviceRequestKVState` now let the native request handle own its device KV cache and report both capacity and written logical length. Request destroy releases the image/prompt buffers and its KV handle together, so C++ serving can treat the request as one lifetime owner.
- `GarnetQwenVLDeviceRequestKVWriteDevice` and `GarnetQwenVLDeviceRequestKVAttentionDevice` now write/read the request-owned KV cache by request handle. KV writes advance request logical length; attention rejects reads beyond that written length. The normal native serving path no longer needs to fetch the raw KV handle for KV operations.
- `garnet.qwen_vl_create_request(...)` now returns a `QwenVLRequestContext` xlang object carrying GPU-backed `X::Tensor` values for `input_ids`, `mm_token_type_ids`, `pixel_values`, and `image_grid_thw`. This is the intended production bridge into `model.forward(...)` and TensorRT subgraphs.
- `model.forward_request(request)` now accepts a `QwenVLRequestContext` or compatible request object, validates that the request tensors are GPU-backed, and returns tensor residency/shape metadata. When the model is configured for `vision_patch_embed`, it can consume `request.pixel_values` directly and run the first visual TRT subgraph without a CPU tensor boundary. This is the first model-owned bridge from prepared request tensors into Garnet execution; the full Qwen-VL runner still needs to expand this pattern across vision tower, multimodal merge, text prefill, decode, and LM head.
- `model.tokenizer(text)` no longer calls Python/Hugging Face `AutoTokenizer`. It uses Garnet's cached native Qwen tokenizer and returns GPU-backed `INT64` `input_ids` and `attention_mask` tensors.
- `model.detokenizer(token_ids, skip_special_tokens=True)` now uses Garnet's cached native Qwen tokenizer to decode generated token IDs. If token IDs are still GPU-backed, it performs the small final-boundary device-to-host copy explicitly before decoding. This keeps tokenizer/detokenizer ownership inside Garnet and avoids Python/Hugging Face in the runtime path.
- `model.debug_probe("logits_top1", logits_tensor)` now performs greedy top-1 sampling from a GPU-backed FLOAT32 logits `X::Tensor` for tests/debug metadata. It is intentionally not a product serving API; production decode should keep sampling as an internal model/generate step.
- `model.create_device_kv_cache(...)` and `model.destroy_device_kv_cache(handle)` now expose GPU-resident paged KV allocation/release from the model object. This moves the KV lifecycle toward model/request ownership instead of keeping it only as standalone test C ABI calls.
- `QwenVLRequestContext` now carries request-owned KV metadata: `kv_cache`, `kv_handle`, `kv_max_tokens`, `kv_page_size`, `kv_logical_pages`, `kv_physical_pages`, `kv_q_heads`, `kv_heads`, and `kv_head_dim`.
- `model.forward_request(request, allocate_kv=True, max_new_tokens=..., page_size=..., q_heads=..., kv_heads=..., head_dim=...)` now allocates request-sized GPU KV pages after validating the GPU-backed request tensors, returns the `kv_cache` dictionary, and writes the KV handle/geometry back onto the request object when it is a packaged `QwenVLRequestContext`. Capacity is derived from `prompt_tokens + max_new_tokens`; the next runner step must write prefill/decode K/V into this handle.
- Device-pointer KV exports now exist next to the host test ABI: `GarnetDevicePagedKVWriteDeviceFP32(...)` and `GarnetDevicePagedKVAttentionDeviceFP32(...)`. These consume CUDA device pointers directly and avoid staging QKV/Q/output through host memory.
- `model.write_device_kv_cache(handle_or_cache_or_request, qkv_tensor, token_count, start_position)` and `model.attention_device_kv_cache(handle_or_cache_or_request, q_tensor, sequence_length, q_width=...)` now accept a raw handle, a KV cache dictionary, or a `QwenVLRequestContext`. Writes update `logical_length` / `kv_logical_length`; attention rejects reads past that logical length when it is available.
- `model.forward(...)` now has a direct TensorRT-output-to-KV bridge for the `text_rope_apply` subgraph: when called with `kv_handle`, `kv_cache`, or `request`, it runs `RunTextRoPEEngine`, keeps the RoPE-applied QKV as a GPU-backed `X::Tensor`, then calls `GarnetDevicePagedKVWriteDeviceFP32(...)` on that device pointer. Without KV owner metadata, the method preserves the old behavior and returns the tensor directly.
- `model.forward(...)` now also has a device-KV read bridge for the `text_attention_core` subgraph: when called with `kv_handle`, `kv_cache`, or `request`, it reads the last-token Q vector from the GPU-backed QKV/RoPE tensor, checks request/cache logical length when present, calls `GarnetDevicePagedKVAttentionDeviceFP32(...)`, and returns a GPU-backed attention output `X::Tensor`. Without KV owner metadata, it preserves the old full attention path.
  - `frame_0.jpg` 1920x1080 -> 1312x736, prompt `"Describe the image and list visible objects with short coordinates."`: `prompt_tokens=965`, `visual_tokens=943`, `pixel_values_count=5793792`, `pixel_values_bytes_device=23175168`.
  - All-device steady result: wall mean 11.376 ms, min 10.421 ms, max 12.636 ms; image mean 5.308 ms; tokenize mean 5.530 ms; token tensor upload mean 0.523 ms; `input_ids_bytes_device=7720`, `mm_token_type_ids_bytes_device=7720`; 50 iterations, 5 warmup discarded.
  - Latest Garnet-owned request handle steady result: wall mean 12.483 ms, min 11.280 ms, max 18.426 ms; API total mean 12.459 ms; image mean 5.575 ms; tokenize mean 6.279 ms; token tensor upload mean 0.601 ms; 50 iterations, 5 warmup discarded. This measurement includes create/get-info/destroy lifecycle.
  - Command shape: `GARNET_TRT_SYNC_CPU_OUTPUTS=0 out/build/x64-Debug/bin/garnet_qwen_vl_prepare_prompt_perf.exe <model_dir> <jpg> <prompt> 50 65536 1003520 <garnet.dll> 5`.
- Native device-KV smoke now validates create -> write -> attention -> destroy without Python runtime:
  - command: `out/build/x64-Debug/bin/garnet_device_kv_native_smoke.exe <garnet.dll>`
  - sequence length: 17, page size: 8, logical pages: 3, Q heads: 4, K/V heads: 2, head dim: 16.
  - host ABI output absolute sum: `2.31298`; output max absolute value: `0.0423`.
  - device-pointer ABI output absolute sum: `2.31298`; output max absolute value: `0.0423`; max diff versus host ABI: `0`.
- Native tokenizer smoke validates Garnet-owned final text conversion without Python runtime:
  - command: `out/build/x64-Debug/bin/garnet_qwen_tokenizer_native_smoke.exe <model_dir> <garnet.dll>`
  - encode/decode round trip: `"Describe visible objects."` -> 4 tokens -> `"Describe visible objects."`
  - Qwen-VL image placeholder token was found: `<|image_pad|> = 151655`.
- Native debug logits sampler smoke validates the internal GPU metadata/test hook without making it the serving API:
  - xlang-facing debug hook: `model.debug_probe("logits_top1", logits_tensor)`.
  - debug/test hook: `GarnetDebugSampleLogitsTop1FP32(...)`.
  - command: `out/build/x64-Debug/bin/garnet_debug_logits_sampler_native_smoke.exe <garnet.dll>`
  - validated last-row greedy argmax on GPU: `rows=3`, `vocab_size=11`, `token_id=7`, `token_value=42.5`.
- Native request+KV smoke now validates a Garnet-owned Qwen-VL GPU request and device KV handle in one C++ executable without Python runtime:
  - command: `out/build/x64-Debug/bin/garnet_qwen_vl_request_kv_native_smoke.exe <model_dir> <jpg> <prompt> 65536 1003520 <garnet.dll>`
  - request output pointers are all device pointers: `input_ids`, `mm_token_type_ids`, and `pixel_values`.
  - same `frame_0.jpg` path produced `prompt_tokens=965`, `visual_tokens=943`, `pixel_values_bytes_device=23175168`.
  - request-owned device KV was allocated for `prompt_tokens + 32 = 997` tokens with page size 16 (`kv_logical_pages=63`), started with logical length 0, wrote device QKV for a 64-token smoke window by request handle, advanced logical length to 64, rejected an intentional 65-token over-read, read valid device attention by request handle, and returned nonzero output (`kv_output_abs_sum=0.244571`).
  - The one-shot smoke intentionally includes cold tokenizer/model file cache cost (`request_total_ms=1898.4 ms` in the latest run). Use the warmed request perf gate above for steady frontend serving latency.
- `phase_22_model_device_kv_bridge` validates the model-owned GPU bridge without copying intermediate tensors back to NumPy:
  - `text_rope_apply` runs TensorRT, returns GPU-backed `X::Tensor`, writes that tensor's device pointer into model-created paged KV, and updates `kv_logical_length`.
  - `text_attention_core` reads the same device KV via model.forward and returns a GPU-backed attention output tensor.
  - The debug probe samples that GPU output tensor to prove it remains usable through xlang as a GPU `X::Tensor`.
- Current slow path is model orchestration: repeated small subgraph calls and incomplete production one-call generate. The next stage should collapse the per-layer loop into a clean Garnet request path using GPU `X::Tensor` inputs/outputs and device KV pages.

### GPU X::Tensor Chain Checkpoint

The cached text path now supports an explicit GPU tensor chain used by the Python test driver:

- `tensor_to_gpu(tensor)` promotes an `X::Tensor` once and preserves its CUDA address for the tensor lifetime.
- TRT input binding now promotes CPU-backed tensors through `X::Tensor` ownership instead of allocating and freeing a temporary CUDA weight buffer on every forward.
- Prefill and decode reuse one GPU weight tensor per model weight across shape-specific TensorRT engines.
- `embedding`, `replace_rows_by_mask`, `tensor_add`, and `tensor_last_row` keep embedding lookup, visual-token splice, residual connections, and final-token selection on GPU.
- Norm, QKV, RoPE, KV write/read, attention, output projection, MLP, LM head, and greedy sampling pass GPU-backed `X::Tensor` values directly between stages.
- Python remains the temporary layer-loop test driver; it no longer reads intermediate decoder tensors. Only sampled token IDs cross to CPU for detokenization and test reporting.

Verified full-depth Qwen3-VL-2B run on `Dataset.1980Love/imgs/frame_0.jpg`, 79 prompt tokens, 60 visual tokens, 28 text layers, four generated tokens:

```text
output: A man sits in
visual tower once: 4019.30 ms
text prefill: 5056.52 ms
decode step 1 (decode model/weight warm-up): 643.10 ms
decode step 2: 152.35 ms
decode step 3: 125.07 ms
decode total for three appended tokens: 921.85 ms
total after frontend: 9997.67 ms
```

The previous device-KV harness averaged about 5.17 seconds per appended token. At this checkpoint the two steady decode steps averaged 138.71 ms/token, about 37x faster. Model and shape initialization still occurred inside the measured request, the visual path still used Python/NumPy orchestration between visual subgraphs, and the layer loop had not yet moved into the model-owned C++ runner. The following checkpoint resolves the text-layer-loop item.

### Model-Owned C++ Text Runner Checkpoint

`QwenTextRunner` now owns the text-layer execution loop in C++:

```text
Python/xlang request driver
  -> QwenTextRunner.prefill(...) once
     -> C++ loop over 28 layer bundles
  -> QwenTextRunner.decode(...) once per generated token
     -> C++ loop over 28 layer bundles
```

Each layer bundle retains its input norm, QKV, RoPE, attention, output projection, post-attention norm, and MLP model objects. It also shares the same persistent GPU weight tensors between prefill and one-token decode shape buckets. Residual additions run inside the C++ runner through GPU tensor operations. There is no Python loop over text layers and no intermediate decoder tensor readback.

Verified four-token full-depth result:

```text
output: A man sits in
gpu_tensor_chain: true
cpp_text_runner: true
visual tower once: 4024.39 ms
text prefill: 5119.54 ms
first decode step including one-time decode runner initialization: 620.70 ms
steady decode step 2: 117.71 ms
steady decode step 3: 102.76 ms
steady decode average: 110.24 ms/token
```

The next optimization boundary is no longer the text layer loop. It is:

1. Create both prefill and decode shape runners during long-lived model startup so the first request does not pay the 620 ms decode initialization.
2. Move the vision block loop and GELU/RoPE glue out of Python/NumPy into a model-owned GPU runner.
3. Expose the complete request lifecycle through the production one-call `generate` boundary.

### Model-Owned C++ Vision Runner Checkpoint

`QwenVisionRunner` now executes all 24 vision blocks and the patch merger in one C++ call. The active performance path keeps patch embeddings, QKV, rotary output, attention, residuals, MLP activations, and merged visual tokens as GPU-backed `X::Tensor` values.

New generic CUDA tensor operations:

- exact Qwen vision RoPE over packed `[tokens, Q|K|V]`
- tanh-GELU matching the Qwen/PyTorch formula
- GPU residual addition and merger reshape/copy

The vision runner owns layer and merger model bundles with persistent GPU weights. The merged `[visual_tokens, 2048]` output goes directly into `replace_rows_by_mask`; it is not converted to NumPy before text prefill.

Verified Qwen3-VL-2B result with both C++ runners enabled:

```text
output: A man sits in
cpp_vision_runner: true
cpp_text_runner: true
vision model prepare: 1865.71 ms (one-time)
vision warm-up: 131.06 ms (one-time)
vision patch/position input: 24.78 ms
24 vision blocks + merger: 106.56 ms
text model prepare: 4861.64 ms (one-time)
text prefill, 79 tokens: 950.10 ms
decode tokens: 84.45 ms, 64.48 ms, 67.09 ms
steady pipeline, image embeddings through three decode tokens: 1299.92 ms
```

Compared with the previous Python/NumPy vision loop at about 4.02 seconds, steady C++ vision execution is about 30.6x faster when including the patch stage (`131.34 ms`) and about 37.7x faster for the 24 blocks plus merger alone (`106.56 ms`). Model preparation is now explicitly separated from request execution and must move into long-lived server/model startup.

The dominant warm-path cost is now text prefill (`950 ms`). The next large optimization should fuse each text decoder layer into one TensorRT graph, retain a shared CUDA stream/workspace, and move the weight/activation path to FP16/BF16. Reworking small Python calls is no longer the useful target.

## Implementation Stages

## Current Checkpoint

Implemented now:

- `GarnetQwenVLPrepareJpegPrompt` C ABI front door:
  - Input: model dir, JPEG path, prompt, min/max pixels.
  - Output buffers: `input_ids`, `mm_token_type_ids`, `pixel_values`, `image_grid_thw`, source/resized sizes.
  - Verified on `Dataset.1980Love/imgs/frame_0.jpg`: 79 prompt tokens, 60 visual tokens, grid `[1, 12, 20]`, resized `192x320`.
- `garnet.qwen_vl_prepare_request(...)` xlang metadata front door:
  - Returns token lists, multimodal token type list, image grid, pixel shape/count, and timings.
  - Does not return xlang tensors yet because Python-side xlang tensor marshaling currently crashes for these tensors.
- Production-shaped `KVCacheManager`:
  - Allocates real GPU K/V arenas.
  - Tracks free/used pages.
  - Supports `allocate`, `append`, `free`, and `stats`.
  - Verified page growth and release in phase 13.
- Phase 08 now consumes the one-call native request prepare path:
  - `processor_npz` reports `garnet_one_call_prepare_jpeg_prompt`.
  - The old separated tokenizer/JPEG path remains available as a fallback.
- Phase 08 now owns a request KV lifecycle:
  - Allocates pages for prefill sequence length.
  - Appends one token per decode step.
  - Frees pages on request finish.
- Text decode cached-attention CUDA primitive:
  - `runTextKVCachedAttentionFP32`: one-token grouped-query attention over contiguous K/V.
  - `runTextPagedKVCachedAttentionFP32`: one-token grouped-query attention over paged K/V using a page table.
  - `runTextPagedKVWriteFP32`: writes post-RoPE K/V slices from QKV into paged cache pages.
  - C ABI test hooks validate both layouts against NumPy reference.
  - Phase 15 validates QKV -> paged K/V write -> paged K/V read attention against last-token full causal attention.
- Qwen decoder-layer cached last-token parity:
  - Phase 16 wires the paged KV write/read primitive into the real Qwen text decoder layer path.
  - It computes full-sequence decoder output as the reference, then recomputes only the last token using paged K/V attention plus `o_proj`, post-attention RMS norm, and MLP.
  - Verified with real Qwen3-VL-2B weights:
    - prompt window: 17 tokens
    - page size: 8
    - QKV shape: `[17, 4096]`
    - K/V page shape: `[5, 8, 8, 128]`
    - max error: `0.00011981`
    - mean error: `0.00002652`
  - Current limitation: phase 08 still uses the slow full-sequence decode loop. The next implementation step is a cached prefill/decode facade that uses this layer path across all text layers.
- Qwen cached prefill/decode facade parity:
  - Phase 17 splits text execution into:
    - prefill: run prompt tokens once and write per-layer paged K/V
    - decode: run one new token through layers using the stored K/V pages
  - The test compares cached decode against full recompute for the same prompt-plus-new-token sequence.
  - Verified with real Qwen3-VL-2B weights:
    - prompt tokens: 11
    - decoded token count: 1
    - layer count tested: 4 of 28
    - page size: 8
    - K/V page shape per layer: `[4, 8, 8, 128]`
    - hidden max error: `0.00121021`
    - logits max error: `0.00721788`
    - top-1 token match: yes
  - This is still a Python harness path that copies host K/V pages through C ABI hooks. Production serving must move the same lifecycle into C++ model-owned request state and keep pages on GPU.
- Prompt+image cached generation harness:
  - Phase 18 wires the cached prefill/decode facade into the native JPEG+prompt generation harness.
  - Verified full-depth Qwen3-VL-2B generation on `Dataset.1980Love/imgs/frame_0.jpg`:
    - input: JPEG + native Qwen tokenizer + nvJPEG/CUDA image preprocess
    - text layers: 28 of 28
    - generated tokens: `A man`
    - visual tokens once: `46582.29 ms`
    - text prefill: `172307.21 ms`
    - cached decode token loop: `118479.98 ms`
    - total after frontend: `337369.49 ms`
  - Same two-token run with the old full-sequence recompute path:
    - generated tokens: `A man`
    - visual tokens once: `46483.46 ms`
    - full recompute decode loop: `60247.09 ms`
    - total after frontend: `106730.55 ms`
- TRT execution-context cache:
  - `TRTBuilder` now caches deserialized TensorRT engines and execution contexts by engine path.
  - This removes repeated engine file reads, deserialization, and execution-context creation in the hot loop.
  - After the cache, full-depth Phase 18 improved to:
    - generated tokens: `A man`
    - visual tokens once: `38477.96 ms`
    - text prefill: `26245.87 ms`
    - cached decode token loop: `5773.79 ms`
    - total after frontend: `70497.62 ms`
  - Same two-token full-sequence recompute path after the cache:
    - generated tokens: `A man`
    - visual tokens once: `45852.10 ms`
    - full recompute decode loop: `59218.46 ms`
    - total after frontend: `105070.56 ms`
  - Conclusion: cached decode is now a real performance win in the harness, but it still copies tensors and K/V pages through host memory. The next implementation must keep K/V pages GPU-resident and cache layer engines inside a C++ model-owned runner.
- Device-resident paged KV handle:
  - Phase 19 adds a C ABI handle for GPU-resident FP32 paged K/V caches.
  - New API shape:
    - `GarnetCreateDevicePagedKVFP32(...)`
    - `GarnetDevicePagedKVWriteFP32(handle, qkv, token_count, start_position, ...)`
    - `GarnetDevicePagedKVAttentionFP32(handle, q, output, sequence_length, ...)`
    - `GarnetDestroyDevicePagedKVFP32(handle, ...)`
  - K/V page arenas and the page table are allocated once on GPU and reused across calls.
  - Phase 19 validates split writes plus attention against NumPy reference:
    - sequence length: 41
    - split write position: 29
    - page size: 8
    - Q heads: 16
    - K/V heads: 8
    - head dim: 128
    - max error: `0.00000003`
  - Current limitation: QKV and Q/output still cross the host C ABI boundary. The next target is passing device tensor pointers between TRT subgraphs and the device KV handle inside a C++ runner.
- Qwen text device-KV prefill/decode parity:
  - Phase 20 wires the device-resident KV handle into the real multi-layer Qwen cached decode harness.
  - Each text layer owns a GPU K/V handle; prefill writes prompt K/V into that handle, and decode appends/reads from it.
  - Verified against full recompute with real Qwen3-VL-2B weights:
    - prompt tokens: 11
    - decoded token count: 1
    - layer count tested: 4 of 28
    - page size: 8
    - hidden max error: `0.00122595`
    - logits max error: `0.00671178`
    - top-1 token match: yes
  - New bridge after Phase 20:
    - `QwenVLRequestContext` owns GPU KV metadata after `forward_request(..., allocate_kv=True)`, so later text prefill/decode stages can consume one request object instead of manually passing a detached handle.
    - `TRTBuilder` attaches TensorRT output device memory to returned `X::Tensor` values.
    - `model.forward(..., kv_handle=... / kv_cache=... / request=...)` for `text_rope_apply` writes the RoPE-applied QKV output directly into device KV by device pointer and updates request/cache logical length.
    - `model.forward(..., kv_handle=... / kv_cache=... / request=...)` for `text_attention_core` reads paged device KV and returns the last-token attention output as a GPU-backed `X::Tensor`.
    - Native device-KV smoke validates host ABI and device-pointer ABI parity: output absolute sum `2.31298`, max diff versus host ABI `0`.
    - Native request-KV smoke validates request-level logical length: initial logical length `0`, after device write `64`, and attention over-read `65` is rejected before the valid `64`-token attention read.
    - Phase 22 validates this same bridge from xlang `model.forward`: synthetic `[3,4096]` QKV writes three tokens into device KV, `text_attention_core` reads it back as GPU output, and `debug_probe("logits_top1")` runs on that output.
  - Completed: `QwenTextRunner` executes all 28 decoder layers in C++ and keeps attention, projection, norm, MLP, residual, and KV values as GPU `X::Tensor` objects.
  - Completed: `QwenVisionRunner` executes all 24 vision blocks plus the merger in C++, with GPU RoPE and GELU operations and no NumPy tensor round trip.

### GPU Asynchronous Chain Checkpoint

The GPU-resident TensorRT path now uses one ordered CUDA per-thread execution stream. TensorRT outputs are wrapped immediately as GPU `X::Tensor` values; synchronization is deferred until `tensor_to_cpu` or another explicit CPU observation. Temporary tensor storage uses CUDA stream-ordered allocation/free so destructors do not introduce a device-wide synchronization after every subgraph.

Full Qwen3-VL-2B smoke, `frame_0.jpg`, 79-token prompt, 60 visual tokens, 28 text layers, four generated tokens on RTX 4080:

- Output remained `A man sits in` across repeated runs.
- Vision blocks plus merger: `80.9-82.6 ms` (previous baseline `113.0 ms`).
- Text prefill: `928-952 ms` (previous baseline `1088 ms`).
- Warm decode samples: `57.1-87.3 ms/token`; first decode after setup can still vary up to about `128 ms`.
- Steady image-to-four-token pipeline: `1.27-1.35 s` after model/frontend setup.

The remaining decode variance comes from per-subgraph output/workspace allocation and separate FP32 decoder engines. The next performance stage is runner-owned persistent workspaces plus fused FP16/BF16 TensorRT decoder-layer engines; it is not Python-loop optimization.

New tests:

- `test2026/tests/phase_13_kv_cache_manager/test.py`
- `test2026/tests/phase_14_qwen_vl_prepare_request/test.py`
- `test2026/tests/phase_15_text_kv_cached_attention/test.py`
- `test2026/tests/phase_16_qwen_text_cached_decoder_layer/test.py`
- `test2026/tests/phase_17_qwen_text_cached_prefill_decode/test.py`
- `test2026/tests/phase_18_qwen_vl_cached_prompt_image/test.py`
- `test2026/tests/phase_19_device_resident_kv_cache/test.py`
- `test2026/tests/phase_20_qwen_text_device_kv_prefill_decode/test.py`
- `test2026/tests/phase_08_qwen_vl_native_prompt_image/test.py` with one-call request prep and KV lifecycle enabled.
- `test2026/native/qwen_tokenizer_native_smoke.cpp` validates native encode/decode and Qwen-VL special token lookup through the Garnet DLL.
- `test2026/native/logits_sampler_native_smoke.cpp` validates the debug GPU logits top-1 sampler hook used for internal testing.
- `test2026/native/qwen_vl_request_kv_native_smoke.cpp` validates request GPU pointers plus device KV create/write/read/destroy in one native executable.

### Stage A: One-Call Front Door

Add `qwen_vl_generate` API and route one request through the existing components. The first version may internally call current subgraphs, but callers see one function.

Acceptance:

- One xlang call takes image path and prompt.
- Result text matches current phase 08 smoke output.
- Result includes timing fields.

### Stage B: Request-Local Visual Cache

Move visual token computation into `QwenVLRequestContext` and ensure decode never recomputes the vision tower.

Acceptance:

- Log/timing proves vision tower runs once per request.
- Generated text is unchanged.

### Stage C: KV Cache Prefill/Decode

Implement paged KV buffers and split text execution into prefill and decode.

Acceptance:

- Decode step consumes one token after prefill.
- Prompt token count no longer increases full-sequence compute cost linearly each step.
- KV pages are allocated/freed correctly under repeated requests.

### Stage D: Fused Layer Execution

Replace per-layer multi-subgraph Python orchestration with C++ model-owned execution.

Acceptance:

- No Python loop over layers in serving path.
- One decoder layer call or fused model block call per layer.
- Engine objects are cached, not rebuilt or reloaded per token.

### Stage E: Scheduler

Add single-GPU request scheduler with prefill/decode queues.

Acceptance:

- Multiple requests can be submitted.
- Decode batching is explicit.
- Per-request outputs and timings are isolated.
