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
- Native JPEG/CUDA preprocess is validated and no longer the major cost.
- Current slow path is model execution: repeated small subgraph calls and no production KV decode.

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
  - Current limitation: the primitive is not yet wired into full Qwen decoder layer execution, so phase 08 still uses the slow full-sequence decode path.

New tests:

- `test2026/tests/phase_13_kv_cache_manager/test.py`
- `test2026/tests/phase_14_qwen_vl_prepare_request/test.py`
- `test2026/tests/phase_15_text_kv_cached_attention/test.py`
- `test2026/tests/phase_08_qwen_vl_native_prompt_image/test.py` with one-call request prep and KV lifecycle enabled.

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
