# Qwen-VL Autonomous Implementation Contract

## Purpose

This document constrains the Qwen-VL implementation so Codex can work stage-by-stage with minimal human prompting.

The goal is not to debate prompts or product UX during implementation. The goal is:

```text
make HF reference -> processor contract -> Garnet input contract -> weight validation
-> subgraph parity -> end-to-end forward -> decode/serving
```

with clear test gates after each stage.

## Autonomy Rules

Codex should proceed without asking the user when:

- choosing ordinary file names inside the existing repo structure
- adding tests/docs/scripts for the current stage
- defining MVP object-detection prompts
- adding skip-safe tests
- adding artifact writers/readers
- improving diagnostics
- fixing bugs found by the stage test loop
- adding small helper modules to reduce repeated code

Codex should ask the user only when:

- model weights location is required and cannot be inferred
- a command needs network/model download and approval is required
- a design decision changes the public Garnet API shape
- a change would remove or rewrite existing user work
- a test requires long GPU execution and the user has not opted in
- the output schema must match an external product contract not already documented here

## MVP Product Prompt Contract

Use object/event style prompts. Do not wait for custom WorldSense prompts.

### Prompt Suite

Use these prompt IDs:

#### `objects_json`

```text
Detect visible objects in this image. Return JSON only:
{
  "objects": [
    {
      "type": "person|vehicle|bicycle|sign|animal|object|other",
      "bbox": [x1, y1, x2, y2],
      "confidence": "low|medium|high",
      "description": "short phrase"
    }
  ]
}
```

#### `scene_summary`

```text
Describe the scene in one short sentence, focusing on people, vehicles, objects, and activity.
```

#### `event_verify`

```text
Return JSON only:
{
  "event": "entering|leaving|waiting|interacting|moving|unknown",
  "evidence": "one short sentence"
}
```

#### `safety_check`

```text
Return JSON only:
{
  "hazards": [
    {
      "type": "traffic|crowd|blocked_path|fall_risk|unknown",
      "description": "short phrase"
    }
  ]
}
```

### Output Policy

For MVP, generated text equality is not required.

Validation order:

1. Processor tensor parity.
2. Tensor/logit parity for subgraphs.
3. Top-k/logit similarity for end-to-end forward.
4. Structured JSON parseability for generated output.
5. Qualitative text similarity only as a final loose check.

## Artifact Layout

Use this folder for generated local artifacts:

```text
test2026/artifacts/qwen_vl_reference/
```

Default artifacts are local development artifacts. Do not commit large generated tensors unless explicitly asked.

Recommended `.gitignore` additions when artifacts are added:

```text
test2026/artifacts/
```

Small JSON schemas or examples may be committed under:

```text
test2026/tests/phase_04_qwen_vl_reference/examples/
```

## Stage 1: HF Reference Output

### Goal

Run HF Qwen-VL on the dataset fixture and save reference outputs.

### Inputs

- `data/Dataset.1980Love/imgs/frame_0.jpg`
- adjacent `frame_0.json`
- prompt suite above
- model ID from `HF_QWEN_VL_MODEL_ID`

Default model ID:

```text
Qwen/Qwen3-VL-2B-Instruct
```

### Test Command

```powershell
$env:RUN_HF_QWEN_VL="1"
$env:HF_QWEN_VL_MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"
.\.venv\Scripts\python.exe test2026\tests\phase_04_qwen_vl_reference\test_hf_qwen_vl.py
```

### Required Output

```json
{
  "model": "...",
  "image": "...",
  "prompt_id": "objects_json",
  "prompt": "...",
  "answer": "...",
  "device": "cuda|cpu",
  "latency_ms": 0.0
}
```

### Done Criteria

- test is skip-safe by default
- enabled test generates non-empty answers
- JSON prompts produce parseable JSON when possible
- output path is printed

### Current Status

Real Qwen3-VL image + prompt inference has been run on local dataset images.

Single-image output:

```text
test2026/artifacts/qwen_vl_reference/hf_qwen3_vl_frame0_answer.json
```

Dataset prompt runner:

```text
test2026/tests/phase_04_qwen_vl_reference/test_hf_qwen_vl_dataset.py
```

Verified command:

```powershell
$env:RUN_HF_QWEN_VL_DATASET="1"
$env:HF_QWEN_VL_MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"
$env:HF_QWEN_VL_PROMPT="Describe the visible people, objects, and scene context in one concise paragraph."
$env:HF_QWEN_VL_MAX_IMAGES="3"
$env:HF_QWEN_VL_MAX_NEW_TOKENS="64"
.\.venv\Scripts\python.exe test2026\tests\phase_04_qwen_vl_reference\test_hf_qwen_vl_dataset.py
```

Verified output:

```text
test2026/artifacts/qwen_vl_reference/hf_qwen3_vl_dataset_answers_3.json
```

## Stage 2: Processor Contract

### Goal

Dump exact HF processor inputs so Garnet can consume the same prepared tensors.

### Required Dump Files

For each prompt/image pair:

```text
processor_frame_0_objects_json.json
processor_frame_0_objects_json.npz
```

### Required Metadata JSON

```json
{
  "schema_version": "0.1",
  "model": "Qwen/Qwen3-VL-2B-Instruct",
  "processor_class": "...",
  "image": "...",
  "prompt_id": "objects_json",
  "prompt": "...",
  "chat_template": "...",
  "arrays": {
    "input_ids": {"shape": [1, 0], "dtype": "int64"},
    "attention_mask": {"shape": [1, 0], "dtype": "int64"},
    "pixel_values": {"shape": [], "dtype": "..."},
    "image_grid_thw": {"shape": [1, 3], "dtype": "int64"}
  }
}
```

### Required NPZ Keys

At minimum:

```text
input_ids
attention_mask
```

When processor returns them:

```text
pixel_values
image_grid_thw
pixel_values_videos
video_grid_thw
position_ids
```

### Test Command

```powershell
$env:RUN_HF_QWEN_VL_PROCESSOR="1"
$env:HF_QWEN_VL_MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"
.\.venv\Scripts\python.exe test2026\tests\phase_04_qwen_vl_reference\test_processor_contract.py
```

### Done Criteria

- dumps metadata JSON and NPZ
- reloads NPZ and validates keys/shapes/dtypes
- validates `attention_mask.shape == input_ids.shape`
- validates prompt text appears in chat template or fallback prompt
- prints all tensor shapes

## Stage 3: Garnet Processor Dump Reader

### Goal

Garnet-side test reads the Stage 2 dump and converts arrays into Garnet tensors.

### Inputs

```text
GARNET_QWEN_VL_PROCESSOR_DUMP_JSON
GARNET_QWEN_VL_PROCESSOR_DUMP_NPZ
```

If env vars are absent, use the newest processor dump in:

```text
test2026/artifacts/qwen_vl_reference/
```

### Test Command

```powershell
$env:RUN_GARNET_QWEN_VL="1"
$env:GARNET_DLL_PATH="D:\CantorAI2026\Garnet\out\build\x64-Debug\bin\garnet.dll"
.\.venv\Scripts\python.exe test2026\tests\phase_04_qwen_vl_reference\test_garnet_qwen_vl.py
```

### Done Criteria

- imports `xlang`
- imports `garnet` from explicit DLL path or discovered DLL path
- reads dump metadata and NPZ
- prints all input names/shapes/dtypes
- creates Garnet tensors or reports the exact missing tensor API
- does not run full model yet

## Stage 4: Config And Weight Validation

### Goal

Validate Garnet model metadata before execution.

### Inputs

- Qwen config JSON
- safetensors index or weight folder
- xlang model source

### Required Report

```json
{
  "missing_weights": [],
  "extra_weights": [],
  "mismatched_shapes": [],
  "matched_weights": 0
}
```

### Done Criteria

- fails before CUDA if a required shape/name is wrong
- prints mapping from HF weight name to xlang name
- Workbench can consume or display the report later

## Stage 5: Subgraph Parity

### Goal

Compare Garnet subgraph output against HF/PyTorch reference tensors.

### Priority

1. `Qwen3TextMLP`
2. `VisionMLP`
3. `rms_norm`
4. `layer_norm`
5. multimodal merge
6. attention projection + RoPE
7. one decoder layer

### Done Criteria

- each subgraph has reference input/output dump
- Garnet output shape matches
- numeric tolerance is documented
- mismatch report includes max error, mean error, and first mismatching index

### Current Status

Implemented and passing with deterministic tiny `Qwen3TextMLP` and `VisionMLP` reference gates:

```text
test2026/tests/phase_05_subgraph_parity/test_text_mlp_reference.py
```

It writes:

```text
test2026/artifacts/qwen_vl_subgraphs/qwen3_text_mlp_tiny_reference.json
test2026/artifacts/qwen_vl_subgraphs/qwen3_text_mlp_tiny_reference.npz
```

Formula:

```text
gate = x @ W_gate.T
up = x @ W_up.T
hidden = silu(gate) * up
output = hidden @ W_down.T
```

The Garnet parity test executes this tiny MLP through Garnet/TensorRT when enabled:

```text
test2026/tests/phase_05_subgraph_parity/test_text_mlp_garnet.py
```

Enabled command:

```powershell
$env:RUN_GARNET_TEXT_MLP_PARITY="1"
$env:GARNET_DLL_PATH="D:\CantorAI2026\Garnet\out\build\x64-Debug\bin\garnet.dll"
.\.venv\Scripts\python.exe test2026\tests\phase_05_subgraph_parity\test_text_mlp_garnet.py
```

Tiny-subgraph verified error:

```text
Qwen3TextMLP:
  max_error ~= 4.0e-6
  mean_error ~= 1.2e-6

VisionMLP:
  max_error ~= 3.0e-8
  mean_error ~= 5.3e-9
```

Additional VisionMLP files:

```text
test2026/tests/phase_05_subgraph_parity/test_vision_mlp_reference.py
test2026/tests/phase_05_subgraph_parity/test_vision_mlp_garnet.py
test2026/tests/phase_05_subgraph_parity/vision_mlp_trt.x
```

Implemented and passing with real `Qwen/Qwen3-VL-2B-Instruct` safetensors for layer-0 subgraphs:

```text
test2026/tests/phase_05_subgraph_parity/test_real_qwen_mlp_subgraphs.py
test2026/tests/phase_05_subgraph_parity/rms_norm_trt.x
test2026/tests/phase_05_subgraph_parity/layer_norm_trt.x
test2026/tests/phase_05_subgraph_parity/text_qkv_proj_trt.x
test2026/tests/phase_05_subgraph_parity/text_qkv_head_norm_trt.x
test2026/tests/phase_05_subgraph_parity/text_o_proj_trt.x
test2026/tests/phase_05_subgraph_parity/text_rope_apply_trt.x
test2026/tests/phase_05_subgraph_parity/text_attention_core_trt.x
test2026/tests/phase_05_subgraph_parity/text_post_rms_norm_trt.x
test2026/tests/phase_05_subgraph_parity/text_lm_head_trt.x
test2026/tests/phase_05_subgraph_parity/linear_bias_trt.x
test2026/tests/phase_05_subgraph_parity/vision_patch_embed_trt.x
test2026/tests/phase_05_subgraph_parity/vision_attention_core_trt.x
test2026/tests/phase_05_subgraph_parity/text_mlp_trt.x
test2026/tests/phase_05_subgraph_parity/vision_mlp_trt.x
```

Verified command:

```powershell
$env:RUN_GARNET_REAL_QWEN_MLP_PARITY="1"
.\.venv\Scripts\python.exe test2026\tests\phase_05_subgraph_parity\test_real_qwen_mlp_subgraphs.py
```

Real-weight verified errors:

```text
Text RMSNorm:
  input [7,2048], weight [2048]
  max_error ~= 1.2e-7
  mean_error ~= 1.8e-9

Vision LayerNorm:
  input [9,1024], weight [1024]
  max_error ~= 9.5e-7
  mean_error ~= 1.8e-8

Text QKV projection:
  input [4,2048]
  q [2048,2048], k [1024,2048], v [1024,2048]
  output [4,4096]
  max_error ~= 3.4e-5
  mean_error ~= 4.6e-6

Text QKV projection + per-head Q/K RMSNorm:
  input [4,2048]
  q_heads 16, kv_heads 8, head_dim 128
  output [4,4096] as concat(q_norm, k_norm, v)
  max_error ~= 6.6e-2
  mean_error ~= 7.2e-4
  q_max_error ~= 1.8e-2
  k_max_error ~= 6.6e-2
  v_max_error ~= 2.2e-5

Text output projection:
  input [4,2048], weight [2048,2048]
  output [4,2048]
  max_error ~= 2.8e-5
  mean_error ~= 4.6e-6

Text RoPE apply:
  input [4,4096], cos/sin [4,128]
  q_heads 16, kv_heads 8, head_dim 128
  output [4,4096] as concat(q_rope, k_rope, v)
  max_error ~= 9.5e-7
  mean_error ~= 4.5e-9
  v_max_error = 0

Text attention core:
  input [4,4096] as concat(q_rope, k_rope, v)
  GQA repeat 8 kv heads -> 16 q heads
  causal softmax prefill
  output [4,2048]
  max_error ~= 9.3e-5
  mean_error ~= 2.1e-6

Text decoder layer chain:
  input [4,2048]
  chain: input RMSNorm -> QKV/head norm -> RoPE -> attention -> output projection -> residual -> post RMSNorm -> MLP -> residual
  output [4,2048]
  max_error ~= 8.6e-4
  mean_error ~= 1.4e-4
  post_norm_path = garnet_trt

Text decoder layer vs HF module:
  reference: transformers.Qwen3VLTextDecoderLayer(layer_idx=0, eager)
  HF rotary embeddings are passed into Garnet RoPE
  output [4,2048]
  max_error ~= 8.8e-4
  mean_error ~= 1.1e-4

Two text decoder layers vs HF modules:
  reference: layer 0 -> layer 1, eager
  output [4,2048]
  max_error ~= 1.1e-2
  mean_error ~= 4.0e-4

Two-layer logits:
  chain: two decoder layers -> final RMSNorm -> tied embedding LM head
  logits [4,151936]
  max_error ~= 1.3e-2
  mean_error ~= 1.8e-3
  last-token top-10 overlap = 10/10

Processor-token logits:
  input source: processor_frame_0_objects_json.npz input_ids
  token window: first 8 ids from the real image prompt sequence
  chain: tied token embedding lookup in Python -> two Garnet decoder layers -> final RMSNorm -> tied embedding LM head
  logits [8,151936]
  max_error ~= 1.1e-2
  mean_error ~= 1.3e-3
  last-token top-10 overlap = 10/10

Visual-splice logits:
  input source: processor_frame_0_objects_json.npz input_ids, mm_token_type_ids, pixel_values, image_grid_thw
  token window: first 8 ids from the real image prompt sequence
  visual positions: [4,5,6,7]
  visual token source: first 16 processor patches -> patch embed -> interpolated pos_embed -> optional 24 vision blocks -> merger
  chain: replace placeholder embeddings with Garnet-compatible visual tokens -> two Garnet decoder layers -> final RMSNorm -> tied embedding LM head
  visual tokens [4,2048], logits [8,151936]
  patch+pos+merger mode:
    max_error ~= 7.4e-3
    mean_error ~= 8.7e-4
    last-token top-10 overlap = 10/10
  all-vision-blocks mode:
    vision_block_count = 24
    visual_token_max_error ~= 3.4e-3
    visual_token_mean_error ~= 3.6e-4
    logits max_error ~= 2.4e-2
    logits mean_error ~= 2.1e-3
    last-token top-10 overlap = 10/10

Full prompt-window logits:
  input source: processor_frame_0_objects_json.npz input_ids, mm_token_type_ids, pixel_values, image_grid_thw
  verified token windows: first 8 ids and first 12 ids from the real image prompt sequence
  8-token window visual positions: [4,5,6,7]
  12-token window visual positions: [4,5,6,7,8,9,10,11]
  visual path: processor patches -> patch embed -> interpolated pos_embed -> 24 vision blocks -> merger
  text path: all 28 text decoder layers -> final RMSNorm -> tied embedding LM head
  8-token logits [8,151936]:
    max_error ~= 5.3e-2
    mean_error ~= 4.2e-3
    last-token top-10 overlap = 10/10
    expected_next_token_id = actual_next_token_id = 151645
    expected_next_token_text = actual_next_token_text = <|im_end|>
12-token logits [12,151936]:
    visual_token_max_error ~= 4.5e-2
    visual_token_mean_error ~= 4.5e-3
    max_error ~= 9.9e-1
    mean_error ~= 3.1e-2
    last-token top-10 overlap = 10/10
    expected_next_token_id = actual_next_token_id = 151645
    expected_next_token_text = actual_next_token_text = <|im_end|>

Cache rule found during widening:

```text
TensorRT cache directories must include shape-specific suffixes such as patch_32 and tokens_12.
Do not reuse a static-shape engine cache across different token or patch counts.
```

Model forward façade:

```text
GarnetQwen3VLForwardFacade.forward(
  input_ids,
  pixel_values,
  image_grid_thw,
  mm_token_type_ids,
  cos,
  sin
) -> logits
```

Verified behavior:

```text
input source: processor_frame_0_objects_json.npz
token window: first 12 ids
visual positions: [4,5,6,7,8,9,10,11]
vision path: 32 processor patches -> patch embed -> interpolated pos_embed -> 24 vision blocks -> merger
text path: all 28 text decoder layers -> final RMSNorm -> tied embedding LM head
logits [12,151936]
max_error ~= 3.8e-1
mean_error ~= 1.3e-2
last-token top-10 overlap = 10/10
expected_next_token_id = actual_next_token_id = 151645
expected_next_token_text = actual_next_token_text = <|im_end|>
```

Current limitation:

```text
The façade is a Python class that calls Garnet subgraph models internally.
It still receives precomputed text RoPE cos/sin from the HF rotary helper.
The next implementation step is to move this façade contract into a Garnet model-level entry and replace HF rotary preparation.
```

Native MRoPE position preparation:

```text
qwen3vl_mrope_position_ids_numpy(input_ids, mm_token_type_ids, image_grid_thw, video_grid_thw, attention_mask)
```

Verified against local `transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLModel.get_rope_index` using a shape-consistent synthetic image span:

```text
input_shape = [1,13]
image_grid_thw = [[1,4,8]]
visual_span_len = 8
position_ids_shape = [3,1,13]
mrope_position_deltas = [-4]
max_position = 8
```

Native MRoPE cos/sin preparation:

```text
qwen3vl_text_mrope_cos_sin_numpy(position_ids, head_dim, rope_theta, mrope_section)
```

Verified against local `Qwen3VLTextRotaryEmbedding`:

```text
input_shape = [1,13]
position_ids_shape = [3,1,13]
cos_shape = [1,13,128]
sin_shape = [1,13,128]
max_error ~= 4.8e-7
```

Native-RoPE model-forward facade:

```text
GarnetQwen3VLForwardFacade.forward(
  input_ids,
  pixel_values,
  image_grid_thw,
  mm_token_type_ids
) -> logits
```

Verified behavior:

```text
input source: processor_frame_0_objects_json.npz pixel_values, first 32 patches
synthetic shape-consistent prompt length: 13 tokens
image_grid_thw = [[1,4,8]]
visual positions: [3,4,5,6,7,8,9,10]
position_ids_shape = [3,1,13]
cos_shape = [1,13,128]
sin_shape = [1,13,128]
vision path: 32 processor patches -> patch embed -> interpolated pos_embed -> 24 vision blocks -> merger
text path: all 28 text decoder layers -> final RMSNorm -> tied embedding LM head
logits [13,151936]
max_error ~= 2.8e-2
mean_error ~= 2.4e-3
last-token top-10 overlap = 10/10
expected_next_token_id = actual_next_token_id = 151644
expected_next_token_text = actual_next_token_text = <|im_start|>
```

Important caveat:

```text
The saved real processor prefix tests use only the first 8 or 12 tokens of a much larger image span.
Those truncated prefixes are useful for lightweight logits parity, but they are not valid standalone MRoPE sequences for the real image_grid_thw [[1,68,120]], because the full visual span is 2040 LLM visual tokens.
Native MRoPE should be used either with a complete processor prompt or with synthetic/reduced grids whose visual-token count exactly matches the placeholder span.
```

Vision patch embed:
  input source: processor_frame_0_objects_json.npz pixel_values
  token window: first 16 flattened processor patches
  weight: visual.patch_embed.proj.weight flattened from [1024,3,2,16,16] to [1024,1536]
  output [16,1024]
  max_error ~= 4.8e-7
  mean_error ~= 1.5e-8

Vision patch merger:
  input source: first 16 processor patches -> patch embed
  chain: merger LayerNorm -> reshape [16,1024] to [4,4096] -> linear_fc1 + GELU -> linear_fc2
  output [4,2048]
  max_error ~= 1.6e-3
  mean_error ~= 3.3e-4

Vision patch + positional embed + merger:
  input source: processor_frame_0_objects_json.npz pixel_values + image_grid_thw [1,68,120]
  token window: first 16 flattened processor patches
  positional path: HF-compatible bilinear pos_embed interpolation and spatial-merge reorder
  chain: patch embed -> add interpolated pos_embed -> merger LayerNorm -> reshape [16,1024] to [4,4096] -> linear_fc1 + GELU -> linear_fc2
  output [4,2048]
  max_error ~= 1.4e-3
  mean_error ~= 3.1e-4

Vision QKV projection:
  input source: first 16 processor patches -> patch embed + interpolated pos_embed
  chain: visual.blocks.0.attn.qkv linear+bias
  output [16,3072]
  max_error ~= 1.7e-3
  mean_error ~= 1.3e-4

Vision attention core:
  input source: QKV from first 16 patch+pos tokens, with HF-compatible vision RoPE applied in the test harness
  chain: non-causal 16-head attention, head_dim 64
  output [16,1024]
  max_error ~= 1.9e-6
  mean_error ~= 3.5e-8

Vision block 0:
  input source: first 16 processor patches -> patch embed + interpolated pos_embed
  chain: norm1 -> qkv -> vision RoPE -> non-causal attention -> proj -> residual -> norm2 -> MLP -> residual
  output [16,1024]
  max_error ~= 4.4e-4
  mean_error ~= 3.5e-5

Text MLP:
  input [3,2048]
  gate/up [6144,2048], down [2048,6144]
  max_error ~= 1.5e-6
  mean_error ~= 3.2e-7

Vision MLP:
  input [5,1024]
  fc1 [4096,1024], fc2 [1024,4096]
  max_error ~= 1.8e-4
  mean_error ~= 1.5e-5
```

Current default regression:

```text
14 Passed, 0 Failed
```

Next implementation target:

1. move `GarnetQwen3VLForwardFacade.forward()` from Python orchestration toward a Garnet model-level entry for `Qwen3VLModel`
2. widen full prompt-window logits parity toward the complete processor prompt after model-level forward is stable
3. add a decode loop that emits text from Garnet logits, then compare generated answer shape/parseability with the HF reference
4. add KV cache-aware decode kernels after prefill parity is stable

## Stage 6: End-To-End Forward

### Goal

Run one image/prompt through Garnet and produce logits.

### Done Criteria

- output logits shape is correct
- top-k tokens are compared to HF reference
- first unsupported op, if any, is reported by name/source location

### Current Status

Stage 6 has a tested forward-window path in the Phase 05 parity runner:

```powershell
$env:RUN_GARNET_REAL_QWEN_MLP_PARITY="1"
$env:RUN_GARNET_REAL_QWEN_FULL_PROMPT_LOGITS_PARITY="1"
$env:GARNET_QWEN_FULL_TEXT_USE_ALL_VISION_BLOCKS="1"
$env:GARNET_QWEN_FULL_TEXT_TOKEN_WINDOW="8"
.\.venv\Scripts\python.exe test2026\tests\phase_05_subgraph_parity\test_real_qwen_mlp_subgraphs.py
```

This now has a Python model-forward façade with one `.forward()` boundary, but it still uses Python orchestration internally to call Garnet subgraph models. It is not yet one native C++ `Qwen3VLModel.forward()` entry.

Additional verified Stage 6 command for native MRoPE inside the facade:

```powershell
$env:RUN_GARNET_REAL_QWEN_MLP_PARITY="1"
$env:RUN_GARNET_REAL_QWEN_MODEL_FORWARD_NATIVE_ROPE_FACADE_PARITY="1"
.\.venv\Scripts\python.exe test2026\tests\phase_05_subgraph_parity\test_real_qwen_mlp_subgraphs.py
```

## Stage 7: Decode And KV Cache

### Goal

Generate tokens with decode loop and KV cache.

### Done Criteria

- one prompt generates non-empty text
- KV cache allocation is explicit
- per-token latency is reported
- repeated system prompt cache path is testable

### Current Status

Stage 7 has a first non-KV decode smoke test through the Python model-forward facade:

```powershell
$env:RUN_GARNET_REAL_QWEN_MLP_PARITY="1"
$env:RUN_GARNET_REAL_QWEN_MODEL_FORWARD_NATIVE_ROPE_DECODE="1"
$env:GARNET_QWEN_NATIVE_ROPE_DECODE_TOKENS="2"
.\.venv\Scripts\python.exe test2026\tests\phase_05_subgraph_parity\test_real_qwen_mlp_subgraphs.py
```

Verified behavior:

```text
prompt: reduced valid chat template with 8 image placeholders
image source: processor_frame_0_objects_json.npz pixel_values, first 32 patches
image_grid_thw = [[1,4,8]]
visual_token_count = 8
prompt_token_count = 27
decode mode: greedy, no KV cache, full prefill rerun per token
generated_token_ids = [73594, 2236]
generated_text = ```json
vision path: 24 vision blocks
text path: 28 text decoder layers
```

Remaining Stage 7 work:

Paged KV allocation status:

```text
Garnet exports KVCacheManager(max_num_pages, page_size, head_dim, num_kv_heads).
allocate(seq_id, sequence_length) returns page IDs.
free(seq_id) releases pages.
stats() reports max/free/used pages and active sequence count.
```

Verified command:

```powershell
.\.venv\Scripts\python.exe test2026\tests\phase_02_kv_cache\test.py
```

Verified behavior:

```text
max_num_pages = 1024
page_size = 16
sequence_length = 40
allocated_pages = 3
used_pages after allocate = 3
free_pages after free = 1024
```

Remaining Stage 7 work:

1. move decode from full-prefix rerun to explicit prefill/decode split
2. attach real K/V tensors to allocated pages and route text attention through them
3. add per-token latency metrics
4. run a longer structured JSON generation check after KV-backed decode exists

## Stage 8: Serving And Scheduler

### Goal

Move from one-shot scripts to serving behavior.

### Done Criteria

- request object exists
- prefill/decode split exists
- traces are emitted
- Workbench can display runtime stage timing
- benchmark JSON is generated

## Error Reporting Contract

Every failed enabled test should print:

- stage name
- model ID
- image path
- prompt ID
- xmodel path if Garnet stage
- Garnet DLL path if Garnet stage
- exact missing dependency/op/weight
- suggested next command

Avoid vague errors like:

```text
NoneType object is not callable
```

Wrap them with context.

## Commit Policy

Do not commit automatically unless the user asks.

When asked to check in:

- include source/docs/tests/artifacts intentionally created for the stage
- exclude `.venv`, `__pycache__`, `.vs`, `out`, generated large artifacts unless explicitly requested

## Stage 0: TRT Tensor Expression Preflight

Before Qwen-VL work, verify the smallest risky path and keep it on the real public Garnet model API:

```text
garnet.load_model(.x, weights=..., cache_dir=...)
  -> load xlang tensor expression
  -> model owns weights
  -> generate/cache TensorRT engine if missing
  -> attach engine to Model
  -> model.forward(input)
  -> compare with NumPy
```

Test files:

```text
test2026/tests/phase_00_trt_expression_preflight/simple_trt_linear.x
test2026/tests/phase_00_trt_expression_preflight/function_trt_linear.x
test2026/tests/phase_00_trt_expression_preflight/branch_trt_linear.x
test2026/tests/phase_00_trt_expression_preflight/repeat_trt_linear.x
test2026/tests/phase_00_trt_expression_preflight/nested_function_trt_linear.x
test2026/tests/phase_00_trt_expression_preflight/test.py
```

Run:

```powershell
$env:RUN_GARNET_TRT_PREFLIGHT="1"
$env:GARNET_DLL_PATH="D:\CantorAI2026\Garnet\out\build\x64-Debug\bin\garnet.dll"
.\.venv\Scripts\python.exe test2026\tests\phase_00_trt_expression_preflight\test.py
```

Default behavior is skip-safe. Enabled behavior should identify the first break point:

- `dll_import`
- `load_model_or_expression_build`
- `trt_engine_export_missing`
- `trt_engine_export_empty`
- `trt_engine_run`
- `trt_engine_run_returned_none`
- `trt_engine_output_missing_numpy_or_tolist`

### Stage 0 Done Status

Status on 2026-07-06: implemented and passing for these cases:

- simple tensor expression
- function call wrapping a TensorRT matmul expression
- branch selecting a TensorRT matmul expression
- repeat block with `range(5)` and shape-stable `[4,4]` weight
- nested functions shaped like `model -> block -> project`

All enabled cases run:

```text
load_model -> internal engine cache -> model.forward -> NumPy parity
```

The phase-00 test intentionally proves the public model path. Do not reintroduce:

- public `garnet.run_trt_matmul`
- `.x`-level `export_trt(...)`
- passing weights directly to `forward`

### Stage 0 Known Gap

Current `Model::Forward` executes one internal matmul engine for the preflight case. The repeat `.x` file has `range(5)` to prove xlang expression loading/control-flow shape, but full multi-op/multi-layer TensorRT graph execution is not implemented yet.

Fresh `XTensor` output allocation from this varfunc path is now implemented for the phase-00 TRT matmul runner. The Python test validates the returned tensor shape directly instead of comparing an input-carrier prefix.

### Stage 0 Next Required Fixes

1. Move from single matmul engine export to graph capture/lowering for multiple ops.
2. Add a repeat execution parity test where `range(5)` actually executes five matmuls, not only expression-load plus one forward matmul.
3. Remove temporary verbose TRT preflight logging once graph lowering diagnostics exist.

Do not start Qwen-VL model execution debugging until Stage 0 keeps passing after the fresh-output tensor fix.

## Current Next Task

Implement the next Stage 5/6 bridge:

```text
text attention parity -> one decoder layer parity -> logits for one prompt/image
```

The next code task is not processor setup anymore. Processor dumps, real HF image answers, real weight contract, and real-weight TRT subgraph parity are already in place.
