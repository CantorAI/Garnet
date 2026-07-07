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

## Stage 6: End-To-End Forward

### Goal

Run one image/prompt through Garnet and produce logits.

### Done Criteria

- output logits shape is correct
- top-k tokens are compared to HF reference
- first unsupported op, if any, is reported by name/source location

## Stage 7: Decode And KV Cache

### Goal

Generate tokens with decode loop and KV cache.

### Done Criteria

- one prompt generates non-empty text
- KV cache allocation is explicit
- per-token latency is reported
- repeated system prompt cache path is testable

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

Fresh `XTensor` output allocation from this varfunc path is also not stable yet. Phase 00 temporarily writes the TRT output into the input tensor carrier and the Python test compares the valid output prefix. Before Qwen-VL subgraphs use this path, implement a real output tensor factory/allocation path and remove the carrier workaround.

### Stage 0 Next Required Fixes

1. Implement stable fresh output tensor allocation/return from C++ `TRTBuilder::RunMatmulEngine`.
2. Replace the carrier workaround with a correctly shaped output tensor.
3. Move from single matmul engine export to graph capture/lowering for multiple ops.
4. Add a repeat execution parity test where `range(5)` actually executes five matmuls, not only expression-load plus one forward matmul.

Do not start Qwen-VL model execution debugging until Stage 0 keeps passing after the fresh-output tensor fix.

## Current Next Task

Finish the Stage 0 output tensor fix, then implement Stage 2:

```text
test_processor_contract.py
```

with:

- prompt suite
- HF processor dump
- JSON metadata
- NPZ tensor file
- reload validation
- `.venv` test command
