# Qwen-VL Milestones

## Milestone 0: Repository Baseline

Status: mostly done.

Deliverables:

- Qwen3-VL `.x` model skeleton exists.
- Workbench can visualize model/function/matrix graphs.
- Phase 04 tests exist.
- Dataset fixture exists.

Exit criteria:

- New developers can open Workbench and see Qwen3-VL block/matrix structure.
- `.venv` can run Phase 04 tests in skip-safe mode.

## Milestone 1: HF Reference Harness

Goal: establish a stable Qwen-VL reference path.

Tasks:

- Define fixed prompt suite:
  - scene caption
  - visible people count
  - object/vehicle question
  - event verification question
- Add prompt IDs and expected output schema.
- Save HF output JSON with model ID, image path, prompt, answer, device, dtype, latency.
- Add optional processor dump:
  - `input_ids`
  - `attention_mask`
  - `pixel_values`
  - image grid metadata
  - prompt/chat template string

Exit criteria:

- `RUN_HF_QWEN_VL=1` generates reference outputs.
- The output file can be reused by Garnet parity tests.

## Milestone 2: Garnet Import And Input Contract

Goal: make the Garnet test use the same input contract as HF.

Tasks:

- Keep robust `garnet.dll` discovery.
- Add explicit xlang/Garnet import diagnostics.
- Define the first Garnet request object:

```text
image tensor
input ids
attention mask
image grid metadata
generation config
```

- Add a processor dump reader.
- Make Garnet test fail only at the exact unsupported runtime step.

Exit criteria:

- `RUN_GARNET_QWEN_VL=1` can import xlang and Garnet when DLL is present.
- Test prints the exact xmodel path, config path, image path, and input tensor shapes.

## Milestone 3: Config And Weight Mapping

Goal: validate that Garnet understands Qwen3-VL model metadata.

Tasks:

- Add config loader for:
  - text layers
  - hidden size
  - attention heads
  - KV heads
  - vision depth
  - vision hidden size
  - patch size
  - merge size
  - vocab size
- Add safetensors index reader.
- Map HF weight names to xlang model names.
- Validate dimensions.
- Report missing, extra, and mismatched weights.

Exit criteria:

- Garnet can load model metadata and produce a weight validation report.
- Workbench can display the same report.

## Milestone 4: Subgraph Execution

Goal: run isolated Qwen blocks before end-to-end inference.

Subgraphs:

- VisionMLP
- VisionAttention without varlen batching
- VisionPatchMerger
- Qwen3TextMLP
- Qwen3TextAttention without paged KV
- Multimodal merge
- MRoPE index/embedding path

Tasks:

- Create PyTorch/HF reference tensor dumps for each subgraph.
- Create Garnet execution path for each subgraph.
- Compare outputs with tolerances:
  - fp32 debug: tight tolerance
  - fp16/bf16: relaxed tolerance

Exit criteria:

- At least VisionMLP and TextMLP pass numeric parity.
- Unsupported ops are listed in `operator-coverage.md`.

## Milestone 5: End-To-End Forward

Goal: run one image/prompt through Garnet.

Tasks:

- Load weights.
- Run vision encoder.
- Merge visual tokens into text sequence.
- Build position ids/MRoPE metadata.
- Run text decoder prefill.
- Produce logits for next token.

Exit criteria:

- Garnet returns logits with expected shape.
- Garnet logits are comparable to HF for one test request.

## Milestone 6: Decode Loop And KV Cache

Goal: generate tokens, not only one forward pass.

Tasks:

- Implement KV cache layout.
- Add paged KV update.
- Add paged attention or compatible decode attention.
- Add simple greedy decode.
- Add max-new-token cap.

Exit criteria:

- Garnet generates non-empty text from one image/prompt.
- Decode loop records per-token latency.

## Milestone 7: Serving Runtime

Goal: move from test script to serving architecture.

Tasks:

- Define request lifecycle:
  - preprocess
  - vision prefill
  - multimodal merge
  - text prefill
  - decode
  - postprocess
- Add scheduler queues:
  - fresh image jobs
  - cached image/prefix jobs
  - decode jobs
- Add Workbench traces.

Exit criteria:

- One local process can serve repeated VLM requests.
- Workbench can show request timeline and selected tensor snapshots.

## Milestone 8: Performance Baseline

Goal: measure before optimizing.

Metrics:

- end-to-end latency
- vision encoder latency
- multimodal merge latency
- text prefill latency
- time to first token
- per-token decode latency
- GPU memory usage
- CPU preprocessing time

Target hardware:

- RTX 4080 first.
- Linux GPU server later.

Exit criteria:

- Benchmark script produces JSON and human-readable summary.
- Results identify the largest bottleneck.
