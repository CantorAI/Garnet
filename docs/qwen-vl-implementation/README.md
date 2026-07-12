# Qwen-VL Garnet Implementation Plan

## Purpose

This folder is the execution plan for bringing Qwen3-VL style models into Garnet.

The higher-level strategy lives in:

- `docs/garnet-vlm-serving-plan.md`
- `docs/garnet-architecture-v2.md`

This folder answers the implementation question:

```text
How do we move from HF reference inference to a Garnet Qwen-VL runtime that can
load inputs, lower xlang model expressions, run the model, compare correctness,
and then optimize latency?
```

## Normative Architecture

The production ABI, compiled xmodel plan, parallel pipeline, scheduler, KV cache,
probe, and multi-GPU contract is defined in:

- `compiled-xmodel-pipeline-and-multigpu-architecture.md`
- `compiled-runtime-implementation-plan.md`
- `continuous-batching-and-vlm-scheduler-design.md`

The architecture document is the normative design; the implementation plan is
its required staged execution and test sequence. They supersede older notes
where those notes permit Python-assembled subgraphs, direct DLL orchestration,
or hardcoded Qwen C++ runners.

## Current Baseline

Already in the repo:

- Qwen3-VL xlang model skeleton:
  - `qwen_vl/xmodel/qwen_vl_model.x`
  - `qwen_vl/xmodel/vision_encoder.x`
  - `qwen_vl/xmodel/vl_adapter.x`
  - `qwen_vl/xmodel/qwen_llm.x`
- Workbench MVP for model graph visualization:
  - `workbench/`
- Phase 04 VLM reference tests:
  - `test2026/tests/phase_04_qwen_vl_reference/`
- Local VLM fixture dataset:
  - `data/Dataset.1980Love/`

## Implementation Tracks

The work should move in parallel across five tracks.

### Track A: Reference And Test Harness

Goal: make Hugging Face Qwen-VL the correctness oracle.

Deliverables:

- stable dataset fixture
- stable prompt set
- saved HF reference outputs
- tensor/logit capture where possible
- Garnet parity test with the same input contract

Docs:

- `test-and-parity-plan.md`

### Track B: Input Processor And Token Contract

Goal: make Garnet consume the same conceptual inputs as HF.

Deliverables:

- image loading and resizing policy
- visual token grid metadata
- chat template and tokenizer path
- `input_ids`, image tensors, image grid shape, position ids
- prompt set used by both HF and Garnet tests

The production path uses Garnet's native tokenizer and GPU image processor. HF
is a correctness oracle only and is not a production preprocessing fallback.

### Track C: Model Loading And Weight Binding

Goal: load Qwen3-VL model config and weights into Garnet.

Deliverables:

- config parser
- safetensors index reader
- weight-name mapping from HF keys to xlang names
- dtype policy, initially fp16/bf16
- weight shape validation in Workbench and tests

Docs:

- `runtime-architecture.md`

### Track D: xlang Lowering And Backend Coverage

Goal: make the current Qwen `.x` model expressions lower into executable Garnet/TensorRT/CUDA paths.

Deliverables:

- tensor expression capture for all major Qwen-VL blocks
- backend handler coverage table
- TensorRT lowering for dense projections/MLP/norm where useful
- custom CUDA kernels for multimodal merge, RoPE/MRoPE, KV update, attention
- clean skip/fallback/debug behavior for unsupported ops

Docs:

- `operator-coverage.md`

### Track E: Serving Runtime And Performance

Goal: move from one-shot correctness to serving behavior.

Deliverables:

- request object for image/video/text prompts
- prefill/decode split
- KV cache blocks for text decoder
- vision embedding cache where reuse is meaningful
- scheduler for mixed fresh-frame and decode requests
- trace snapshots for Workbench
- benchmarks on RTX 4080 first

Docs:

- `milestones.md`
- `runtime-architecture.md`
- `continuous-batching-and-vlm-scheduler-design.md`

## Success Criteria

Phase 2 success:

- HF reference test runs on one dataset image with fixed prompts.
- Garnet parity test imports Garnet through xlang and reaches a clearly documented skip/fail point.
- Input contracts are documented.

Phase 3 success:

- Garnet loads Qwen3-VL config and validates expected weight names/shapes.
- Garnet can execute at least a small model subgraph such as VisionMLP or TextMLP and compare to PyTorch/HF tensor output.

Phase 4 success:

- Garnet executes an end-to-end Qwen-VL forward path for one image and prompt.
- Output logits or generated tokens can be compared with HF within an agreed tolerance or qualitative reference.

Phase 5 success:

- Garnet runs a serving loop with prefill/decode split, KV cache, traces, and measured latency.

## Immediate Next Steps

Follow stages 0-3 of `compiled-runtime-implementation-plan.md`:

1. Freeze the current correctness/performance baseline and forbidden-path
   counters.
2. Freeze the production `load_model`, `forward`, prefix, and probe ABI.
3. Implement root `.x` graph capture plus validated captured-graph reuse.
4. Complete generic backend lowering before further Qwen runner optimization.
