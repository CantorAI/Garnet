# Workbench MVP Design

## Purpose

Workbench is not only a pretty graph viewer. It is the tool that lets us adapt a
new model family into Garnet without guessing.

For Qwen3-VL, it should expose the path from:

```text
.x model source
  -> parsed model blocks
  -> TensorExpression / TensorGraph nodes
  -> backend lowering decision
  -> runtime trace and optional tensor snapshots
```

## Non-Goals For MVP

- no full training visualization
- no full tensor dump by default
- no remote multi-user dashboard
- no promise that every `.x` construct is executable on day one
- no automatic proof that Garnet numerics match PyTorch

## Core Concepts

### Model Block

A human-sized component such as:

- `Qwen3VisionEncoder`
- `VisionBlock[0]`
- `VisionAttention[0]`
- `VisionPatchMerger`
- `Qwen3PrepareInputsEmbeds`
- `Qwen3TextDecoderLayer[0]`
- `Qwen3TextAttention[0]`
- `PagedKVUpdate`
- `PagedAttention`

Blocks are what the UI should show first.

### Op Node

A lower-level graph node emitted from `.x`, usually one of:

- standard tensor op
- `T.binary_op(...)`
- `T.unary_op(...)`
- backend plugin op
- custom CUDA op
- debug-only op

Op nodes are what backend engineers inspect.

### Backend Status

Each node should have one backend status:

- `native`: supported by Garnet tensor expression/JIT path
- `trt`: supported by TensorRT lowering
- `cuda`: supported by custom Garnet CUDA kernel
- `plugin`: supported by TRT plugin or external kernel wrapper
- `debug`: supported only by CPU/debug snapshot path
- `missing`: known needed but not implemented
- `unknown`: parser saw it but no registry entry exists
- `mismatch`: expected shape/dtype/backend contract does not match runtime data

This is where the earlier "missing/unknown/mismatched" idea belongs: not as a
replacement for writing correct `.x` model files, but as a Workbench status layer
over correct model source.

## MVP Views

### 1. Model Graph

Show a block-level graph with collapsible detail:

```text
Image/Input
  -> Vision PatchEmbed
  -> Vision Blocks
  -> Vision Merger + DeepStack
  -> Visual Placeholder Merge
  -> Text Decoder Prefill
  -> Paged KV Cache
  -> Decode
  -> LM Head
```

Minimum interactions:

- click block to see source file/function
- expand block to see op nodes
- filter by backend status
- search by op name or weight prefix

### 2. Op Support Table

Columns:

- op name
- source file/function
- expected input shape
- expected output shape
- dtype
- backend target
- implementation file
- status

This table becomes the implementation checklist.

### 3. Config Inspector

For Qwen3-VL-2B, show important config values:

- text layers: 28
- text hidden size: 2048
- attention heads: 16
- KV heads: 8
- head dim: 128
- max positions: 262144
- MRoPE sections: `[24, 20, 20]`
- vision depth: 24
- vision hidden size: 1024
- vision heads: 16
- patch size: 16
- temporal patch size: 2
- spatial merge size: 2
- DeepStack indexes: `[5, 11, 17]`

### 4. Runtime Trace Timeline

Show events from Garnet runtime:

- request accepted
- image preprocess
- vision prefill
- multimodal merge
- text prefill
- decode step
- KV cache allocate/update
- scheduler wait
- kernel launch
- snapshot saved
- error/mismatch

MVP can load a JSONL trace file from disk.

### 5. Tensor Snapshot Viewer

Show metadata first, not huge tensor contents:

- node id
- block id
- shape
- dtype
- device
- min/max/mean if collected
- file path to snapshot
- comparison target if available

For GPU data, Garnet can optionally copy a small sampled slice or full tensor to
CPU only when debug flags are enabled.

## MVP Pipeline

### Phase A: Static Graph Generator

Read `.x` files and produce graph JSON.

MVP parser can be simple:

- parse function definitions
- parse `T.binary_op("name")`
- parse `T.unary_op("name")`
- parse obvious calls between Qwen3-VL helper functions
- attach file and line numbers
- use Qwen3-VL config to expand repeated layers symbolically

This parser does not need to execute the model.

### Phase B: Registry Join

Join graph nodes with an op registry:

```text
op name -> backend target -> implementation status -> source file
```

At first this can be a JSON file. Later it should be generated from C++ handler
registration.

### Phase C: Browser UI

Use a local web page:

- left: model graph
- right: selected node details
- bottom: op table / trace events

The first UI can be static HTML/JS reading local JSON.

### Phase D: Runtime Overlay

When Garnet emits trace JSONL, Workbench overlays runtime timing and snapshot
metadata onto graph nodes.

## First Success Criteria

- Load Qwen3-VL `.x` source.
- Produce a graph with vision, adapter, text, KV, attention, and LM head blocks.
- Show all Qwen3-VL open ops from current `.x` files.
- Mark each op as supported/missing/unknown from a registry.
- Click an op and jump to file/function/line.
- Load a fake trace JSONL and color nodes by timing.

## Later Additions

- compare Garnet tensor snapshot against PyTorch/HF reference output
- per-node numerical error heat map
- memory timeline
- KV cache page map
- scheduler queue visualization
- backend partition editor
- CUDA graph capture status
- Windows and Linux runtime comparison

