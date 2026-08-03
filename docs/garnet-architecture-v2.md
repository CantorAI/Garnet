# Garnet Multi-Backend Architecture

## Contract

Garnet uses the xlang tensor-expression graph as its single model IR:

```text
xModel .x source
  -> xlang tensor-expression capture
  -> TensorGraph
  -> validation and partitioning
  -> selected backend lowering
  -> compiled executable
```

Model source describes a model, not a runtime. A model must not call
`set_backend`. The host selects the backend when it compiles or loads the
model:

```python
model = garnet.load_model(
    "xModel/qwen3/vl_2b_instruct/qwen_text_decode_batch.x",
    runtime_mode="compiled_xmodel",
    backend="tensorrt",
)
```

TensorRT is the first backend. Other backends implement the same lowering
contract; they do not require a second model graph.

## Source Layout

```text
src/
  core/                         backend-neutral lowering context
  tensor/                       TensorGraph capture/replay bridge
  model/                        model loading and compiled runtime
  runtime/
    scheduler/                  continuous batching policy
    kv/                         global paged GPU KV pool
    executor/                   persistent scheduler-to-graph bindings
  backends/
    tensorrt/                   TensorRT lowering and plugins
  cuda/                         optimized semantic CUDA operations
  tokenizer/                    native tokenization
  image/                        native vision preprocessing
  entry/                        public xlang package boundary

xModel/
  qwen3/
    vl_2b_instruct/              Qwen3-VL-2B-Instruct model programs
    text_1_7b/                   Qwen3-1.7B pure-text model programs
```

There is no permanent legacy or direct CUDA-source-generation tree.

## Tensor Frontend

`GarnetTensor` is deliberately small. It records and replays only generic
unary and binary operations plus TensorGraph structure. It does not own a
TensorRT builder, execute eager model kernels, or generate CUDA source.

Model programs use semantic operation names:

```python
qkv = x * T.unary_op("qwen3_text_qkv_packed", ...)
state = qkv * T.binary_op("paged_kv_bind_key_pages") * key_pages
attention = state * T.unary_op("paged_kv_decode_masked_bf16", ...)
```

The selected lowerer must either map each operation to an executable backend
implementation or fail compilation explicitly. Silent fallback is forbidden.

## Tensor and Device Memory

The xlang `Tensor` is the cross-runtime data ABI. It carries:

- shape and data type;
- CPU or device type;
- data pointer;
- device context and device operations;
- optional descriptor metadata.

Backend boundaries exchange tensors, not backend-owned public buffer types.
Paged KV arenas remain allocated for the lifetime of the serving pool. Request
page tables map logical pages to those physical GPU pages; changing a batch
never copies the KV contents.

## TensorRT Backend

TensorRT lowering lives entirely under `src/backends/tensorrt`.

Standard dense operations become TensorRT layers. Scheduler-sensitive
operations such as paged attention become TensorRT plugins backed by optimized
CUDA kernels. This still produces one compiled TensorRT execution graph; the
model source does not directly launch the kernels.

Graph annotations such as `cuda_graph=True` are captured in the execution plan
and passed explicitly to the runtime. Optimization behavior must not depend on
file or cache-directory names.

## Continuous Batching

The scheduler is backend-neutral. Each scheduling tick produces tensor-ready
batch plans:

- request IDs and token IDs;
- context lengths and KV write slots;
- logical-to-physical page tables;
- a fixed bucket size;
- an active-row mask.

Decode uses fixed B1/B2/B4/B8 profiles. Inactive rows are masked inside the
paged-attention plugin and cannot write or read KV pages. Fixed buffer
addresses permit CUDA-graph replay. Requests may join or leave at every token
without rebuilding or moving the global KV pool.

The compiled decode executor converts those plans into persistent xlang GPU
tensors. KV arenas are borrowed tensor bindings owned by the global pool;
metadata and output buffers are owned by the bucket executor. It invokes the
generic compiled runtime, not TensorRT APIs, so the scheduling layer remains
independent of the selected backend.

Prefill and decode are separate graph entrypoints. Decode receives priority;
prefill is admitted under a token budget so it cannot create unbounded
inter-token latency.

The `qwen3_text` frontend applies the Qwen3 chat template, tokenizes directly
in native code, constructs one-component position IDs and paged-KV bindings,
then hands the first sampled token to the cached decode graph. Frontend choice
remains a runtime option; it is not embedded in the model graph.

## Model Organization

Model-specific graph programs belong below `xModel`, not in the generic
runtime:

```text
xModel/qwen3/vl_2b_instruct/
  qwen_vl_model.x
  qwen_vl_prefill.x
  vision_encoder.x
  vl_adapter.x
  qwen_llm.x
  qwen_text_prefill.x
  qwen_text_decode.x
  qwen_text_decode_batch.x

xModel/qwen3/text_1_7b/
  qwen_llm.x
  prefill.x
  decode.x
  decode_batch.x
```

Reusable runtime mechanisms—scheduling, memory allocation, lowering,
execution, sampling, and probes—remain model-neutral C++ code.

## Validation and Debugging

Correctness is checked at the Garnet boundary:

- graph capture and explicit unsupported-op failure;
- backend engine construction and cache invalidation;
- native safetensors loading;
- paged KV write/read and inactive-row preservation;
- end-to-end model output;
- debug probes for selected shapes, statistics, and tensor snapshots.

PyTorch/Hugging Face may be used to establish an external reference, but
production code and the permanent test architecture do not pair every Garnet
operator with a PyTorch implementation.

## First Production Target

The first optimized target is Qwen3-VL-2B-Instruct on TensorRT with BF16,
batched decode, continuous scheduling, paged KV cache, GPU sampling, and
CUDA-graph-capable fixed bucket execution. Qwen3-1.7B text-only should reuse
the same runtime after its separate model graph and checkpoint mapping are
validated.
