# Garnet VLM Serving Plan

The current serving design is defined by:

- [Garnet Multi-Backend Architecture](garnet-architecture-v2.md)
- [Compiled xModel Pipeline and Multi-GPU Architecture](qwen-vl-implementation/compiled-xmodel-pipeline-and-multigpu-architecture.md)
- [Continuous Batching and VLM Scheduler Design](qwen-vl-implementation/continuous-batching-and-vlm-scheduler-design.md)

The active implementation target is Qwen3-VL-2B-Instruct:

1. Native tokenizer and image preprocessing produce xlang tensors.
2. Backend-neutral `.x` model programs capture one TensorGraph.
3. The host selects TensorRT when loading/compiling the graph.
4. TensorRT layers and semantic plugins execute prefill and decode.
5. A global paged GPU KV pool persists independently of batch membership.
6. The scheduler forms fixed decode buckets with active-row masks.
7. GPU sampling and fixed-address buffers support low-overhead replay.
8. Debug probes provide targeted visibility without a paired PyTorch runtime.

The removed NVRTC/direct source-generation implementation is not a production
fallback and must not be reintroduced through `GarnetTensor`.
