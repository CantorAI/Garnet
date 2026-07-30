# TensorRT Backend Notes

The current normative design is
[Garnet Multi-Backend Architecture](garnet-architecture-v2.md).

TensorRT is a backend selected by the host at compile/load time. Model `.x`
files are backend-neutral and never call `T.set_backend`.

TensorGraph operations lower to:

- native TensorRT layers for dense static math;
- TensorRT plugins for semantic operations such as paged KV attention;
- explicit compile errors when an operation has no TensorRT implementation.

The removed NVRTC/direct-CUDA-generation path is not a fallback. New backends
must implement the common lowering context and consume the same captured
TensorGraph.
