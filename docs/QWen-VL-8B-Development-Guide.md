# Garnet Architecture and QWen-VL-8B Development Guide

## Overview
Garnet is a high-performance deep learning inference engine that utilizes XLang's Tensor Expression system. Instead of executing operations sequentially like traditional eager-execution frameworks (e.g., standard PyTorch), Garnet embraces a **fusion-first** strategy. By representing the entire model or sub-graphs as Tensor Expressions, it generates a unified, highly optimized CUDA kernel (Fusion Kernel) at runtime. 

This document serves as a foundational understanding of Garnet's architecture and provides a guide for implementing large-scale models like **QWen-VL-8B**.

---

## Architecture Breakdown

### 1. XLang Tensor Expression & Graph Construction
Garnet bridges XLang and CUDA. When a model's forward pass is executed in XLang, the operations are not immediately computed. Instead, they build a computational graph:
- **`TensorExpression` (`xlang/Tensor/tensor_expression.h`)**: Captures operations (Add, Mul, Matmul, etc.) and their operands as AST nodes. It also supports tracking control flows (`BranchBegin`, `BranchEnd` for if/else blocks).
- **`TensorGraph` (`xlang/Tensor/tensor_graph.h` & `tensor_build_graph.cpp`)**: Traverses the `TensorExpression` nodes and linearly schedules them into a series of `TensorRunItem`s. It manages control flow block states to ensure the generated code retains structural logic.

### 2. Code Generation (`CodeGenerator`)
Once the computational graph is constructed, it is translated into a single CUDA source file string:
- **`CodeGenerator` (`xlang/Tensor/code_generator.h`)**: Iterates over the scheduled `TensorRunItem`s.
- **`GarnetTensor` (`Garnet/src/tensor/garnet_tensor.h`)**: Registers handlers with the `CodeGenerator`. Handlers like `Header()`, `Trailer()`, `Add()`, and `Matmul()` return the actual C++ / CUDA code strings that correspond to the tensor operations.

### 3. JIT Compilation (`CudaJitCompiler`)
- **`CudaJitCompiler` (`Garnet/src/compiler/cuda_jit_compiler.h`)**: Receives the full CUDA source string. It caches previously compiled modules using MD5 hashing of the code to avoid redundant recompilation.
- Uses **NVRTC** (NVIDIA Runtime Compilation) to compile the CUDA C++ string dynamically into PTX code, optimized for specific GPU architectures (e.g., `--gpu-architecture=compute_89`).
- Loads the PTX into a `CUmodule` and retrieves the `CUfunction` kernel pointer.

### 4. Fusion Execution (`Fusionist`)
- **`Fusionist` (`Garnet/src/tensor/garnet_fusion.cpp`)**: Acts as the executor. It manages the lifecycle of the JIT compilation and the kernel launch.
- Dynamically prepares kernel arguments. Scalar values, tensor data pointers, and shape dimensions are grouped together into a unified `kernelArgs` array.
- Dispatches the compiled `CUfunction` onto the GPU via `cuLaunchKernel`.

---

## Developing QWen-VL-8B with Garnet

To implement an advanced Vision-Language Model like QWen-VL-8B using Garnet, development should follow these principles:

### 1. Model Definition in XLang
Define the QWen-VL architecture entirely using XLang scripts. You will define the vision encoder (e.g., ViT), the multimodal adapter, and the LLM decoder using Garnet's tensor operations.
- Group logical blocks (e.g., self-attention, cross-attention, MLP) into functions decorated with the `@fusion` (or equivalent) keyword. This signals the `Fusionist` to compile these blocks into unified CUDA kernels.

### 2. Maximizing Kernel Fusion
QWen-VL-8B has complex attention mechanisms (like RoPE, Flash Attention). 
- Avoid breaking the fusion chain. Ensure that operations like slicing, reshaping, and broadcasting are supported natively by `GarnetTensor` handlers so they can be folded into a single kernel.
- **Custom Optimized Ops**: For highly specialized operations (like optimized Flash Attention or specific INT8/FP8 quantization dequantization steps used by QWen), you may need to extend `GarnetTensor` in `Garnet/src/tensor/garnet_tensor.cpp` to emit optimized `ptx` or leverage pre-compiled `cuda_lib.cu` optimized TensorCore operations.

### 3. Precision and TensorCore Utilization
QWen-VL-8B is large, and optimal inference requires leveraging FP16, BF16, or FP8 (supported in `CudaCodeGen::GetTypeString`).
- Ensure that the XLang script properly types the model weights.
- When expanding `Matmul` operations for the Linear layers in the transformer blocks, bind them to CUDA libraries or custom TensorCore implementations inside the JIT compiler handlers.

### 4. Control Flow inside Kernels
QWen's generation loop or dynamic vision-patch processing might require conditional logic.
- Utilize XLang's standard `if/else` conditions. `TensorGraph` maps these into `BranchBegin` and `BranchEnd` handlers, which emit corresponding CUDA `if/else` statements directly into the kernel. 
- This avoids host-to-device synchronization overheads for dynamic routing (useful for Mixture of Experts or dynamic sequence lengths).

## Next Steps for QWen-VL-8B Integration
1. **Implement Missing Handlers**: Verify that all operators required by QWen (e.g., LayerNorm, RMSNorm, RoPE, Silu) are implemented in `GarnetTensor`'s code generation (`Garnet/src/tensor/garnet_tensor_gen.cpp`).
2. **Memory Management**: For an 8B model, KV-cache management must be optimized. Implement KV-cache operations as fusible XLang expressions.
3. **Benchmarking**: Use `Fusionist` logs to extract the generated `.cu` and `.ptx` files. Profile them using Nsight Compute to ensure TensorCores are active and register spillage is minimized.
