# Garnet Custom TRT-Hybrid Architecture
**Target Application:** High-Performance QWen-VL 3 8B Inference on Windows

## Executive Summary
To achieve state-of-the-art inference performance for advanced Vision-Language Models (like QWen-VL 3) without suffering from the Linux-centric complexities of building TensorRT-LLM, Garnet will evolve into a **Custom TRT-Hybrid Engine**. 

Instead of treating TensorRT as a black box that handles the entire model, Garnet will implement a **Partitioned Graph Strategy**. Core TensorRT will be utilized strictly for static, heavy matrix mathematics, while Garnet will natively manage complex LLM memory structures (Paged KV Cache) and dynamic attention mechanisms (FlashAttention 2).

---

## 1. Model Definition & Backend Selection (`.x`)
The neural network's topology (Vision Encoder, VL Adapter, QWen LLM blocks) is described purely in XLang (`.x`) using the `CpuTensor` builder API. 

Before constructing or running the graph, the `.x` script explicitly configures the desired compilation backend directly on the `T` module namespace. We do not use separate tensor classes; the frontend remains unified.
```python
import CpuTensor as T

# Select the backend execution strategy for the graph
T.set_backend("TensorRT") # Enables highly optimized TensorRT Engine compilation
# or
T.set_backend("JIT")      # Falls back to the original NVRTC custom CUDA compilation
```

---

## 2. Python Host API Workflow (End-to-End)
The overarching orchestration of the model will be driven by a standard Python script. Python will utilize `xlang` to bind to Garnet and trigger the model lifecycle.

### A. Weight Loading & Weightless Engine Initialization
Garnet utilizes a **Weightless Engine Blueprint** architecture. The TensorRT `.engine` file contains only the optimized execution graph and kernel selections, keeping the file incredibly small. The massive model weights are loaded separately into GPU VRAM at runtime and passed into the engine dynamically as input bindings.

This allows developers to hot-swap fine-tuned weights (like LoRAs) without ever needing to recompile the TensorRT engine!
```python
from garnet import garnet
T = garnet.tensor()

# 1. Load the pre-trained weights into GPU VRAM
weights = garnet.load_weights("qwen_vl_3_8b/*.safetensors")

# 2. Initialize the execution engine using the .x graph description
engine = garnet.load_model(
    "qwen_vl/xmodel/qwen_vl_model.x", 
    weights=weights,
    cache_dir="./engine_cache"
)
```
When `load_model` is called, Garnet orchestrates the setup:
* **Cache Hit:** If the lightweight `.engine` blueprint already exists, Garnet instantly loads it. The `weights` map is mapped into VRAM and bound to the engine's weight inputs.
* **Cache Miss:** If the `.engine` doesn't exist, the `TRTBuilder` compiles the graph. It designates all weight matrices as *Dynamic Inputs* rather than Constants, saving the resulting tiny blueprint to disk.

### B. Pure Inference Loop (Tensor In -> Tensor Out)
Once the engines are loaded, Python controls the high-level autoregressive generation loop. Garnet handles the execution of the partitioned graph, internal KV Cache state, and memory completely under the hood. The Python API remains pure: **Tensor In, Tensor Out**.
```python
# Host prepares the initial input tensors (Image pixels and Text token IDs)
image_tensor = garnet.Tensor(image_pixel_values)
text_tensor = garnet.Tensor(prompt_token_ids)

# Garnet executes the partitioned graph (Vision Encoder + Adapter + LLM)
# It internally allocates and manages the Paged KV Cache state
logits = engine.forward(image_tensor, text_tensor)
generated_tokens = []

while not is_stop_token(logits):
    next_token = sample(logits)
    generated_tokens.append(next_token)
    
    # Prepare the next input tensor (only text tokens needed for decoding phase)
    next_input_tensor = garnet.Tensor([next_token])
    
    # Garnet seamlessly bounces data between the TRT engine and custom FA2 kernels
    logits = engine.forward(next_input_tensor)
```

---

## 3. The Partitioned Graph Strategy
Under the hood, Garnet intercepts the `.x` tensor expressions and partitions them into two execution domains:

1. **TensorRT Domains (Static Math):** Standard dense operations such as `Matmul`, `LayerNorm`/`RMSNorm`, and `MLP` feed-forward networks. These are compiled into highly optimized `.engine` blocks using the Core TensorRT C++ API.
2. **Garnet Domains (Dynamic Kernels & Memory):** Operations that TRT handles poorly or rigidly, such as Rotary Positional Embeddings (RoPE), Paged KV Cache updating, and Scaled Dot-Product Attention. These are executed directly by Garnet launching specialized CUDA kernels (`cuda_lib.cu`).

---

## 4. Core Components to Implement

### A. TensorRT Engine Builder (`Garnet/src/tensor/trt_builder.cpp`)
A new builder class that implements the `IExecutionBackend` interface.
* **Translation:** Maps XLang operators directly to TensorRT network layers.
* **Attention Holes:** The TRT engine for a single Transformer block will have inputs for $Q, K, V$ and an output for the final projection, purposefully leaving an "execution hole" where the attention mechanism belongs.

### B. Open Ops Architecture (Extensible Registry)
Instead of hardcoding every possible operation (like `trt_matmul`) in the Garnet C++ source (`garnet_tensor.cpp`), Garnet will utilize a dynamic **Open Ops Architecture**. 
* **String-Based Resolution:** In XLang, calling `T.any_custom_op()` simply passes the string name `"any_custom_op"` to the AST. 
* **Dynamic Registration:** Garnet will expose a C++ `std::unordered_map` registry. The `.x` script (or Python host) can dynamically register new operations. Garnet maps these string names directly to TensorRT standard layers, TensorRT Plugins (`IPluginV2`), or custom `nvrtc` CUDA kernels at runtime.
* **Extensibility:** When a new LLM kernel is released, developers can dynamically load the `.cu` or `.dll/.so` plugin from Python/XLang without ever modifying or recompiling Garnet's core C++ engine.

### C. Paged KV Cache Manager (`Garnet/src/model/kv_cache_manager.cpp`)
An LLM memory allocator managed entirely by Garnet.
* **Paged Allocation:** Pre-allocates a massive pool of GPU memory divided into discrete "Pages" (e.g., 16 or 32 tokens per page).
* **Block Tables:** Maintains a CPU-side hash map tracking which physical GPU pages belong to which sequence/user request.
* **Token Slicing Kernel:** A custom CUDA kernel that takes the $K$ and $V$ tensors emitted by the TensorRT engine and writes them into the correct non-contiguous pages in VRAM.

### D. FlashAttention-2 Integration (`Garnet/src/cuda/flash_attn_kernel.cu`)
Garnet will directly compile the open-source FlashAttention-2 (FA2) kernels.
* **Execution:** Garnet will expose a high-level `T.flash_attention2()` operator in the Open Ops registry.
* **Memory Binding:** The FA2 kernel will read directly from Garnet's non-contiguous Paged KV Cache pointers.

---

## 5. The Runtime Flow (Autoregressive Generation)
During the Python `forward` loop, execution bounces rapidly between the TensorRT Engine and Garnet's custom kernels without leaving the GPU:

1. **TensorRT Execution (Phase 1):** Garnet feeds the new token into the `qwen_trt.engine` to compute the $Q, K, V$ projections and apply RMSNorm.
2. **Garnet JIT (RoPE):** Garnet applies Rotary Positional Embeddings to $Q$ and $K$.
3. **Garnet Memory (KV Update):** Garnet allocates a new token slot in the `KVCacheManager` and writes the new $K/V$ values to the Paged VRAM.
4. **Garnet Execution (FA2):** Garnet launches the custom FlashAttention-2 kernel, reading $Q$ from TRT and $K, V$ from the Paged Cache.
5. **TensorRT Execution (Phase 2):** Garnet feeds the FA2 output back into the `qwen_trt.engine` to compute the SwiGLU MLP and final Logits.
