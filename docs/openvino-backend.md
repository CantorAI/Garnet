# OpenVINO backend

Garnet can lower the same captured XLang TensorGraph used by TensorRT to the
native OpenVINO runtime. Model `.x` files remain backend-neutral; the runtime is
selected when `garnet.load_model` compiles the graph:

```python
model = garnet.load_model(
    "model.x",
    runtime_mode="compiled_xmodel",
    backend="openvino",
    precision="fp16",
    entry_function="Model",
    input_shapes=[[1, 4]],
)
```

`precision` may be `bf16` (the default), `fp16`, or `int4_fp16`. It is a
runtime compilation profile, not a model declaration: `.x` files remain
unchanged. `int4_fp16` compresses eligible transformer linear weights to INT4
while retaining FP16 activations, norms, embeddings, LM head, and KV cache.
The selected precision is included in Garnet's graph-cache fingerprint.

The default device is `CPU`. Set `GARNET_OPENVINO_DEVICE` to an OpenVINO device
name such as `GPU.0` to compile for another available device. The device name is
part of Garnet's graph-cache fingerprint, so a compiled blob is never imported
on a different device accidentally.

## Build integration

OpenVINO is optional. CMake first checks a configured system/archive SDK. If it
is not configured, CMake discovers the native SDK shipped in an installed
OpenVINO Python wheel. Builds without an SDK retain TensorRT support and return
an explicit `backend_not_available` error when OpenVINO is requested.

On Windows, the build copies the OpenVINO runtime and device-plugin DLLs beside
`garnet.dll`. NVIDIA runtime libraries are delay-loaded, so loading Garnet for
OpenVINO execution does not require TensorRT, cuBLAS, NPP, or nvJPEG to be
installed. A future backend-plugin split should remove those references from
the core DLL entirely.

## Implemented lowering

- Elementwise add, subtract, multiply, and divide
- Matrix multiplication and transposed Qwen-style linear projection
- ReLU, sigmoid, tanh, SiLU, and GELU
- Checkpoint-backed embedding and dense linear operations
- FP32-accumulating BF16 RMSNorm
- Packed Qwen3 gate/up projection with SwiGLU
- Packed Qwen3 QKV projection, Q/K RMSNorm, RoPE, grouped-query causal
  attention, and output-head merging
- Explicit paged key/value cache outputs for prefill
- Per-layer device-resident key/value state for decode; Intel GPU state uses
  FP16 internally and is seeded once from the BF16 prefill cache
- Optional native FP16 compilation profile: BF16 checkpoint constants and
  internal activations are lowered to FP16, while RMSNorm and softmax retain
  FP32 accumulation and external cache tensors retain their BF16 contract
- Persistent, session-isolated OpenVINO inference requests
- Device-side greedy Top-1 selection, so decode returns one token ID instead
  of copying the full vocabulary logits to the CPU
- Full batch-one Qwen3-1.7B prefill and autoregressive decode
- Native compiled-model export/import cache
- CPU tensors at the OpenVINO boundary, including safe copies from a
  GPU-resident Garnet tensor

## Validation

OpenVINO 2026.2.1 was validated on:

- CPU elementwise, matrix-multiply, activation, and compiled-cache tests
- Intel Iris Plus Graphics 650 on a Windows 10 Intel NUC, using the same
  backend-neutral add, matrix-multiply, and activation graphs
- A real Qwen3-1.7B layer-0 RMSNorm, packed gate/up SwiGLU, and down projection
  using the official sharded BF16 checkpoint
- Full Qwen3-1.7B prompt-to-token generation with the 18-token prompt
  `Say: Garnet works.` and 12 greedy decode tokens

The real-weight CPU test measured about 3.5 ms for one `[1, 1, 2048]` token and
matched a PyTorch reference with maximum absolute error `0.003906`.

The remote Iris Plus 650 validation compiled the first graph in about 2.53
seconds and reloaded its compiled engine cache in about 9.69 ms.

The full Qwen3-1.7B generation test produced the same token sequence as an
independent PyTorch Qwen3 reference:

```text
[39814, 0, 5692, 594, 264, 5810, 1616, 311, 1977, 330, 38, 1885]
```

On the Windows 10 Intel NUC with an i7-7567U and Iris Plus 650, the original
host-cache implementation and the optimized device-state implementation
measured:

| Intel GPU path | Warm TTFT | Decode tokens/s | Request time |
|---|---:|---:|---:|
| Original host cache/full logits | 0.73 s | 2.72 | 4.78 s |
| Device KV state/device Top-1 | 0.85–1.00 s | 4.15–4.30 | 3.41–3.60 s |

The optimized path reuses an inference request and keeps all 28 layers of
decode KV state on the Intel GPU. Separate Garnet runtime instances receive
separate OpenVINO requests and variable state while sharing the imported
compiled model.

For context, Ollama on the same NUC measured 22.62 decode tokens/s on CPU for
its `qwen3:1.7b` package. That package is a 1.4 GB Q4_K_M model, whereas this
Garnet validation uses the official 4.06 GB BF16 checkpoint, so this is not a
precision-equivalent backend comparison. It demonstrates that quantized
weight kernels are the dominant remaining optimization rather than a hardware
limitation.

### Native FP16 profile result

The Iris Plus 650 reports FP16 but not BF16 in its OpenVINO optimization
capabilities. Garnet therefore also tested an explicit native FP16 profile.
Both profiles generated the exact reference token sequence above.

Alternating warm runs with separately cached BF16 and FP16 engines measured:

| Profile | Warm decode tokens/s, three-run mean |
|---|---:|
| BF16 graph profile | 3.66 |
| Explicit native FP16 | 3.55 |

The difference is within the run-to-run variability of this older integrated
GPU, but FP16 did not improve performance. This indicates that the OpenVINO GPU
plugin was already selecting FP16-compatible execution for the BF16 graph.
Explicit FP16 remains available for device compatibility and future hardware
measurements, but BF16 remains Garnet's default OpenVINO profile.

### INT4 weight-only profile result

`int4_fp16` uses Garnet's backend-neutral symmetric INT4 quantizer. The CPU
profile uses block 128; GPU profiles use block 64.
OpenVINO receives the equivalent compressed U4-plus-zero-point graph form for
compatibility with its compressed-weight transformations. Quantization runs
only on a cold compile; warm loads import the precision-specific OpenVINO
engine cache.

On the i9-14900K development CPU, the first compressed implementation improved
from about 6.9-7.0 tokens/s in BF16 to 9.8-10.2 tokens/s in INT4. Profiling then
identified the packed gate/up projection as an accidental FP32 materialization.
Keeping gate and up as separate compressed fully-connected operations, using
block-128 CPU weights, and compressing the LM head raised sustained 64-token
decode to 32.83 tokens/s. The real instruction response remained coherent.

The automated same-machine comparison is:

```powershell
python test2026\tests\phase_24_compiled_xmodel_runtime\benchmark_cpu_int4_vs_ollama.py
```

On `SHAWN-SRV-001` (i9-14900K), a 64-token run of the same Qwen3-1.7B prompt
measured Garnet at 31.87 decode tokens/s and Ollama at 9.56 decode tokens/s, a
3.33x ratio.

On ShawnPC002 (i7-7567U), the U4-compatible OpenVINO CPU INT4 path measured
about 8.2-8.8 decode tokens/s and the CPU-specific signed-I4 graph measured
9.09 tokens/s. Profiling showed OpenVINO selecting FP32 accumulation on this
old AVX2 CPU. The opt-in
`GARNET_OPENVINO_CPU_NATIVE_DECODE=1` profile therefore lowers the captured
Qwen3 decode graph to Garnet's channel-wise Q4/Q8 AVX2 operators while keeping
OpenVINO prefill and Garnet-owned paged KV tensors. It does not invoke another
model runtime.

The final real-prompt tests generated the complete coherent response and
measured 20.98-22.26 decode tokens/s over 11-17 decode steps. Warm Garnet token
latency was 42.4-44.9 ms.

The native decode pack is persisted beside the fingerprinted decode engine.
The versioned file validates checkpoint tensor bytes/count, model dimensions,
matrix contracts, and total file size before binding its weights and FP32
runtime scales through a read-only mapping. A missing or invalid file falls
back to quantization and is atomically republished. On ShawnPC002:

- one-time quantization and pack creation: 53.68 seconds
- full cold `load_model` including creation: 61.91 seconds
- isolated mapped engine preparation: 4.21 ms (19.45 ms process wall time)
- full cached `load_model`: 9.08-9.91 seconds, 9.45-second mean
- pack size: 863,130,288 bytes
- mapped-page prefault: about 195 ms on a warm filesystem cache

A four-prompt cold-process comparison used a fresh Garnet process per prompt,
explicitly stopped Ollama before every prompt, selected `keep_alive: 0`, and
verified Ollama released the model afterward. With a 64-token cap, Garnet
averaged 20.264 decode tokens/s and Ollama averaged 19.570 tokens/s, a 1.035x
Garnet ratio. Garnet won three of four prompts. Ollama loaded faster (2.19
seconds mean) and produced the more accurate paged-KV explanation; Garnet
incorrectly referred to training data in that answer and reached the token cap
on two longer responses. Performance and response quality are therefore both
reported rather than treating token throughput as the only acceptance signal.

The Iris Plus 650 test compiled both signed-I4 and U4-zero-point forms, but its
legacy GPU plugin produced non-finite logits for the full model. Garnet detects
this condition and returns `compiled_logits_non_finite`; it does not return a
fabricated token. Use `bf16` on this GPU or `int4_fp16` with the CPU device.
Newer Intel Xe/Arc hardware must be validated independently before enabling
this profile in production.

## Current scope

The completed OpenVINO graph targets batch one with fixed compiled token/cache
profiles. In the regular profile, prefill writes the cache to CPU once to seed
OpenVINO decode state. In the full native CPU-decode profile, Garnet passes and
updates its BF16 paged KV tensors directly and returns only the selected token.

Session-isolated requests make interleaved generations safe, but Garnet's
existing `decode_batch.x` paged-KV graph is not yet lowered as a fused OpenVINO
batch. True continuous batching still requires active-row masking and correct
batched paged-KV gather/scatter in the OpenVINO lowering. It should be measured
as a throughput optimization; it will not close the single-request BF16 versus
Q4 performance gap.
