# Qwen-VL Operator Coverage Plan

## Purpose

This file tracks what Garnet must support to run Qwen3-VL.

Status values:

- `planned`: known requirement, not implemented
- `debug`: can run in a slow/debug path
- `native`: implemented with Garnet tensor expression/codegen
- `trt`: lowered to TensorRT
- `cuda`: custom CUDA kernel path exists
- `blocked`: design or dependency blocker

## Vision Tower

| Operator / Block | Need | First Path | Target Path | Status |
| --- | --- | --- | --- | --- |
| patch embed Conv3D | image/video patch projection | TensorRT/open op | TensorRT/custom CUDA | planned |
| vision LayerNorm | pre/post norm | Garnet native | fused CUDA/TensorRT | planned |
| vision QKV projection | attention projection | TensorRT/Garnet matmul | TensorRT | planned |
| vision RoPE | position encoding | open op | custom CUDA | planned |
| vision varlen attention | variable visual token attention | open op | FlashAttention-style CUDA | planned |
| vision output projection | dense projection | TensorRT/Garnet matmul | TensorRT | planned |
| VisionMLP fc1/fc2 | dense MLP | TensorRT/Garnet matmul | TensorRT/fused | planned |
| GELU | MLP activation | Garnet native/open op | fused | planned |
| patch merger | visual token downsample/project | open op | custom CUDA/TensorRT hybrid | planned |
| DeepStack features | selected vision layer taps | xlang branch/list path | native scheduler support | planned |

## Multimodal Adapter

| Operator / Block | Need | First Path | Target Path | Status |
| --- | --- | --- | --- | --- |
| visual placeholder detection | find image token slots | Python/debug | native/CUDA | planned |
| visual embedding scatter | merge visual embeddings into text sequence | open op | custom CUDA | planned |
| MRoPE index builder | multimodal position ids | Python/debug | custom CUDA/native | planned |
| token embedding gather | text embedding | Garnet gather | TensorRT/custom CUDA | planned |

## Text Decoder

| Operator / Block | Need | First Path | Target Path | Status |
| --- | --- | --- | --- | --- |
| RMSNorm | decoder norm | Garnet native/open op | fused CUDA | planned |
| q/k/v projection | attention projection | TensorRT/Garnet matmul | TensorRT | planned |
| q/k norm | Qwen attention norm | Garnet native/open op | fused CUDA | planned |
| text RoPE/MRoPE | position encoding | open op | fused attention CUDA | planned |
| KV cache update | write k/v blocks | debug/open op | paged KV CUDA | planned |
| prefill attention | prompt attention | open op | FlashAttention | planned |
| decode attention | token attention | open op | paged attention | planned |
| output projection | dense projection | TensorRT/Garnet matmul | TensorRT | planned |
| gated MLP | gate/up/down projection | TensorRT/Garnet matmul | TensorRT/fused CUDA | planned |
| SiLU | gated activation | Garnet native/open op | fused | planned |
| elementwise multiply | gated MLP combine | Garnet native | fused | planned |
| LM head | logits | TensorRT/Garnet matmul | TensorRT/top-k fused later | planned |

## Runtime Operators

| Operator / Block | Need | First Path | Target Path | Status |
| --- | --- | --- | --- | --- |
| safetensors loading | weights | Python/C++ loader | C++ loader | planned |
| dtype cast | fp32/fp16/bf16 | Garnet native | fused where possible | planned |
| shape/view/reshape | graph shape ops | xlang metadata | no-copy metadata | planned |
| slice/index/select | token/image selection | debug/native | custom CUDA if hot | planned |
| concat | sequence assembly | debug/native | custom CUDA | planned |
| sampling | decode token select | Python first | CUDA/top-k later | planned |

## First Numeric Parity Targets

Priority order:

1. `Qwen3TextMLP`
2. `VisionMLP`
3. `rms_norm`
4. `layer_norm`
5. visual/text merge
6. text q/k/v projection and RoPE
7. one decoder layer prefill
8. one decode step with KV

## Coverage Rules

Every operator should eventually have:

- source `.x` location
- input/output shape rules
- backend choice
- numeric parity test
- trace event name
- Workbench display metadata
