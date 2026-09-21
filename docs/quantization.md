# Garnet weight quantization

Quantization is a runtime compilation profile. XLang TensorGraph and model
`.x` files remain backend-neutral:

```python
model = garnet.load_model(
    "prefill.x",
    backend="openvino",
    precision="int4_fp16",
    # ...
)
```

## Canonical INT4 representation

Garnet owns the checkpoint-to-quantized-weight conversion so model-specific
selection and rounding are identical across runtimes:

- symmetric signed two's-complement values in `[-8, 7]`
- round-to-nearest, ties-to-even
- block size 128 for OpenVINO CPU and 64 for the initial TensorRT/GPU
  lowering, along the linear layer input/K dimension
- one FP16 scale per output-row block
- two INT4 values packed per byte
- weight-only quantization; activations and KV cache remain floating point

OpenVINO CPU additionally supports
`GARNET_OPENVINO_CPU_WEIGHT_GROUP_SIZE=-1` for channel-wise INT4 (one scale
per output row). Block 128 remains the default accuracy profile. Channel-wise
compression is an explicit performance profile because its larger group can
reduce model accuracy and must pass the real-prompt generation test.

The full native CPU decode profile uses channel-wise signed INT4 weights and
tensor-wise signed INT8 activation quantization. Its AVX2 operator applies the
exact zero-point correction, keeps FP32 runtime scales, and lowers fused QKV
and gate/up graph projections without taking ownership of the model, tokenizer,
scheduler, or KV cache.

`GARNET_OPENVINO_CPU_SIGNED_I4=1` emits the current NNCF-compatible symmetric
I4 form without a zero-point subtraction. This is faster on the reference laptop's
AVX2-only CPU and remains separately fingerprinted from the U4-compatible
form required by its legacy Intel GPU plugin.

Qwen transformer Q/K/V/O, gate/up/down, and LM-head projection weights are
quantized. Embeddings and normalization parameters remain unquantized.

## Backend lowering

OpenVINO lowers the canonical data to an equivalent U4 compressed constant,
subtracts the fixed zero-point 8, and applies the FP16 block scale. This form
allows OpenVINO to retain compressed weights through compilation.

Gate and up projections remain separate in the OpenVINO graph. Concatenating
their dequantized values prevents the CPU plugin from selecting compressed
fully-connected kernels and materializes a large FP32 matrix per layer.

TensorRT lowers the same canonical data to an INT4 constant and explicit
blockwise dequantization. Full TensorRT 10.x INT4 WoQ execution is restricted
to supported hardware; Garnet fails fast on pre-Hopper GPUs in this backend.
The RTX-oriented runtime is a separate future backend and is not silently
substituted for full TensorRT.

The OpenVINO backend engine cache persists its compiled quantized artifact.
The full native CPU decode profile also persists its backend-ready biased-Q4
weights and FP32 runtime scales beside the fingerprinted decode engine. Warm
processes validate and directly map this file, prefault its pages during model
preparation, and avoid both checkpoint quantization and an 863 MB memory copy.
The graph cache fingerprint includes backend, precision, graph schema, model
inputs, and checkpoint metadata; the pack adds its own version, checkpoint,
shape, matrix-count, and file-length checks before any operator is bound.
