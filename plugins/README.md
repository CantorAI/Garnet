# Operator plugins

Model-specific kernels and vendor plugin implementations live outside `src/`:

```text
plugins/
  registry.json
  gpt_oss/
    CMakeLists.txt
    module/module.cpp                   # ordinary XLang3 native package
    include/gpt_oss_kernels.h
    cuda/gpt_oss_kernels.cu
    tensorrt/gpt_oss_plugin.{h,cpp}
xModel/gpt_oss/120b/
  model.json
  prefill.py
  decode.py
  decode_batch.py
  gpt_oss_llm.py
  profiles/tensorrt_mxfp4.json
src/model/operator_plugins.{h,cpp}        # imported-module binding and cache validation
src/backends/tensorrt/gpt_oss_*           # TensorRT weight/binding adapter
test2026/gpt_oss/                        # independent parity tests
tools/gpt_oss/                           # build, verification, token driver
```

Reusable tensor expressions, graph capture, KV storage, scheduling, and model
runtime infrastructure remain in Garnet. Qwen's xModel and operator definitions
are unchanged. Additional families can have separate plugin directories.

## xModel requirement contract

Each entrypoint declares requirements with its argument specification:

```python
GARNET_MODEL_SPEC = {
    "arguments": [...],
    "requires": {
        "operator_plugins": [{
            "id": "gpt_oss",
            "module": "garnet_gpt_oss",
            "abi": 1,
            "backend": "tensorrt",
            "operators": [
                "gpt_oss_round_bf16",
                "gpt_oss_apply_yarn_rope_packed",
                "gpt_oss_paged_attention",
                "gpt_oss_moe_mxfp4",
            ],
        }],
    },
}
```

The model declares an ID, XLang3 module name, ABI, backend, and exact required operator names. It
does not choose a filesystem library path or download a plugin. An absent
`requires` field preserves the existing model-loading path and does not open
a native operator module. Invalid, missing, and incompatible requirements fail with
`operator_plugin_requirement_failed` before graph lowering.

## Native-module import and bridge

```python
import garnet
import garnet_gpt_oss

garnet.bind_operator_module(garnet_gpt_oss)
```

The plugin exports XLang3's standard `xlang3_package_abi_version` and `Load`
entrypoints. XLang3 owns native-library import; Garnet does not call
`LoadLibrary` or `dlopen` to load operator plugins. The runtime search path
must contain `garnet_gpt_oss.dll` on Windows or `libgarnet_gpt_oss.so` on Linux.
The normal build places the module beside Garnet.

The module exposes `manifest_json()` and `operator_bridge()`. The latter
returns a native object carrying the versioned `GarnetOperatorModuleBridge`
payload, identified by `garnet.operator_module_bridge.v1`. Garnet validates
its ABI and structure size, reads the operator manifest, registers the backend,
and retains both the imported module and bridge object. Script code never
handles native function addresses. Repeated binding of the same module is
idempotent; unrelated modules and conflicting module identities are rejected.

The shared bridge header is `src/model/operator_module_bridge.h`. Its callbacks
provide the manifest, backend registration, backend factory resolution, and
actual module binary path. The binary digest comes from that loaded module's
path. Explicit import and binding do not require a plugin registry.

For a cached model whose source is not executed, Garnet imports the module named
in its requirement metadata and binds it before engine deserialization. The
resolved execution plan retains this module name. An optional
`plugins/registry.json` maps IDs to module names for older declarations that
omit `module`; `GARNET_OPERATOR_PLUGIN_REGISTRY` selects an alternate mapping.
Explicit imports and GPT-OSS requirements work without this registry. It stores
module names, not library paths. Models without requirements never consult it.

The initial bridge provides the GPT-OSS backend factory; TensorGraph shape
contracts and a small TensorRT weight/binding adapter remain in core. This
is not yet a general interface for third-party shape inference or arbitrary
external graph lowering. GPU kernels and vendor plugin implementations live
in the independently built native module. Qwen operators are unchanged.

## Cache lifecycle

The execution plan records resolved requirements, plugin version, and a digest
of the library. Warm loading verifies and registers these plugins before engine
deserialization. A different library invalidates that engine cache and causes
recompilation. Replacing a library already loaded in a process requires a
process restart; live unloading is deliberately unsupported. Plugin binaries
must be built against the TensorRT/CUDA versions used by Garnet.

`GARNET_BUILD_GPT_OSS_OPS=OFF` excludes the family DLL/SO from the build. Existing
models work without it; a GPT-OSS model reports its missing requirement.

## GPT-OSS operator contracts

All activation outputs are FP32 carriers with explicit BF16 rounding boundaries.
Expert blocks/scales use opaque INT8 carriers for original unsigned checkpoint
bytes; these are MXFP4 E2M1 values with E8M0 scales, not Garnet's signed INT4
weight-quantization format.

- YaRN: packed `[batch, tokens, (q_heads + 2 * kv_heads) * head_dim]`, INT64
  `[batch, tokens]` positions, split-half rotary layout, unchanged V.
- Attention: BF16 KV pages `[layers, physical_pages, page_size, kv_heads,
  head_dim]`; INT32 page table `[batch, logical_pages]`, context lengths, slot
  positions and active flags `[batch]`; learned sink logits `[q_heads]`.
  Prefill accepts equal-length unpadded rows; decode uses one token per row and
  permits different context lengths. Inactive rows never write KV. Requests
  must own disjoint physical pages, valid page IDs, and sufficient capacity.
  Even layers use sliding windows, odd layers use full causal attention. Both
  retain full KV storage in this baseline; there is no rolling-cache eviction.
- MoE: top-k selected-logit softmax, original interleaved gate/up rows, expert
  biases, clipped GPT-OSS SwiGLU, weighted expert accumulation. Weights stay
  compressed and selected matrices are never gathered into a giant temporary.

The CUDA implementation is a correctness baseline. It supports batches but
does not yet use an optimized token-by-expert grouped GEMM. Continuous batch
admission, tensor/expert parallelism, native Harmony frontend integration, and
an HTTP serving API are not claimed by this plugin. The token driver supplies
the host-side generation loop for the first rented-GPU tests.
