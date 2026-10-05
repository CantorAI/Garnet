# GPT-OSS GPU validation

Work from the CantorAI workspace, keeping Garnet, xlang3, and ThirdPartySDK as
siblings. The feature branch is `feature/gpt-oss-120b`. Do not add model weights
or generated engines to Git.

## Local build

Use an existing XLang3 Release runtime. This avoids rebuilding or changing
XLang3. Run in a Visual Studio x64 developer shell, using a CPython executable
and CMake available on your machine:

```powershell
python Garnet/tools/gpt_oss/build.py `
  --runtime-executable out/build/x64-Release/bin/xlang3.exe `
  --runtime-library out/build/x64-Release/bin/xlang3_runtime.lib `
  --runtime-dll out/build/x64-Release/bin/xlang3_runtime.dll `
  --tensorrt-root ThirdPartySDK/TensorRT `
  --build-dir out/build/gpt-oss-ready --cuda-architectures 89

python Garnet/tools/gpt_oss/verify.py `
  --runtime-dir out/build/gpt-oss-ready/bin `
  --work-dir work/gpt-oss-verification --tensorrt-root ThirdPartySDK/TensorRT
```

The build emits `garnet.dll`, `garnet_gpt_oss.dll`, the runtime executable,
and `plugins/registry.json`. The verification tool writes individual logs and
checks GPU kernels, full synthetic xModel prefill, cached batch decode, cache
reload/refit, plugin rejection cases, and greedy generation against an
independent CPU reference. It also exercises existing Qwen text/VL capture
checks and native CPU/CUDA APIs with a deliberately missing plugin registry.

The original production-capture script has a stale total-module count (27;
the current tree imports 29). The focused compatibility check reuses its
existing text/VL capture functions without changing Qwen files or that test.
ASR/TTS are outside this GPT-OSS compatibility check.

## Rented Linux GPU through SSH

Once a rented instance provides an SSH host, user, port, and key, connect using
those details. Transfer the current source changes and use the same sibling
workspace layout. No SSH host or credentials are embedded in these scripts.

For the first full checkpoint test, use a CUDA GPU with enough measured VRAM
for the packed checkpoint, KV cache, and engine workspace. Host RAM during
TensorRT compilation/refit must also be measured. The pipeline planner supplies
an estimated GPU memory budget; full-model memory use and performance still
require measurement on the target hardware.

Build an XLang3 Linux Release runtime using its repository instructions, then
build Garnet against it. TensorRT 10 headers and libraries are required:

```bash
python3 Garnet/tools/gpt_oss/build.py \
  --runtime-executable /path/to/xlang3 \
  --runtime-library /path/to/libxlang3_runtime.so \
  --tensorrt-root /path/to/TensorRT \
  --build-dir out/build/gpt-oss --cuda-architectures native

export LD_LIBRARY_PATH="$PWD/out/build/gpt-oss/bin:/path/to/TensorRT/lib:$LD_LIBRARY_PATH"
python3 Garnet/tools/gpt_oss/verify.py \
  --runtime-dir out/build/gpt-oss/bin \
  --work-dir work/gpt-oss-verification --tensorrt-root /path/to/TensorRT
```

Alternatively, the operator library builds independently:

```bash
cmake -S Garnet/plugins/gpt_oss -B out/build/gpt-oss-plugin \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native \
  -DGARNET_TENSORRT_ROOT=/path/to/TensorRT
cmake --build out/build/gpt-oss-plugin --parallel 4
ctest --test-dir out/build/gpt-oss-plugin --output-on-failure
```

After synthetic checks pass, download the official **original** checkpoint
layout separately. The Transformers checkpoint layout is not accepted:

```bash
hf download openai/gpt-oss-120b --include 'original/*' --local-dir models/gpt-oss-120b
python3 -m pip install openai-harmony
python3 Garnet/tools/gpt_oss/prepare_prompt.py 'Say hello.' work/request.json --max-new-tokens 32
out/build/gpt-oss/bin/xlang3 Garnet/tools/gpt_oss/run_tokens.py \
  models/gpt-oss-120b/original work/gpt-oss-engine-cache work/request.json work/result.json
```

The driver releases prefill before loading decode, avoiding two simultaneous
resident copies of the 120B weights. It accepts Harmony IDs and stop IDs,
generates greedily, and records token IDs and timings. It caps total context at
4096 tokens and output at 256 for the first correctness tests. It does not
start a network service or claim pretrained parity. Compare logits/tokens with
the official implementation, measure memory and latency, then evaluate larger
contexts, batch schedules, and expert-kernel optimization.

Reference architecture and weight layout:
[OpenAI model](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py),
[OpenAI weights](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/weights.py).

The GPT-OSS extension is an ordinary XLang3 native module. xModels import
`garnet_gpt_oss` and call `garnet.bind_operator_module(garnet_gpt_oss)`.
Verification includes import, repeated binding, unrelated-module rejection,
and explicit binding with a deliberately missing registry. Cached engines use
the requirement mapping to import and bind the module before deserialization.
For a standalone plugin build, provide `GARNET_XLANG_RUNTIME_LIBRARY` pointing
to the existing XLang3 runtime import library (Windows) or shared library (Linux).

## Automatic multi-GPU layer placement

`run_pipeline_tokens.py` detects visible GPUs, reads actual packed checkpoint
sizes, budgets local BF16 KV pages and working buffers, and chooses contiguous
layer ranges that fit each GPU individually. Each stage has its own engine,
weights and KV cache. Only activations and request controls cross GPUs. No
Qwen xModel files are changed.

```bash
python3 Garnet/tools/gpt_oss/verify.py \
  --runtime-dir out/build/gpt-oss/bin --work-dir work/gpt-oss-verification \
  --tensorrt-root /path/to/TensorRT --multi-gpu
out/build/gpt-oss/bin/xlang3 Garnet/tools/gpt_oss/run_pipeline_tokens.py \
  models/gpt-oss-120b/original work/gpt-oss-pipeline-cache work/request.json work/result.json
```

The request can include `"device_ids": [0, 1]`, `"memory_fraction": 0.9`, and
`"reserve_mb": 1024`. By default all visible GPUs are used; select at most one
GPU per transformer layer. The builder saves `placement.json`, generated stage
xModels and engines under the hardware/placement cache key. Runtime engine
fingerprints additionally validate source, weights, shapes and plugin binary.
Changing available memory can change the placement; identical placement on
identical hardware reuses the cache.

Backbone APIs: `cuda_devices_json()` reports per-GPU free/total memory,
architecture, PCI bus ID and P2P capability; `cuda_set_device(id)` selects the
calling thread's device and returns its previous ID; `tensor_to_device(t, id)`
performs a checked transfer; `tensor_zeros(shape, dtype)` allocates directly on
the selected GPU. The generic planner/scheduler is `python/garnet_pipeline.py`.
GPT-OSS-specific size estimates and graph instantiation are in
`tools/gpt_oss/pipeline.py` and `xModel/gpt_oss/120b/stage.py`.

This first scheduler executes stages sequentially and fences their per-thread
streams. P2P copies are used where supported; otherwise transfers go through
host RAM. It does not overlap requests, split individual operators, perform
tensor/expert parallelism, or offload model weights. An insufficient-memory
plan fails before engine generation. Estimates include headroom but remain
estimates: full 120B engine memory, build host RAM, pretrained parity and speed
must be measured on the rented GPUs. Prefill engines are released before
decode engines load; the local KV caches are retained.
