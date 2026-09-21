# Getting Started with Garnet

This guide builds Garnet, validates its compiler/runtime path, runs a small
weight-free xModel, and then shows how to run a pretrained model or begin an
xModel of your own.

## What You Are Building

Garnet is loaded into XLang3 as a native package. XLang3 executes xModel `.py`
source with symbolic tensors; Garnet captures the Tensor Expressions, lowers
the resulting TensorGraph, and executes a compiled backend artifact.

The shortest useful validation path is:

```text
xModel .py source
  -> XLang3 expression execution
  -> TensorGraph capture
  -> Garnet backend lowering
  -> compiled engine execution
```

The repository does not bundle model checkpoints, CUDA, TensorRT, OpenVINO, or
other vendor runtimes.

## Current Validation Target

The currently documented end-to-end build is Windows Release with:

- a Visual Studio C++ toolchain;
- CMake and Ninja;
- the CUDA Toolkit;
- TensorRT 10;
- an adjacent XLang3 checkout;
- optionally, an OpenVINO C++ runtime or Python package.

The CMake project also contains Linux support, but it does not yet have the
same documented validation coverage. Garnet currently requires CUDA and cannot
be built natively on macOS.

## Workspace Layout

Keep Garnet and XLang3 as sibling repositories. TensorRT may be anywhere when
its path is passed explicitly, although this layout matches Garnet's defaults:

```text
workspace/
  Garnet/
  xlang3/
  ThirdPartySDK/
    TensorRT/
```

From an empty workspace:

```powershell
git clone https://github.com/CantorAI/xlang3.git
git clone https://github.com/CantorAI/Garnet.git
```

Install the CUDA Toolkit and unpack TensorRT separately. Confirm that the
TensorRT directory contains `include/NvInfer.h`, `lib/nvinfer_10.lib`, and its
runtime DLLs under `bin`.

## Configure and Build

Run these commands from a Visual Studio x64 developer shell. Adjust
`$tensorRtRoot` if TensorRT is not in the default sibling location.

```powershell
$workspace = (Get-Location).Path
$garnetSource = Join-Path $workspace "Garnet"
$xlangSource = Join-Path $workspace "xlang3"
$tensorRtRoot = Join-Path $workspace "ThirdPartySDK/TensorRT"
$buildRoot = Join-Path $garnetSource "build"
$runtimeRoot = Join-Path $buildRoot "bin"

cmake -S $garnetSource -B $buildRoot -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$runtimeRoot `
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$runtimeRoot `
  -DXLANG3_ROOT=$xlangSource `
  -DGARNET_TENSORRT_ROOT=$tensorRtRoot

cmake --build $buildRoot --target garnet xlang3 garnet_tensor_tests
```

OpenVINO is optional. Garnet first looks for a configured OpenVINO CMake
package; if that is absent, configuration also checks an OpenVINO Python
installation for its bundled C++ package.

## Validate Graph Capture

The capture tests do not require pretrained weights. They verify symbolic
Tensor Expressions, fusion regions, graph dependencies, structured outputs,
and production xModel source capture.

```powershell
ctest --test-dir $buildRoot -C Release `
  -R "garnet_(tensor_capture|production_capture|native_api)" `
  --output-on-failure
```

Successful production capture proves that xModel source imports and produces a
valid graph. It does not by itself prove that every captured operator is
implemented on every backend. See the
[production capture coverage](../test/xlang3/models/production-capture.md) for
the exact boundary.

## Run a Weight-Free Compiled xModel

`run_models.py` compiles and executes small add, matrix-multiply, and registered
operator graphs. It checks numerical results, disk-cache reload, changed input,
output ownership, and engine reuse without downloading model weights.

Put the vendor runtime directories on `PATH`, then run from the directory that
contains `xlang3.exe` and `garnet.dll`:

```powershell
$env:PATH = "$tensorRtRoot/bin;$env:CUDA_PATH/bin;$runtimeRoot;$env:PATH"
$cacheRoot = Join-Path $buildRoot "model-cache/tensorrt"

Push-Location $runtimeRoot
./xlang3.exe "$garnetSource/test/xlang3/models/run_models.py" `
  tensorrt $cacheRoot all
Pop-Location
```

The first run builds backend artifacts. Later runs reuse the cache. A requested
backend that is unavailable or cannot lower an operator fails explicitly.

If OpenVINO was enabled at configure time, use a separate cache directory:

```powershell
$env:GARNET_OPENVINO_DEVICE = "CPU"
$openVinoCache = Join-Path $buildRoot "model-cache/openvino"

Push-Location $runtimeRoot
./xlang3.exe "$garnetSource/test/xlang3/models/run_models.py" `
  openvino $openVinoCache all
Pop-Location
```

## Run Qwen3-1.7B

Download the official `Qwen/Qwen3-1.7B` checkpoint separately. The supplied
directory must contain its config, tokenizer, safetensors index, and checkpoint
shards. These assets retain their original licenses and are not part of Garnet.

```powershell
$weightsRoot = "C:/models/Qwen3-1.7B"
$qwenCache = Join-Path $buildRoot "model-cache/qwen3-1.7b-tensorrt"

Push-Location $runtimeRoot
./xlang3.exe "$garnetSource/test/xlang3/models/run_qwen_text.py" `
  tensorrt $weightsRoot $qwenCache
Pop-Location
```

The runner checks tokenization, a reference first token and text prefix,
generation, request-state isolation, and cache reload. It intentionally does
not claim full-sequence parity or cross-framework throughput parity. See the
[native model integration tests](../test/xlang3/models/README.md) for the exact
contract.

## Create an xModel

An xModel package separates model semantics from execution policy:

```text
xModel/my_family/my_model/
  model.json
  prefill.py
  decode.py
  shared_model_code.py
  profiles/
    tensorrt_bf16.json
    openvino_cpu.json
```

The `.py` files are XLang3 Tensor Expression programs written with
Python-compatible syntax:

```python
import garnet

T = garnet.tensor()


def linear(x, weight_name):
    return x * T.unary_op("linear", weight_name=weight_name)


@T.fusion(name="example_prefill", role="transformer_prefill",
          boundary="required")
def ExamplePrefill(input_ids, weights, config):
    hidden = input_ids * T.unary_op(
        "embedding", weight_name="model.embed_tokens.weight")
    return linear(hidden, "lm_head.weight")
```

The expression operators are lazy. XLang3 records their operands and Garnet
attributes; Garnet later replays the captured graph into the selected lowering
context. Do not import TensorRT, OpenVINO, or CUDA APIs into xModel source.

For each new xModel:

1. Declare entrypoints and external assets in `model.json`.
2. Keep backend and precision choices in profile JSON files.
3. Add a source-import and graph-capture test.
4. Verify every required operator lowers explicitly on the intended backend.
5. Add numerical checks with real weights before claiming pretrained support.
6. Record hardware, precision, shapes, warmup, and cache state for benchmarks.

Use [Qwen3-1.7B](../xModel/qwen3/text_1_7b/model.json) as a complete package
example and [tensor_graph.py](../test/xlang3/tensor_graph.py) as a compact graph
capture example.

## Troubleshooting

### Garnet cannot find XLang3

Pass `-DXLANG3_ROOT=<path>` to a sibling XLang3 checkout. Garnet builds against
the public XLang3 SDK and native runtime target.

### TensorRT headers or libraries are missing

Pass `-DGARNET_TENSORRT_ROOT=<path>` and verify the `include`, `lib`, and `bin`
subdirectories. Runtime DLLs must be discoverable when `garnet.dll` is loaded.

### The first run is slow

TensorRT optimization can take minutes for a full model and static shape. Keep
the engine cache. Source, dependency, profile, or shape changes intentionally
invalidate incompatible cached artifacts.

### Capture passes but backend compilation fails

Capture accepts registered semantic operator names. Backend compilation still
requires a lowering implementation for every reached operation. This is an
expected and explicit boundary, not an automatic fallback.

## Next References

- [xModel format and current programs](../xModel/README.md)
- [Production capture coverage](../test/xlang3/models/production-capture.md)
- [Qwen3-1.7B performance baseline](qwen3-1.7b-performance.md)
- [Qwen3-VL performance baseline](qwen-vl-implementation/performance-baseline.md)
- [OpenVINO backend notes](openvino-backend.md)
