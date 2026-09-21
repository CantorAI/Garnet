# Garnet integration examples

These examples show the smallest supported host integrations for Garnet. They
all perform a weight-free runtime and hardware check. When the model environment
variables are set, they also load `Qwen3-1.7B` and run one real inference.

Garnet does not download model or acceleration packages. The host supplies
trusted local paths.

## Environment

On Windows PowerShell:

```powershell
$env:GARNET_RUNTIME_ROOT = "D:\path\to\runtime\bin"
$env:GARNET_MODEL_ROOT = "D:\path\to\Qwen3-1.7B"
$env:GARNET_XMODEL_ROOT = "D:\path\to\Garnet\xModel\qwen3\text_1_7b"
$env:GARNET_CACHE_ROOT = "D:\path\to\garnet-cache"
$env:GARNET_ACCELERATION_ROOT = "D:\path\to\verified-acceleration" # optional
$env:GARNET_PROMPT = "Explain why model programs should remain inspectable."
```

Only `GARNET_RUNTIME_ROOT` is needed for the CPython and C++ runtime checks.
The direct XLang3 example is normally launched from the runtime directory, so
it can import `garnet` without that variable. Model variables are optional; if
`GARNET_MODEL_ROOT` is absent, each example stops after the hardware check.

## XLang3 program

The `.py` file uses XLang3's Python-compatible syntax and runs in XLang3, not
CPython:

```powershell
Push-Location $env:GARNET_RUNTIME_ROOT
./xlang3.exe D:\path\to\Garnet\examples\xlang3\main.py
Pop-Location
```

## CPython bridge

The CPython example uses the XLang3 bridge to import the same native Garnet
package:

```powershell
python examples/python/main.py
```

`GARNET_RUNTIME_ROOT` must contain the XLang3 CPython bridge and `garnet.dll`.

## Native C++

The C++ example embeds the public XLang3 runtime API:

```powershell
cmake -S examples/cpp -B build/examples-cpp `
  -DXLANG3_ROOT=D:/path/to/xlang3
cmake --build build/examples-cpp --config Release
./build/examples-cpp/Release/garnet_cpp_example.exe
```

Run the executable with the Garnet runtime directory available on the platform
library search path. The example also adds `GARNET_RUNTIME_ROOT` as an XLang3
import root.

## Electron

The Electron example requires the CantorAI XLang3-enabled Electron runtime. A
stock Electron build does not provide `electron/main`'s `xlang` bridge.

```powershell
Push-Location examples/electron
electron .
Pop-Location
```

All examples use the same public Garnet functions:

```text
detect_acceleration_json()
activate_acceleration_path_json(path)
serve_model(model_root, xmodel_root, cache_root, profile_json, model_id)
infer_json(prompt, image_or_empty, max_new_tokens)
stop_serving()
```
