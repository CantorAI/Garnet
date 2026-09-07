# Native Garnet Model Integration

Run these scripts with the built XLang3 executable next to the actual Garnet DLL
and its runtime dependencies. They import `garnet` directly and use no mocks,
test-only packages, downloaded weights, or checkpoints.

```powershell
.\xlang3.exe <Garnet>/test/xlang3/models/run_models.py tensorrt <fresh-cache-directory> build
.\xlang3.exe <Garnet>/test/xlang3/models/run_models.py tensorrt <same-cache-directory> reload
.\xlang3.exe <Garnet>/test/xlang3/models/run_contracts.py tensorrt <contract-cache-directory>
.\xlang3.exe <Garnet>/test/xlang3/models/run_source_isolation.py tensorrt <fresh-work-directory>
```

Replace `tensorrt` with `openvino` to exercise OpenVINO. An unavailable backend
fails; tests do not silently skip it. `run_models.py ... all` performs build and
reload in one process. Separate invocations above additionally verify disk-cache
reload after process exit. Keep each backend's cache directory separate.

Coverage includes decorated add/matmul/registered-ReLU graphs, real engine
compilation and execution, changed inputs, retained output ownership, execution
plan capture, disk-cache hits without recapture, invalid request/profile/dtype
contracts, same-named model packages in different directories, and dependency
source edits in a running process.

The TensorRT driver also runs `run_reuse.py` (1,000 calls retaining one output
identity) and `run_concurrent.py` (1,212 calls across 12 model instances sharing
an engine). Both check numerical results and clean runtime release/process exit.

Frontend tests cover rejection of missing tokenizer assets. They deliberately do
not claim successful Qwen preprocessing or model inference without those assets.
Run from an output directory without unrelated `tokenizer.json` files. Full Qwen
inference requires the separately supplied model assets.

The scripts leave engines and generated isolation fixtures in the supplied
directories for inspection. A success marker is printed only after all assertions
for that script pass.

`run_production_capture.py` additionally imports all migrated Qwen modules and
captures their production entry points with tiny symbolic profiles. See
[production capture coverage](production-capture.md) for the exact verified
scope and backend lowering gaps found. It does not compile pretrained models.

## Pretrained Text Generation

`run_qwen_text.py` runs the full 28-layer Qwen3-1.7B BF16 model directly in
XLang3. Supply its original checkpoint shards, index, config and tokenizer files:

```powershell
$env:GARNET_OPENVINO_DEVICE = 'CPU'
.\xlang3.exe <Garnet>/test/xlang3/models/run_qwen_text.py openvino <weights-directory> <fresh-cache-directory>
.\xlang3.exe <Garnet>/test/xlang3/models/run_qwen_text.py tensorrt <weights-directory> <different-cache-directory>
```

This asset-dependent test is separate from the ordinary CTest suite. It checks
the fixed prompt's token count, the independent reference's first token and text
prefix, and successful generation of up to 12 tokens. It does not establish
full-sequence numerical parity or throughput parity with the old runtime.
It repeats the request to check that previous KV state does not contaminate
the result. Run it again with the same cache directory to exercise engine reload
in a fresh process.
