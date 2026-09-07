# XLang3 Migration Status

The native Garnet module and serving host now build with XLang3 on Windows
using the Release configuration and Visual Studio 2026. The native module is
loaded through the XLang3 SDK, not the old XLang runtime. Successful small-model
integration tests do not establish full pretrained Qwen inference readiness.

## Port Scope

- Native package registration uses the XLang3 C ABI and C++ SDK. Package classes,
  properties, methods, tensor factories, and native instances use real runtime
  values. The separate Garnet serving C API is retained.
- Tensor unary/binary operator factories register with XLang3; fusion decorators
  capture graph operations and invocation regions. The graph adapter retains
  operands, tensor-valued attributes, outputs, and regions without copying tensor
  payloads. Backend lowering consumes that graph and infers operation metadata.
- Compiled-model execution uses explicit host ownership, symbolic SDK tensors,
  graph capture, fusion partitions, TensorRT/OpenVINO lowering, native executor
  bindings, and backend execution. CPU kernel paths remain in use.
- Python counterparts of the Qwen text, VL, ASR, and TTS model programs are under
  `xModel/`. Relative imports resolve inside a fresh private package namespace
  per capture, avoiding collisions between same-named model directories. Model
  release removes its namespace registry entries. Bundled manifests select
  `.py` entrypoints and the 27 paired `.x` programs have been removed. External
  legacy `.x` paths can still select a sibling `.py` when present; this does
  not make arbitrary old XLang syntax Python-compatible.
- Graph fingerprints include model dependencies and package initializers.
  Fingerprinted engine filenames prevent changed source from selecting a stale
  engine cached under a fixed filename.
- CUDA tensor storage uses owned or retained borrowed allocations through the
  SDK. Image preprocessing, Qwen frontends, model management, and serving-host
  calls have been migrated away from the old XHost/raw tensor interfaces.
- The unused MoE prototype in `src/transformer/moe_transformer.h` is excluded
  from the migrated build. It is not a supported, migrated MoE implementation.

FP16/BF16 have packed storage/capture support; this does not imply general CPU
arithmetic support for those formats. Native backend conversion and computation
paths are distinct from XLang3's built-in tensor CPU kernels.

## Verified Tests

The Windows Release integration run passed all 16 registered Garnet CTest tests:

| Test | Verified scope |
| --- | --- |
| `garnet_image_preprocessing` | CPU/GPU pixel parity, offset-view ownership, invalid inputs, JPEG file/bytes parity |
| `garnet_weight_quantization` | Native weight quantization |
| `garnet_q4q8_linear` | Native Q4/Q8 linear kernel tests |
| `garnet_model_catalog` | Model catalog behavior |
| `garnet_continuous_batch_scheduler` | Scheduler tests |
| `garnet_serving_process` | Serving-process smoke test |
| `garnet_models_tensorrt` | Actual native small-model integration through TensorRT |
| `garnet_models_openvino` | Actual native small-model integration through OpenVINO |
| `garnet_tensor_capture` | Tensor provider, graph/fusion capture, storage and ownership tests |
| `garnet_device_kv_cabi` | Host/device pointer KV calls with matching results |
| `garnet_logits_sampler_cabi` | Native logits sampling |
| `garnet_native_api` | Native classes, writable properties, conversions, and CUDA operations |
| `garnet_production_capture` | All 27 Python modules and 40 captured graphs |
| `garnet_serving_python_defaults` | Python entrypoint selection for all five model families |
| `garnet_tensor_readiness` | Delayed cross-thread CPU/GPU consumers and alias lifetime |
| `garnet_trt_context_pool` | 480 calls across 12 threads with three context slots |

Both model suites use the actual `import garnet` module, not mocks or a
test-only package. They compile decorated add, matmul, and registered-ReLU
models, compare numeric outputs, vary input values, retain outputs across calls
and release, and reload disk caches in a separate process without recapture.
They also verify invalid request/profile/dtype contracts, missing-tokenizer
rejection, same-named package isolation, dependency edits within one process,
and namespace cleanup. No weights are downloaded.

The C ABI checks are included in CTest. Separately,
`test/xlang3/native_api.py` also passed when run with the built Release
`xlang3.exe` from its Release output directory, printing:

```text
garnet-native-api-cpu-passed
garnet-native-api-cuda-passed
```

That native API smoke test covers independent native instances and writable
properties, tensor creation/update/conversion, invalid and strided-input
rejection, registered graph operators, and small CUDA tensor operations.

The model test driver `test/xlang3/run_model_tests.ps1` creates a fresh work
directory per run and configures backend dependency paths. TensorRT runtime DLLs
must be discoverable through its SDK `bin` directory on `PATH`; they are not
assumed to reside beside the Garnet DLL. See
[`test/xlang3/models/README.md`](../test/xlang3/models/README.md) for direct test
commands and coverage boundaries.

The additional `run_production_capture.py` test passed against the actual native
module: all 27 migrated Qwen modules imported, and 19 production entry points
captured with one-layer and two-layer configurations (38 root graphs). Two VL
position/rope helper graphs also captured. Tests verify dependency ordering,
unique graph IDs, required root boundaries, atomic-boundary nesting, fresh
invocation IDs, and graph growth when adding a layer. This exposed and fixed
legacy configuration attribute access in the Python programs; configuration
values now use dictionary indexing, matching native JSON deserialization.

These captures contain 58 distinct provider/operator pairs. A static dispatch
audit found all 54 Garnet operators used by the production roots in TensorRT's
implementation, but 32 absent from OpenVINO's dispatch. Three additional VL
position/rope helper operators are absent from both backends. Thus successful
capture is not proof of complete Qwen lowering support, even without pretrained
assets. See [production capture coverage](../test/xlang3/models/production-capture.md).
These missing dispatch entries were also absent at pre-migration Git HEAD
`8698fe6413556a25656de94500e30a92e70fc882`; they are verified preexisting coverage
limits. Adding previously missing backend operators is outside this migration.

The rebuilt tensor-use ABI and dependent consumers passed these tests. Storage
access leases are shared by views, deduplicate aliases, retain asynchronous
completion, and distinguish input reads from writes. CUDA dependencies use
events; CPU consumers wait for completion. TensorRT contexts and captured CUDA
graphs belong to bounded execution slots.

The TensorRT integration suite additionally passed 1,000 reusable-output calls
with stable output identity and 1,212 calls across 12 native model instances
sharing an engine. Retained outputs remained readable after runtime release.
Model-owned engine references and a weak lookup cache avoid CUDA/TensorRT
resource destruction during static DLL teardown. The previously observed
shutdown failure is covered by process exit checks in the model test driver.

All 54 XLang3 CTests also passed after the SDK/runtime changes, including CPU
tensor readiness, interpreter, native packages, shared-memory IPC, serialization,
and CPython bridge regressions.

## Remaining Verification

### Native Request Recovery

Compiled execution validates the exact input count, tensor types and declared
shapes before dispatching to either backend. Extra inputs are not silently
ignored, and invalid shapes cannot be bound to a cached execution context.
Rejected requests leave the model usable for subsequent valid requests.

After `release_runtime()`, compiled models retain their mode, report `released`
and reject `forward()` with `compiled_graph_not_ready`, rather than falling
through to the legacy runner. Runtime resources are still released; previously
returned independent outputs retain their own storage.

The native model tests check these cases on TensorRT and OpenVINO during both
fresh compilation and engine-cache reload, including repeated release. All 17
Garnet CTests pass with this coverage. These tests do not compare against PyTorch.

### Qwen VL Generation

The Windows native Qwen3-VL-2B path has now generated a complete room-image
inventory through the CPython bridge. Optional TensorRT `repetition_penalty`
sampling applies a sign-aware penalty to previously generated tokens using a
device-resident byte mask. It does not copy logits to the CPU or mutate them.
The default is `1.0`: no history allocation and the original greedy kernel.
The option must be finite and at least one; non-default penalties on other
backends are rejected. Prompt tokens are not included in this history.

FP32/BF16 kernel tests cover positive and negative scores, ties, identity,
invalid penalties and unchanged logits. The full 17-test Garnet suite passes,
including native imports, serving, TensorRT/OpenVINO and context concurrency.
An empty data directory no longer hides the Garnet native module during import.

On the tested room image, penalty `1.1` reached EOS after 396 tokens with nine
complete object records. Repeated requests in one process returned identical
tokens after rejecting invalid penalty requests, confirming independent
generation histories. Unmodified greedy generation repeated records and hit
the 1024-token limit. A diagnostic CPython reference terminated with the same
native image pixels, so this result is not a claim of numerical equivalence.
The reference implementation is not used as an inference fallback.

Follow-up shared-prefix analysis identified a precision-sensitive tie at token
index 31: the BF16 reference rounds tokens 22/23 to the same score, while
Garnet's intentionally FP32 head selects token 23. A reference with an FP32 head
selects the same token and also repeats records in free-running greedy generation
until the 1024-token limit. Native/reference logits have cosine similarity
approximately 0.99994 at the compared position with exactly matching input pixels and tokens.
This explains this first divergence, not every possible numerical difference.
See `test/xlang3/models/QWEN_PARITY.md` for reproduction and measured scores.
Production inference precision is unchanged.

- Full pretrained Qwen text/VL/ASR/TTS inference has **not** been verified.
  Local tokenizer/checkpoint assets were subsequently found under the ignored
  `out/component-test-models/models` directory. The earlier asset search missed
  ignored files. `test/xlang3/models/run_qwen_text.py` exercises real BF16 text
  generation directly in XLang3. OpenVINO CPU BF16 passed with the full 28-layer
  Qwen3-1.7B checkpoint: 18 prompt tokens, reference first token 39814, and 12
  generated tokens beginning `Sure! Here's a natural way`. The process exited
  successfully. This is a short generation check, not full numerical parity.
  TensorRT BF16 on RTX 4080 also passed and produced the same 12 token IDs,
  with a clean process exit. The measured first request was approximately
  113 ms (14 ms to first token); these timings are not a baseline comparison.
  Both backends subsequently passed fresh-process engine-cache reload and two
  identical requests per process, with matching token sequences and clean exits.
  Missing-asset rejection is not successful model inference.
- Windows Release is tested. Linux work and validation are deferred; no Linux
  or macOS build/runtime success is claimed here.
- The tests cover the asynchronous and concurrent scenarios described above,
  not every possible device failure or workload. Signed remote model downloads,
  download cancellation, and deployment-specific dependency discovery have not
  been exercised end to end.
- Compatibility with old checkpoints, production-scale model shapes, long
  generation runs, and deployed serving workloads requires further integration
  testing with the actual assets and environment.
