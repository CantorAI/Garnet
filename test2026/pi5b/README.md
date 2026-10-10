# Garnet Qwen3.5-0.8B on Pi 5

Validated on an 8 GB Pi 5 with ARM64 Debian 13, OpenVINO 2026.2.1,
Python 3.13, and the existing XLang3 SDK/shared runtime. All learned inference
runs in Garnet: xModel graph capture, OpenVINO CPU lowering, and the native
unquantized BF16-to-FP32 output projection. Pillow and NumPy prepare pixels
and independent test references; they are not a replacement inference engine.

## CPU-only package

```sh
cmake -S source -B cpu-runtime-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGARNET_CPU_ONLY=ON -DGARNET_CPU_DOTPROD=ON \
  -DXLANG3_SDK_ROOT=/home/shawn/cantorai-ci/xlang3/sdk \
  -DGARNET_XLANG3_RUNTIME_LIBRARY=/home/shawn/cantorai-ci/out/build/linux-arm64-release/bin/libxlang3_runtime.so \
  -DOpenVINO_DIR=/home/shawn/garnet-pi5b-investigation/venv/lib/python3.13/site-packages/openvino/cmake
cmake --build cpu-runtime-build -j1
```

The `garnet` native module exposes `compiled_graph()`, `tensor_io()`,
`tokenizer()`, and `shutdown()`. Close every graph, then call
`garnet.shutdown()` before host shutdown; shutdown rejects live graphs.
The CPU backend shares one OpenVINO Core while the process is running.
This target does not expose the existing CUDA-dependent serving API.
Default root CMake behavior remains unchanged when GARNET_CPU_ONLY is off.

## Actual image inference

Run from `/home/shawn/garnet-pi5b-investigation`. Download the pinned checkpoint
with `download_full_checkpoint.py`, which verifies its upstream LFS SHA256.
The model revision, image URL/hash, full results, and limitations are recorded
in `docs/investigations/qwen35-vl-pi5b.md` in the local investigation branch.

```sh
venv/bin/python source/test2026/pi5b/prepare_cpu_image.py \
  inputs/RealWorld-04.png models/Qwen3.5-0.8B inputs/RealWorld-04-b1.f32 1
GARNET_OPENVINO_CPU_THREADS=2 venv/bin/python source/test2026/pi5b/measure_process.py \
  /home/shawn/cantorai-ci/out/build/linux-arm64-release/bin/xlang3 \
  source/test2026/pi5b/run_image_decoder.py cpu-runtime-build source \
  models/Qwen3.5-0.8B inputs/RealWorld-04-b1.json
```

For batch 2, use `b2` in artifact names and `2` as the preprocessing argument.
Both rows in this benchmark use the same real image and prompt. The graph
supports independent image rows with equal grid dimensions. Decoder request
states and active masks are independent; numerical fixtures cover pause/resume
isolation. The driver uses capacity 256 and a bounded 65,536-pixel image profile.
Vision weights are released before decoder loading. Model reload/compile and
cold SD reads are included in reported startup, separately from warm decode.

## Validation scope

CTest under `cpu-build` validates native kernels and captured recurrence,
convolution, linear/full attention, and complete decoder blocks. Native-package
tests `test_cpu_io_head.py`, `test_vision_graph.py`, and
`test_cpu_image_layout.py` check the CPU API, vision graph, and exact pixel layout.
`test_pretrained_native_head.py` checks the full-vocabulary native output head.

On this Pi's glibc 2.41, the separately linked component-test host intermittently
fails in the loader's DFS DSO sorting during shutdown, after numerical checks
pass. Core reuse and explicit cleanup alone do not eliminate that host failure.
The documented per-process compatibility setting makes its repeated checks pass:

```sh
GLIBC_TUNABLES=glibc.rtld.dynamic_sort=1 ctest --test-dir cpu-build --output-on-failure
```

This affects that test process only; it does not update system libraries or
change numerical tolerances. The actual native Garnet package passed ten
repeated API runs without that setting. Do not claim the embedded host is
reliable under the default loader setting from those native-package results.

`test_pretrained_vision_graph.py` is an additional strict endpoint diagnostic;
its unchanged rtol 5e-5/atol 5e-6 currently fails on small accumulated differences
in the 12-block FP32 comparison. Do not report it as passing or claim full
cross-framework parity. Actual photo generation, component verification, and
that stricter diagnostic are reported separately. General Unicode tokenizer
parity, fast-processor pixel parity, long contexts, video, batch 4, efficient
chunked prefill, and a persistent serving API are not established by these tests.
