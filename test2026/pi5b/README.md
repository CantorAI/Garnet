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

## Optimization profiles

Keep the original 65,536-pixel image budget for quality. The optional fifth
preprocessor argument selects a pixel budget, for example `1 16384`; this is
an experimental detail/speed tradeoff, not an equivalent-quality profile.
The 16,384-pixel sample changed a headband into a reported glowing orb.

For the faster execution profile, explicitly set both:

```sh
GARNET_OPENVINO_CPU_THREADS=2 GARNET_OPENVINO_CPU_INFERENCE_PRECISION=f16 \
  venv/bin/python source/test2026/pi5b/measure_process.py \
  /home/shawn/cantorai-ci/out/build/linux-arm64-release/bin/xlang3 \
  source/test2026/pi5b/run_image_decoder.py cpu-runtime-build source \
  models/Qwen3.5-0.8B inputs/RealWorld-04-b1.json
```

The FP32 default remains the numerical reference. The FP16 performance setting
preserved all generated IDs on the measured photo; general accuracy equivalence
is unverified. F32 with four threads was slower than two in the actual image run.
A CPU having four cores does not establish the best thread count for every
small matrix/recurrent graph. See the investigation's optimization logs for
complete observations; cold compile/load times depend on cache residency.

Experimental controls (defaults remain unchanged):
- Fifth inference-driver argument `4` enables memory-bounded chunk prefill;
  compilation overhead made the executed one-shot photo slower.
- `GARNET_CPU_HEAD_THREADS=4` changes only output projection worker count.
- `GARNET_CPU_HEAD_PRECISION=q4q8` enables native quantized output projection;
  the executed caption changed, so treat accuracy as unverified.
- `GARNET_CPU_WEIGHT_PRECISION=int4_fp16` enables experimental OV weight packing.
- Use `GARNET_PI_MAX_RSS_MIB=4096 GARNET_PI_MIN_AVAILABLE_MIB=1024
  GARNET_PI_TIMEOUT_SECONDS=180` on the measurement wrapper. These are polling
  termination guards, not hard memory caps. The earlier resident chunk attempt
  exhausted memory and required a manual reboot.

## CPU architecture scope

The CPU implementation targets x86-64 and ARM64. This directory records the
original Pi investigation; native component tests also run on x86-64.
On 2026-10-11, GCC 13.3 on x86-64 Linux built the Q4Q8, gated-delta and native
decode sources and passed both available numerical CTests. Complete x86
Qwen3.5 vision/OpenVINO package execution remains unvalidated. ARM64 Linux
has actual Pi image-plus-text execution and OpenVINO component tests.
The x86 kernel build requires AVX2/F16C; older CPUs are not validated.
Windows ARM64 is not validated.

Reproduce native x86 validation:
```sh
cmake -S test2026/pi5b -B build/cpu-native-validation
cmake --build build/cpu-native-validation -j4
ctest --test-dir build/cpu-native-validation --output-on-failure
```

Full x86 CPU/OpenVINO build, model parity and server performance still need
validation before production support is claimed. Measure precision and thread
settings per architecture; Pi settings are not server defaults.

## Apple GPU investigation (feature/qwen35-metal)

The sibling Garnet-metal worktree is isolated from the CPU investigation and
the original gpt-oss checkout. Its initial GARNET_METAL build target contains
a native C++/Objective-C++ Metal context and an MPS matrix multiplication
primitive. This is a backend foundation, not a complete Garnet xModel package.
The numerical test covers non-square matrices, transposed weights, padded MPS
row strides, and invalid dimensions against an independent CPU reference.

On the actual Mac, run:
```sh
cmake -S . -B build/metal -DGARNET_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/metal -j4
ctest --test-dir build/metal --output-on-failure
```

On 2026-10-11 the target built with AppleClang 17.0.0 on the reachable M4 Mac
mini (16 GiB RAM, Xcode installed). The actual Metal/MPS numerical CTest passed
(1/1, 2.59 seconds including test startup). This is a GPU primitive correctness
result, not Qwen model inference or a throughput benchmark.
The Linux platform-rejection check passed: configuring Metal on Linux reports
that macOS and a real Metal GPU are required. No Apple GPU speed is claimed.
The Tailscale address subsequently connected using the previously saved
ZeroTier host key through HostKeyAlias. The connection is intermittent.
No SSH identity checks were disabled.

Next milestones: validate native toolchain/GPU; add persistent device tensors
and captured xModel graph lowering; implement Qwen normalization, positional
encoding, recurrent attention, KV attention and vision operations; execute
text-only then actual image-plus-text with numerical references. Only after
that measure and optimize FP16, fused kernels, batched prefill, command-buffer
submissions and GPU output projection. The initial synchronous matmul helper
is for correctness bring-up, not the per-operation schedule for Qwen.

Follow-up executed on the same Apple M4: persistent FP32 Metal matrices,
multiple MPS operations in one command buffer, reuse across two requests,
pending/aborted output guards, GPU RMS normalization (Qwen weight offset +1),
and fused SiLU multiplication all passed independent reference checks. The
expanded actual GPU CTest passed 1/1 in 1.45 seconds including startup. Shared
storage currently exposes explicit upload/download at request boundaries;
intermediate matrices stay GPU-backed. Contexts are single-thread-owned.
The initial RMS kernel is a correctness implementation with a serial reduction
per row; it requires performance profiling and parallelization. FP16, attention,
checkpoint-backed xModel graph execution and full Qwen inference remain undone.
The existing Mac XLang3 SDK was found under /Users/zsdx/CantorAI/xlang3/sdk;
shared project files/builds remain read-only for this investigation.
