# Raspberry Pi 5 Qwen3.5 investigation

Date: 2026-10-10 (America/Los_Angeles)
Branch: feature/qwen35-vl-pi5b
Baseline main: bb5870fc5a20d5a8b32a9f1159a5677c65b8c134
Checkout: D:/CantorAI/Garnet-pi5b

## Execution evidence

Command: ssh -o BatchMode=yes -o ConnectTimeout=10 shawn-pi5 "uname -m"
Result: exit 255 from SSH (shell invocation returned failure); hostname shawn-pi5 could not be resolved.
No remote command executed. No Pi hardware/software measurements or inference results exist.
Requested reachable Pi hostname/IP and SSH username. Do not infer connection details from unrelated hosts.

## Main-branch source audit

- CMakeLists.txt declares CXX and CUDA languages and requires CUDA/CUDAToolkit.
- Core tensor helpers, model/frontends, image preprocessing, paged KV, batch executor, and entry points reference CUDA directly. Making CUDA optional requires source separation and host execution boundaries, not just a CMake option.
- src/backends/cpu/q4q8_linear.cpp includes immintrin.h and requires AVX2. ARM kernels or guarded exclusion are necessary.
- compiled_model_runtime.cpp accepts tensorrt and openvino backend names; OpenVINO partitioned execution is explicitly unimplemented.
- Existing xModel families include qwen3 and gpt_oss; no Qwen3.5 package was found.
- Sibling xlang3 exists and is a C/C++ project. Use separate build output directories and do not edit its shared working tree.

## Intended milestones

1. Inspect Pi OS/architecture/RAM/swap/storage/CPU features/thermals/toolchain/OpenVINO.
2. Establish dependency-free CPU base and ARM64 packaging; verify small real numerical graphs.
3. Implement Qwen3.5-0.8B xModel semantics and backend operator coverage against a trusted reference.
4. Design fixed batch 1 and 2 request state, KV, positions, masks, and independent stopping; validate both.
5. Execute actual image-plus-text inference. Record image encoding, prefill, TTFT, decode tokens/s, aggregate throughput, peak memory, thermals, outputs, command lines, versions, and limitations.
6. Tune precision/threads/image tokens/cache reuse; evaluate ARM-native Q4/Q8 kernels. Evaluate 2B and batch 4 only with measured capacity.

## Status

Investigation started. Pi access is unresolved. No successful build or model execution is claimed. Original D:/CantorAI/Garnet checkout was not modified.

## Pi connection and inventory (2026-10-10)

Connection succeeded using saved alias shawn-pi5 with HostName=192.168.1.157 and User=shawn, from WorkSense/doc/tool_use_memo.md. No credential values are recorded.

Verified Pi 5 Model B Rev 1.0; aarch64; Debian 13.5; four Cortex-A76 cores up to 2400 MHz. CPU flags include asimd, fphp, asimdhp, and asimddp (NEON, FP16, and integer dot product). RAM 7.9 GiB, about 5.4 GiB available at observation; active swap is 2 GiB zram. /var/swap occupies another 1.5 GiB on disk but was absent from /proc/swaps; do not delete without investigating configuration.

Python 3.13.5, CMake 3.31.6, GCC 14.2.0. System Python module discovery: numpy present; openvino, torch, transformers absent. This does not exclude installations in other virtual environments.

Root filesystem is a 29 GiB ext4 partition on a 29.7 GiB SD device, with about 239 MiB available. No external block storage was listed. Inodes are 34% used: capacity, not inode exhaustion, is the issue. /tmp is a RAM-backed filesystem and is not equivalent to additional persistent storage.

Home storage: cantorai-ci 9.8 GiB (out 6.1 GiB); CantorAI 5.8 GiB (out 2.1 GiB); .local 2.1 GiB; release tests 1.8 GiB. Each source tree contains roughly 1.6 GiB ThirdPartySDK. CI out includes package 1.4 GiB, build/deps/validation roughly 1.3 GiB each, deploy 905 MiB. Compiler and xlang3 processes were live; no cleanup performed.

Temperature 54.3 C; get_throttled=0xe0000, historical flags were present, with low current bits zero at that observation. Sustained-load thermal behavior remains unmeasured.

Requested storage choice: additional storage or review/approval of selected stale build artifacts. No dependency installation or model download attempted with root storage almost full.

## Authorized storage cleanup

User authorized cleanup of safe candidates. Immediately before deletion, inspected accessible process cwd/executable/open-file/memory-map references and verified exact resolved paths were not symlinks. Deleted only:
- /home/shawn/CantorAI/out/build
- /home/shawn/cantorai-ci/out/validation/official-linux-arm64-fadf41a
- /home/shawn/cantorai-ci/out/validation/official-linux-arm64-665c4fe

Root available space increased from 269,651,968 to 2,490,028,032 bytes (about 2.32 GiB). Active cantorai-ci/out/build, running release-tests/current deployment, source trees, and SDKs were preserved. No process was stopped.

## OpenVINO ARM execution prerequisite (actual Pi execution)

Installed isolated venv at /home/shawn/garnet-pi5b-investigation/venv using python3 -m venv, then venv/bin/python -m pip install --no-cache-dir openvino==2026.2.1. Installed ARM64 cp313 OpenVINO 2026.2.1-21919 and numpy 2.4.6. System Python was not modified.

Ran test2026/pi5b/probe_openvino.py via SSH stdin using that venv. It compiles MatMul+ReLU with [batch,64] input and [64,32] weights, compares against numpy, exports/imports the compiled model, and checks numerical cache reload.

Initial default-precision test failed FP32 tolerance (max absolute difference 0.02911282). Final probe explicitly sets INFERENCE_PRECISION_HINT=f32 and two inference threads; no tolerance relaxation was used.

Verified OpenVINO devices: CPU. Batch 1: compile 37.37 ms, mean inference 0.245 ms, maximum absolute reference error 0. Batch 2: compile 31.80 ms, mean inference 0.268 ms, maximum absolute reference error 0. Export/import numerical checks passed for both. Timings are tiny prerequisite graphs under concurrent Pi CI activity; they do not predict model throughput.

This tests OpenVINO directly, NOT Garnet's lowering/runtime, Qwen execution, KV handling, or image inference. Those remain incomplete. An ARM syntax-only compile of #include <immintrin.h> failed because that x86 header is absent, confirming the native Q4Q8 source requires portability changes.

## Garnet native Q4/Q8 ARM kernel execution

Ported src/backends/cpu/q4q8_linear.cpp in isolated branch: ARM64 excludes x86 intrinsics, converts FP16 scales through ARM half values, and executes packed-nibble dot products with NEON SDOT when compiled with dot-product support. Generic ARMv8 compilation uses a scalar fallback. Existing x86 AVX2 execution remains guarded. Optimized binary targets Cortex-A76 explicitly; deploy only to compatible hardware.

Copied the actual kernel and required headers plus test2026/pi5b/q4q8_arm_smoke.cpp to the isolated Pi investigation directory. Built and ran:
- g++ -O2 -std=c++17 -pthread -mcpu=cortex-a76 q4q8_linear.cpp q4q8_arm_smoke.cpp -o q4q8_neon
- ./q4q8_neon
- g++ -O2 -std=c++17 -pthread -march=armv8-a q4q8_linear.cpp q4q8_arm_smoke.cpp -o q4q8_scalar
- ./q4q8_scalar

Both binaries passed canonical signed INT4 weight reference comparisons and persisted packed-view comparisons for group sizes 32/64/128, with worker counts 1/2/4. Test dimensions: 9 output rows, 128 input columns; inputs include signed integer extrema with exactly representable quantization scales. This establishes kernel correctness for those cases, not all scale patterns, full native decode, batched model support, or Qwen3.5 inference. Full CPU-only Garnet dependency separation remains required.

## Qwen3.5 architecture verification and ARM decode portability

User confirmed Qwen3.5-0.8B as the target. Official config source: https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/config.json . It declares Qwen3_5ForConditionalGeneration, 24 text layers, repeating three linear-attention layers followed by one full-attention layer, hidden size 1024, gated attention outputs, partial/interleaved multimodal RoPE, and a 12-layer vision encoder. Therefore the existing Qwen3 native decoder cannot execute this model merely by changing weights/configuration. Linear-attention recurrent/convolution state and vision semantics need explicit implementation; fixed batches 1 and 2 require separate state for each request.

Removed x86-only intrinsics from the ARM compilation branch of the existing Qwen3 native decoder: scalar ARM RMSNorm, BF16 attention accumulation, and SiLU use the same mathematical operations, while matrix projections use the new ARM Q4/Q8 implementation. Existing x86 paths retained. Compilation on the Pi is being checked; full decode numerical equivalence still requires a checkpoint/fixture test. This is shared CPU portability work, not claimed Qwen3.5 execution.

## Repeatable CPU-component build

Added test2026/pi5b/CMakeLists.txt: a separate validation build compiles the actual Q4/Q8 kernel, native Qwen3 decoder, weight quantizer, and safetensors mapper into garnet_native_cpu with no CUDA, OpenVINO, or XLang3 dependency. This is a reusable CPU-component validation target, not a replacement for the full Garnet package or completion of CPU-only serving.

Command on Pi: cmake -S /home/shawn/garnet-pi5b-investigation/source/test2026/pi5b -B /home/shawn/garnet-pi5b-investigation/cpu-build -DCMAKE_BUILD_TYPE=Release -DGARNET_PI5B_DOTPROD=ON ; cmake --build ... -j 1 ; ctest --test-dir ... --output-on-failure. Configure succeeded. Full build/test was running at this record; outcome must be checked before claiming success. An isolated source subset was transferred, and other source/build trees were not edited.

Separately compiled qwen3_native_decode_engine.cpp to an ARM64 object successfully. Full decode fixture validation remains pending.

Pinned official Qwen3.5-0.8B revision from HF API: 2fc06364715b967f1860aea9cf38778875588b17. Official safetensors payload is 1,746,942,600 bytes; tokenizer.json is 12,807,982 bytes. Only config was downloaded locally; weights are not yet downloaded. Remaining storage must cover weights, runtime, compilation caches, and build outputs.

## CPU-library and Qwen3.5 recurrent primitive verification

GPU-free CMake build completed on the Pi: libgarnet_native_cpu.a and q4q8_arm_smoke. Initial CTest passed 1/1. The native Qwen3 decoder object and library compiled successfully; this does not establish pretrained decode correctness.

Added Garnet::GatedDeltaStep with caller-owned FP32 recurrent state [batch,heads,key_dim,value_dim], FP32 Q/K normalization (epsilon 1e-6), query head scaling, decay, delta update, and output. Optional active-row mask preserves inactive request state and returns zero output. There is no global request state. It is a native primitive pending xModel/OpenVINO integration, causal convolution, projections, gated normalization, full-attention/vision processing, and model execution.

Semantics checked against HF transformers Qwen3.5 recurrent fallback at https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py (l2norm and torch_recurrent_gated_delta_rule). This primitive assumes Q/K already expanded to the same head count as V; raw projection/gate conversion is outside its contract.

Added gated_delta_probe.cpp and independent vectorized NumPy test. Actual Pi CTest: 2/2 passed, total 0.84 s. Reference test: batches 1 and 2, two heads, key dimension 7/value dimension 5, five recurrent steps with random nonzero initial state, differing input/gate values, row inactivity, and zero Q/K. State/output tolerances rtol 2e-5 and atol 2e-6. Tests do not yet cover the complete production-size model or FP16/BF16 graph execution. Existing Q4/Q8 reference test also passed.

## OpenVINO recurrent lowering and xModel component

Added gated_delta_lowering.h using standard OpenVINO FP32 operations; it validates static tensor shapes, produces updated recurrent state/output, and preserves inactive rows through Select. openvino_builder.cpp now binds Qwen3.5 recurrent operands, lowers the step, packs output+state, and supports output/state extraction. No hidden process-wide request state is introduced.

Added host_tensor.h so GARNET_CPU_ONLY builds can accept materialized host tensors and explicitly reject accelerator transfers without including CUDA headers; normal GPU builds retain the existing TensorHelper transfer path.

Actual Pi build/test: gated_delta_openvino_probe compiles/runs the C++ lowering helper through OpenVINO 2026.2.1 with FP32 and two threads. Independent NumPy tests passed for batch 1/2 across recurrence, inactive row isolation, zero Q/K. Pi CTest: 3/3 passed (Q4/Q8, native recurrence, OpenVINO recurrence), total 1.38 s. Full openvino_builder.cpp also compiled on Pi as garnet_openvino_lowering with GARNET_CPU_ONLY=1, GARNET_WITH_OPENVINO=1, public XLang3 SDK at /home/shawn/cantorai-ci/xlang3/sdk, and no CUDA headers/libraries. This object is not yet a linked Garnet serving package.

Added xModel/qwen35/vl_0_8b/gated_delta.py component using backend-neutral tensor expressions, compatible with surrounding nested fusion. It is explicitly not a complete model manifest/entrypoint. Actual desktop xlang3.exe with native Garnet captured the source component and extraction operations for batches 1 and 2 (nine Garnet operations each); topological dependency checks passed. The first direct helper invocation outside the fusion returned a graph rather than tensor; the test was corrected to capture helpers inside an enclosing fusion. No mock Garnet module was used.

Captured xModel execution through the Pi's full Garnet package is still unverified. Causal convolution, projections, gated norms, full attention, vision encoder/frontend, complete model loading/generation, and performance measurements remain outstanding.

## Actual captured xModel execution on Pi

Linked the real Garnet tensor factories, graph capture/replay, OpenVINO builder, CPU library, and the Pi's existing libxlang3_runtime.so in a narrow validation executable. This is a component integration harness, not the full Garnet serving entrypoint. It imports the actual xModel/qwen35/vl_0_8b/gated_delta.py source and captures seven symbolic inputs, then builds/exports/imports/runs the graph on OpenVINO CPU. Batches 1 and 2 execute four recurrent steps; output and updated state are compared with the native reference, with inactive second-row isolation. FP32 and two CPU threads are explicit.

First execution failed with an invalid graph during capture. Diagnosis: GARNET_WITH_OPENVINO was private on the backend target, so the executable compiled a smaller OpenVINOBuilder class without its implementation pointer, causing memory corruption. Making the class-affecting definitions public resolved it. No XLang runtime modification was needed. Actual Pi integration CTest passed 1/1 in 0.26 seconds. This is total test time for a small component, not model latency or tokens/sec.

CPU precision selection now applies both when compiling and importing the OpenVINO graph; the precision environment variable is included in the compiled-model cache fingerprint. Errors identify the failing compilation stage; graph capture errors identify output/replay access. The source runtime fingerprint change has not been compiled in the full CUDA-dependent serving package.

Reproduction: configure the component build with XLANG3_RUNTIME_LIBRARY=/home/shawn/cantorai-ci/out/build/linux-arm64-release/bin/libxlang3_runtime.so and XLANG3_SDK_ROOT=/home/shawn/cantorai-ci/xlang3/sdk; build gated_delta_graph_probe; run ctest --test-dir /home/shawn/garnet-pi5b-investigation/cpu-build -R gated_delta_xmodel_graph --output-on-failure. Existing CI sources/runtime are reused read-only.

The model weights, complete vision/text graph, image preprocessing, and real image-plus-text generation remain outstanding. No end-to-end inference success, memory footprint, first-token latency, or token throughput is claimed.

## Causal convolution and complete linear-attention lowering

Added causal_conv_lowering.h and the corresponding backend-neutral xModel component. Input is one projected token [B,C], weights [C,K], caller-owned history [B,C,K] oldest first, and active rows [B]. The graph shifts by one token, computes depthwise convolution followed by SiLU, and preserves inactive row state. Actual captured xModel execution through Garnet/OpenVINO on Pi passed batches 1 and 2 over seven tokens with nonzero initial history, varying tokens, and pause/resume row isolation. Kernel width 4 matches this checkpoint. Extraction operators expose output and history for the future serving caller.

Added qwen35_linear_lowering.h combining QKV/Z/A/B projections, causal convolution, optional Q/K head repetition, gate transforms, gated-delta recurrence, gated RMS normalization, and output projection. States remain caller-owned and independent per batch row. Backend operation qwen35_linear_step loads its nine checkpoint weights through Garnet's existing SafeTensorsIndex/GetWeight path. xModel/linear_attention.py captures the layer and provides output/convolution/recurrent-state extraction. This is not yet the full decoder or serving entrypoint.

The mathematical lowering executes on Pi and passes an independent NumPy reference at rtol 5e-5/atol 5e-6: batches 1/2, equal and repeated Q/K head configurations, seven varying steps, random nonzero state, pause/resume isolation. Actual xModel capture of the complete layer plus all three extraction operations passes topology/seven-operation checks for both batches. Full captured layer execution with checkpoint weights remains pending; the existing captured runtime test numerically executes convolution and gated delta components only.

An initial layer-reference failure exposed OpenVINO ARM SoftPlus numerical error. Focused native OpenVINO 2026.2.1 FP32 tests over [-4,4] measured maximum absolute error 0.00039702654 and relative error 0.018241782, unchanged by PERFORMANCE versus ACCURACY execution hints. Exp and Sigmoid in the same probe had relative errors below 3e-7. An explicit Log/Exp SoftPlus expression reproduced the error. The production graph now evaluates stable SoftPlus as max(x,0) + 2*(u+u^3/3+...+u^13/13), u=exp(-abs(x))/(2+exp(-abs(x))) <= 1/3. Its analytic truncation error is below 1.05e-8. A separate independent NumPy logaddexp regression spans [-100,100], random values, near zero, and +/-1000 at rtol 1e-6/atol 2e-6; it passed on Pi. The complete layer passed the original tolerances without relaxing them.

Pi CTest after the implementation: 5/5 passed, total 5.37 seconds. Full-layer reference total 3.94 seconds includes process startup and graph compilation across fixtures; it is not production layer latency or token throughput. Latest capture-only rerun passed in 0.32 seconds.

Inspected only the official pinned safetensors header with test2026/pi5b/inspect_checkpoint.py. Revision 2fc06364715b967f1860aea9cf38778875588b17, 59,840-byte header, 488 entries. Confirmed model.language_model.layers.0.linear_attn prefix, in_proj_qkv BF16 [6144,1024], conv1d BF16 [6144,1,4], norm F32 [128], out_proj BF16 [1024,2048]. Header saved remotely in checkpoint-header.json and locally in ignored work/qwen35/checkpoint-header.json. No full model payload has been downloaded. Free Pi root space at this stage: 2,299,809,792 bytes; available RAM approximately 5.8 GB, active zram swap usage 541 MB. Preserve space for artifacts rather than assuming the checkpoint's 1.75 GB is the only required disk use.

Next unresolved work: execute the complete captured layer through checkpoint loading, implement full attention with Qwen3.5 gates/partial multimodal RoPE, vision encoding and CPU image preprocessing, assemble all layers with request state and tokenizer/serving, then execute real image-plus-text inference and measure memory, TTFT, token rate, and batch tradeoffs. Goal remains active and no full model inference success is claimed.
