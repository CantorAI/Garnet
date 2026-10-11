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

## Captured checkpoint layer execution and Qwen3.5 full attention

Added graph_fixture_probe: a narrow native harness linking Garnet's actual tensor factories, graph capture/replay, safetensors loader, and OpenVINO builder against the Pi's existing XLang3 runtime. It imports the actual xModel source, builds/exports/imports its graph, executes supplied input steps, and checks every output against independent numerical fixtures. This remains a validation executable, not the full serving package.

The captured linear-layer test uses BF16/F32 safetensors fixtures, batches 1/2, equal/repeated QK heads, seven sequential steps, pause/resume isolation, and separate packed/output/convolution/recurrent extraction graphs. It exposed an extraction path not included in the earlier compiled snapshot: an initializer-list-only Reshape helper had been called with a vector. Fixed the operation to construct the OpenVINO shape constant explicitly, rebuilt the current source, and executed all extraction paths successfully.

Added download_checkpoint_subset.py: downloads only selected pinned tensors, verifies each HTTP response is 206 with the exact requested Content-Range, preserves original dtype/shape/data bytes, records SHA256 of each transferred tensor, and publishes the subset only after all ranges succeed. Signed redirect URLs and exception text are never logged. It does not claim independent cryptographic validation against an upstream per-tensor hash. Pinned revision remains 2fc06364715b967f1860aea9cf38778875588b17. Downloaded layer 0 linear_attn: 9 tensors, 21,086,816 bytes; layer 3 self_attn: 6 tensors, 14,681,088 bytes. Full checkpoint still not downloaded.

Actual full-size pretrained layer 0 execution on Pi, via captured xModel and Garnet checkpoint loading, passed four varying input steps for batch 1 and 2, including inactive/resumed row state preservation. Reference uses FP32 arithmetic on decoded official BF16/F32 tensor values; it is not a full Hugging Face BF16-model parity assertion. Original tolerances rtol 5e-5/atol 5e-6 retained. CPU FP32, two threads, OPENBLAS_NUM_THREADS=2. Batch 1 compile/export 234.785 ms, RunCapturedEngine calls [8.730557,8.422610,8.401350,8.498426] ms. Batch 2 compile/export 215.463 ms, calls [12.196272,11.195560,11.001540,11.410951] ms. Engine was prepared/imported before timed calls. These calls include Garnet's output/state handling, exclude test input construction and reference validation, and are not whole-model tokens/sec. Native probe cumulative max RSS: 315,136 KiB after batch 1, 538,864 KiB after batch 2; these include large JSON reference fixtures and validation buffers, so are not isolated model memory measurements.

Implemented qwen35_attention_lowering.h and full_attention.py with per-head Q/query-gate split, Q/K RMSNorm using 1+weight, partial rotary embedding, interleaved THW frequency selection, grouped-query attention, contiguous-prefix causal masking, sigmoid attention-output gate, output projection, caller-owned fixed-capacity KV buffers, and active-row isolation. Caller must validate active slots within capacity. No hidden shared request state. Packed output/K/V extraction operators are supported.

Independent NumPy captured-graph tests execute the full-attention source plus Garnet checkpoint loading and all extraction operators for batch 1/2, grouped and equal KV heads, distinct THW positions, partial-rotation pass-through, random nonzero future cache entries, six steps, paused/resumed rows with independent slots. These fixtures use BF16 checkpoint weights and verify that future/padding entries are masked. All cases passed at rtol 5e-5/atol 5e-6.

Actual full-size pretrained layer 3 execution, eight query heads/two KV heads/head dimension 256/rotary dimension 64/mRoPE sections [11,11,10], passed both batches across four input steps. Validation capacity was 8 tokens; performance at longer contexts remains unmeasured. Batch 1 compile/export 169.158 ms, RunCapturedEngine calls [4.650595,4.625336,4.255093,4.185573] ms. Batch 2 compile/export 167.403 ms, calls [4.822245,4.317074,4.291297,4.619169] ms. Native probe cumulative max RSS 148,736 KiB after batch 1 and 150,288 KiB after batch 2, including validation fixtures. No whole-decoder or image-generation performance claimed.

Reproduce: use download_checkpoint_subset.py with checkpoint-header.json, output checkpoint/layer0.safetensors, prefix model.language_model.layers.0.linear_attn; repeat for layer3 and prefix model.language_model.layers.3.self_attn. Run venv/bin/python test_pretrained_linear_graph.py cpu-build/graph_fixture_probe source checkpoint/layer0.safetensors, then test_pretrained_attention_graph.py with layer3. Environment: OPENBLAS_NUM_THREADS=2, GARNET_OPENVINO_CPU_THREADS=2, GARNET_OPENVINO_CPU_INFERENCE_PRECISION=f32. Paths above are relative to /home/shawn/garnet-pi5b-investigation (scripts under source/test2026/pi5b). Repeatable offline numerical tests run through CTest.

Built all current Pi targets and ran CTest: 7/7 passed, total 11.82 seconds. Final free root disk 2,259,992,576 bytes; temperature 56.2 C. Existing CI/build/service trees were not changed. Source changes remain in isolated Garnet-pi5b branch.

Remaining end state: assemble complete decoder with Qwen3.5's offset RMS norms, MLPs, embeddings and final head; build a usable CPU-only Garnet entrypoint/profile; implement vision graph and CPU preprocessing; supply all pretrained weights without exhausting disk/cache/RAM; execute actual image-plus-text generation and measure complete performance/limits. No full image-plus-text inference has yet occurred. The active goal is unchanged.

## CPU-only native package and complete decoder blocks

Root CMake now supports GARNET_CPU_ONLY=ON through cmake/GarnetCPU.cmake. Built the real shared libgarnet.so on Pi with the existing XLang SDK/runtime, OpenVINO 2026.2.1, and Cortex-A76 instructions. Dependency inspection found OpenVINO, XLang, TBB, and normal system libraries; no CUDA or TensorRT dependency. The native module exposes tensor capture, backend_info(), and compiled_graph().build/run/close with compile and inference timings. This is a usable low-level graph runtime, not yet a complete VLM serving API.

CPU-only graph precision defaults to FP32. A new optional memory-only compilation path preserves the existing export/import default and avoids writing a second large weight blob onto the SD card. It requires explicit caller-owned state tensors and rejects the legacy hidden-state decode cache heuristic. Close explicitly releases cached executions. During the first memory-only fixture run all numerical checks passed but process exit hit a glibc loader assertion; explicit cache release before exit eliminated the observed failure. The precise underlying loader cause is unproven; no system libraries or existing services were changed.

Added complete linear and full-attention decoder blocks: input offset RMSNorm, attention, residual, post-attention offset RMSNorm, SwiGLU MLP, and residual. Inactive rows retain attention state and emit zero hidden output. Independent references cover batch 1/2, nonzero states, paused/resumed rows, and all outputs at the original rtol 5e-5/atol 5e-6. Actual public libgarnet.so loading and captured xModel execution passed synthetic blocks and official pretrained blocks 0 and 3, including their MLP and norm tensors.

Pretrained complete-block timings, FP32/two CPU threads, four sequential steps:

| Block | Batch | Compile ms | Run calls ms |
| --- | --- | --- | --- |
| Linear, layer 0 | 1 | 257.276 | 15.207, 13.916, 13.872, 14.022 |
| Linear, layer 0 | 2 | 301.576 | 18.058, 17.328, 17.817, 17.406 |
| Full attention, layer 3 | 1 | 228.147 | 10.915, 10.233, 10.208, 10.230 |
| Full attention, layer 3 | 2 | 266.232 | 10.847, 10.292, 10.251, 10.495 |

Full-attention test capacity remains eight tokens. These are block calls, not full-model token rates. Child cumulative maximum RSS reached 565,184 KiB and includes large fixtures and numerical validation; it is not isolated model memory. All eight component CTests passed in 12.68 seconds; the separate real-package synthetic and pretrained tests also passed.

Reproduction from the investigation directory: configure source into cpu-runtime-build with GARNET_CPU_ONLY=ON, GARNET_CPU_DOTPROD=ON, XLANG3_SDK_ROOT=/home/shawn/cantorai-ci/xlang3/sdk, GARNET_XLANG3_RUNTIME_LIBRARY=/home/shawn/cantorai-ci/out/build/linux-arm64-release/bin/libxlang3_runtime.so, and OpenVINO_DIR=/home/shawn/garnet-pi5b-investigation/venv/lib/python3.13/site-packages/openvino/cmake. Build with one job. Run OPENBLAS_NUM_THREADS=2 GARNET_OPENVINO_CPU_THREADS=2 venv/bin/python source/test2026/pi5b/test_pretrained_decoder_graph.py /home/shawn/cantorai-ci/out/build/linux-arm64-release/bin/xlang3 source checkpoint cpu-runtime-build.

The full official Qwen/Qwen3.5-0.8B checkpoint is now downloaded at models/Qwen3.5-0.8B/model.safetensors. Pinned revision 2fc06364715b967f1860aea9cf38778875588b17; 1,746,942,600 bytes; upstream LFS SHA256 verified as 04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696. download_full_checkpoint.py validates resumable byte ranges, disk reserve, full hash, and records garnet_checkpoint_revision.json; metadata includes config, tokenizer, processor config, and chat template. It does not log signed redirect URLs. After download root free space was 410,796,032 bytes, available RAM 5,779 MiB, temperature 51.25 C. In-memory compilation is necessary to avoid duplicating the checkpoint on this disk.

Still outstanding: all 24 decoder layers, embeddings/final head, tokenizer and generation integration, vision encoder and CPU image preprocessing, actual image-plus-text inference, and complete memory/TTFT/tokens-per-second/batch measurements. No end-to-end inference success is claimed. The original goal remains active.

## Actual full text decoder execution

Added CPUDecoder in xModel/qwen35/vl_0_8b/cpu_decoder.py. It captures and compiles all 24 actual pretrained blocks through Garnet, owns independent request state, supports fixed batch 1/2 and a bounded KV capacity, and has a step_hidden entry for future vision embeddings. Native tensor_io provides zero buffers, validated row-wise packed-state extraction, BF16 embedding lookup from the checkpoint mapping, and greedy argmax. State splitting avoids converting large recurrent tensors into script lists. A failed partial step requires reset before reuse.

First actual full-checkpoint batch-1 text run on Pi completed successfully, exit 0. Explicit official token IDs represented <|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n. Generated opening tokens [248068,198,31248,11], decoded as <think>, newline, Okay, comma. This establishes full text decoder execution, not image inference or full-Hugging-Face-model parity. All 24 layers and final norm/tied head executed; existing independent block references still establish the narrower numerical checks.

Initial all-OpenVINO FP32-head version: compile_seconds 27.079 (after a slow script tokenizer JSON parse); nine prefill calls [1.2814,0.7299,0.7248,0.7527,0.7430,0.7227,0.7211,0.7528,0.7432] seconds. Four timed subsequent decode calls [0.72345,0.75636,0.72610,0.72667] seconds, approximately 1.36 tokens/second in this short sample. Final head alone approximately 235-240 ms. Process max RSS 6,254,720 KiB, total wall 87.951 seconds, 19,232 major faults. RAM availability fell to 472 MiB and zram usage rose to 1,948 MiB during the run; temperature 64.45 C. This footprint is too tight for adding vision without optimization. The original run is recorded in /home/shawn/garnet-pi5b-investigation/full-decoder-b1.log (later runs use distinct filenames).

Native embedding/state split and offset-norm/tied-head tests passed independent numerical references on Pi. Batch 1/2 decoder block tests also passed with native state extraction checked against every expected packed segment. New CPU tokenizer bindings reuse Garnet's existing QwenTokenizer; the greeting's token IDs are checked explicitly. General Unicode/pretokenization parity remains unverified.

To reduce memory, the next implementation keeps final normalization on OpenVINO and performs the tied output projection directly from the unchanged mapped BF16 embedding weights in Garnet's CPU backend. Cortex-A76 NEON expands BF16 values to FP32 and accumulates FP32; scalar fallback is provided. There is no quantization in this path. This removes the need to retain a compiled FP32 copy of the 248,320 x 1,024 embedding/output table. Full-vocabulary reference and updated full-decoder measurements are required before claiming its benefit.

The native BF16 head now passed both a small complete-logit numerical fixture and a real full-vocabulary batch-2 reference. The latter independently computes all 248,320 logits from the official mapped weights in NumPy chunks, verifies the argmax over the entire vocabulary, and checks 256 sampled logits per row spanning the table. The real-weight head call took 118.947 ms for batch 2 in this focused run. It preserves checkpoint BF16 values and uses FP32 arithmetic, without Q4/Q8 conversion.

Updated full-checkpoint batch-1 execution with native head and Garnet's tokenizer completed, exit 0, and reproduced [248068,198,31248,11]. The greeting tokenizer is checked against its nine known official IDs before generation. Compile_seconds 64.130; first prefill call 21.658 seconds with cold mapped-table reads; remaining prefill calls [0.5485,0.5155,0.5036,0.5137,0.4607,0.5478,0.3907,0.4043] seconds. Subsequent decode calls [0.38885,0.38614,0.45170,0.41151] seconds, approximately 2.44 tokens/sec in the short warm sample. Native head approximately 46-47 ms per batch-1 step. Max RSS 4,651,616 KiB (4.44 GiB), compared with initial 6,254,720 KiB (5.96 GiB). Full process wall 93.228 seconds, 6,463 major faults. Report cold startup separately from warm throughput; these sequential runs occurred with existing CI/services active and different cache residency, and are not a controlled latency comparison. New run log full-decoder-native-b1.log preserves the earlier full-decoder-b1.log.

Reproduction: from /home/shawn/garnet-pi5b-investigation set GARNET_OPENVINO_CPU_THREADS=2 and run venv/bin/python source/test2026/pi5b/measure_process.py /home/shawn/cantorai-ci/out/build/linux-arm64-release/bin/xlang3 source/test2026/pi5b/run_full_decoder.py cpu-runtime-build source models/Qwen3.5-0.8B 1. measure_process.py reports child wall/RSS/page faults without requiring the absent /usr/bin/time package. The only completed generation at this point is text input; no vision success is claimed.

Remaining end state: verified image encoder/preprocessing and image-token/mRoPE integration, actual image-plus-text generation, batch comparison and end-to-end measurements. Goal remains active.

## Full text batch 2 and actual image-plus-text execution

Full text decoder batch 2 completed, exit 0, with both rows producing [248068,198,31248,11]. Warm decode calls [0.71854,0.62741,0.63081,0.64789] seconds: 3.05 tokens/sec combined, 1.52 tokens/sec per row in this short sample. Max RSS 4,808,496 KiB (4.59 GiB). Compile 241.733 seconds; first prefill 73.281 seconds; full process wall 325.318 seconds and 13,589 major faults. Direct process inspection during loading found disk sleep in folio_wait_bit_common/mmc_blk_rw_wait with very little CPU time. Kernel warning journal contained no recent entries. Cold SD loading varied greatly; do not attribute this loading difference solely to batch size. Existing CI/services were left running.

Added the real fixed-grid vision graph in xModel/qwen35/vl_0_8b/vision_encoder.py and qwen35_vision_lowering.h: flattened Conv3d patch projection, bilinearly interpolated learned position embeddings, axial 2D rotary embedding, all 12 bidirectional attention/MLP blocks, pre-shuffle LayerNorm merger, exact GELU in merger and tanh GELU in blocks. Each batch row is an independent still image with equal grid dimensions; video/variable-grid batches are unsupported. Synthetic BF16-weight numerical graphs pass batch 1/2, rectangular/fractional position interpolation, nontrivial RoPE, and image-row isolation at the original rtol 5e-5/atol 5e-6.

The all-pretrained-block small-grid comparison is a known accuracy limitation, not reported as passing: FP32 NumPy reference versus Garnet/OpenVINO has max absolute difference 7.3760748e-6, relative L2 difference 1.8809825e-5, and worst original tolerance ratio 1.2057957 for batch 1. The unchanged strict threshold fails. The reference was corrected to keep GELU scalar constants FP32 rather than accidentally promoting the chain to FP64; no tolerance was relaxed. A focused native OpenVINO GELU probe across [-10,10] found max absolute errors 3.60e-7 for tanh GELU and 3.50e-7 for erf GELU, unchanged by execution accuracy/performance hints. The exact source of the accumulated small endpoint difference remains unproven. Full Hugging Face numerical parity is not claimed.

Actual image input is the official Qwen3.5 model-card example: https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/RealWorld/RealWorld-04.png (210,106 bytes), SHA256 eba5d9b6124fbd3b2a809e1c664886c29d842406134bb4c1b281414d0c7101b9. Visually inspected: large stone human head with gold headband, city buildings, hills, and sky. prepare_cpu_image.py uses Pillow 12.3.0 CPU RGB/bicubic resize, the checkpoint's mean/std [0.5,0.5,0.5], temporal frame duplication, and merge-block patch ordering. Source 1818x1372, resized 288x192, grid [1,18,12], 216 patches of 1536 values and 54 merged visual tokens. Pixel-area cap 65,536 bounds Pi work; factor rounding yields 55,296 pixels. Pixel-exact parity with Transformers' TorchVision fast processor is unverified. Pillow/NumPy perform preprocessing only; all learned encoder/decoder math runs in Garnet.

run_image_decoder.py loads the actual vision graph through the real libgarnet.so, encodes the photo, releases vision weights before decoder loading to limit memory, injects the 54 learned image embeddings, applies explicit image THW positions and the text position delta, then generates greedily. KV capacity 256, fixed batch 1, two CPU threads. Prompt: "Describe the main objects in this image in one sentence." The assistant prefix matches the pinned chat template's non-thinking mode. Output, followed by im_end:

> A large statue of a man with a golden headband stands on a balcony overlooking a densely populated city nestled in a mountainous landscape.

Actual Pi process exited 0. Vision compile 1,124.150 ms; vision inference 642.967 ms. Initialization including vision/text loading 18.797 seconds. Prefill 79 tokens, 39.101 seconds; next text mRoPE position 34, correctly distinct from sequence slot 79. First generated token from process-model start 57.898 seconds (excludes separate image preprocessing). Twenty-seven subsequent decode calls average approximately 0.483 seconds, approximately 2.07 generated tokens/sec in this sample. Response reached im_end at 28 total generated tokens. Full child wall 71.506 seconds; peak RSS 4,877,904 KiB (4.65 GiB), 8,191 major faults. A snapshot during generation was 65.55 C; afterward 56.2 C, throttled=0xe0000 with no active low bits. Root free space approximately 374 MB. Log: /home/shawn/garnet-pi5b-investigation/image-b1.log.

Reproduction from the investigation directory: venv/bin/python source/test2026/pi5b/prepare_cpu_image.py inputs/RealWorld-04.png models/Qwen3.5-0.8B inputs/RealWorld-04-b1.f32 1; then GARNET_OPENVINO_CPU_THREADS=2 venv/bin/python source/test2026/pi5b/measure_process.py /home/shawn/cantorai-ci/out/build/linux-arm64-release/bin/xlang3 source/test2026/pi5b/run_image_decoder.py cpu-runtime-build source models/Qwen3.5-0.8B inputs/RealWorld-04-b1.json. Batch 2 uses the same preprocessing command with suffix b2 and final argument 2.

This is actual Garnet image-plus-text execution, not just component compatibility. Remaining investigation work: complete the batch-2 image run, archive reproducible logs/commands and current verification, and audit all requirements before goal completion. Accuracy and serving limitations above remain explicit.

## Batch-2 image execution and measured result

Batch 2 also completed, exit 0, with both rows reaching im_end after exactly the same 28 tokens as batch 1. This benchmark duplicates the same real photo/prompt across two independent rows; different-length request scheduling is not benchmarked. Vision compile 1,716.689 ms, inference 1,347.558 ms. Initialization 17.033 seconds; 79 prompt tokens per row; prefill 41.227 seconds; first output 58.261 seconds from model start. Child wall 74.402 seconds; peak RSS 5,043,472 KiB (4.81 GiB), 9,260 major faults. Temperature after execution 57.85 C. Log image-b2.log.

| Actual photo run | Batch 1 | Batch 2 |
| --- | ---: | ---: |
| Mean warm decode step | 0.4811 s | 0.5739 s |
| Tokens/sec per request | 2.0786 | 1.7425 |
| Tokens/sec combined | 2.0786 | 3.4850 |
| Peak process RSS | 4.65 GiB | 4.81 GiB |
| First output from model start | 57.898 s | 58.261 s |

This sample supports keeping both fixed profiles: batch 1 has lower individual latency, while batch 2 improves combined throughput by about 68% with about 0.16 GiB additional peak RSS. It does not prove gains for batch 4, long contexts, different images, or sustained concurrent workloads. Prompt processing is currently token-at-a-time; prefill dominates response time. Releasing vision weights before decoder loading reduces peak memory but requires vision compilation again for a subsequent image in this one-shot driver. A production persistent serving API, efficient chunked prefill, quantized model execution, larger batches, video, variable image grids within one batch, and full cross-framework numerical parity remain outside what this investigation establishes.

Raw terminal logs and machine-readable metrics are archived in docs/investigations/qwen35-vl-pi5b-results. Model weights and downloaded photo are not committed. The recorded native library SHA256 is d9162e123c1c6d4d159bae34b5b994116e742fd36013a5e17318c0eacaecdfde; existing XLang runtime SHA256 34cafda4e7cdfe69a1e8127cdcc14db3da11a725f53ef8be6f0352ab16823322; CLI SHA256 f3f4f55013f5c9216fe7d3f2640828fd07032774ee104ed6e8cf8b8da06d3eaf. These existing CI artifacts were used read-only.

Primary references: [official Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B), [Transformers Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py). Runtime configurations, weights, processor metadata, and chat template are pinned to the recorded checkpoint revision; the explanatory upstream implementation link is moving main.

## Final verification and completion audit

Final native library SHA256: a95f63ee23d64e47887e5b205892b35912ed3518ca823a9a15dc370e3557ef35. Both complete image runs exited 0 with the final Core-reuse/explicit-shutdown implementation, without GLIBC_TUNABLES. They produced exactly the same photo-specific caption and all 28 token IDs recorded above. These final logs supersede the earlier performance sample; older logs remain for provenance.

| Final photo run | Batch 1 | Batch 2 |
| --- | ---: | ---: |
| Warm tokens/sec per request | 2.5079 | 1.8053 |
| Warm tokens/sec combined | 2.5079 | 3.6106 |
| Peak RSS | 4.6503 GiB | 4.8096 GiB |
| First output from model start | 42.436 s | 51.978 s |
| Prefill | 29.081 s | 38.345 s |
| Child wall | 53.839 s | 67.584 s |

Throughput uses the 27 subsequent decode calls, including the terminal token. Batch 2 improved combined throughput about 44% while reducing per-request speed about 28% in this final sample. Startup varies with SD/cache residency and active CI services; these are measured samples rather than a controlled sustained benchmark. See image-final-b1.log, image-final-b2.log and metrics-final.json.

Final component regression: 8/8 CTest passed with GLIBC_TUNABLES=glibc.rtld.dynamic_sort=1. Default-loader native Garnet decoder blocks passed linear/full attention for batches 1/2, including the live-graph shutdown guard. Native embedding/state/tied-head, synthetic vision batches 1/2 and exact preprocessing layout checks passed. Explicit cleanup and shared Core alone do not cure the separately linked host's intermittent glibc 2.41 DFS shutdown assertion. The per-process workaround is documented in test2026/pi5b/README.md; no OS libraries were changed. Primary references: [glibc dynamic-linking tunables](https://sourceware.org/glibc/manual/latest/html_node/Dynamic-Linking-Tunables.html), [OpenVINO Core lifecycle](https://docs.openvino.ai/2026/api/ie_python_api/_autosummary/openvino.Core.html).

The additional strict all-pretrained-vision endpoint comparison remains FAIL at the unchanged tolerance, as detailed above and preserved in pretrained-vision-reference.log. Full cross-framework parity is unverified. This investigation establishes actual image-plus-text execution, not production readiness: quantization, efficient prefill, persistent serving, general Unicode tokenizer parity, video, larger batches and long contexts remain unverified.

Audit: isolated D:/CantorAI/Garnet-pi5b clone on feature/qwen35-vl-pi5b from main; original D:/CantorAI/Garnet branch remains feature/gpt-oss-120b and was not modified. Existing xlang3 SDK/runtime were read-only. Saved SSH alias used with verified address overrides, no credentials logged. Pi OS/architecture/RAM/swap/storage/CPU/thermals/runtime checks, approved inactive-output cleanup, pinned checkpoint hashes, reproduction commands and actual latency/memory/throughput limits are recorded. Qwen3.5-0.8B uses Garnet xModel/OpenVINO encoder/decoder plus Garnet native CPU tensor/tokenizer/tied-head operations. No inference substitution was used. All requested investigation work is complete with the explicit limitations above.

## Follow-up optimization: four cores, FP16 and native projection

The Pi has four Cortex-A76 cores. Actual full-photo profiling establishes that
more worker threads are not automatically faster for these small decoder graphs:
FP32 four threads measured 1.59 tokens/sec, 52.729 s prefill, versus the earlier
two-thread FP32 2.51 tokens/sec and 29.081 s prefill. The FP16 hint with four
threads measured 3.48 tokens/sec and 19.132 s prefill; two-thread FP16 initially
measured 3.93 tokens/sec and 16.191 s prefill. Four-thread FP16 changed one word
from populated to packed, retaining the same scene description. Temperatures
sampled during the experiments were 60.6-71.05 C; throttled=0xe0000 has no active
low status bits. Existing CI services were left running.

The explicit GARNET_OPENVINO_CPU_INFERENCE_PRECISION=f16 performance profile
reduces the native package's working memory substantially. It is a backend
precision hint, not a claim that every operation/state or the native BF16 output
head executes in FP16. FP32 remains the default numerical reference; no existing
test tolerance was relaxed. Before the native projection change, batch-2 FP16
measured 5.05 tokens/sec combined, 23.872 s prefill, and 2.94 GiB peak RSS; both
rows matched all 28 baseline token IDs. General FP16 accuracy parity remains
unverified beyond these executed examples.

The native CPU tied-head kernel now uses four independent NEON FMA accumulators
per 16-element block and combines them before the existing tail path. This
reduces the serial accumulator dependency without weight quantization. The
independent full-vocabulary BF16 reference passed its unchanged argmax checks
over all 248,320 outputs and 256 sampled logits per row, batch 2. The head call
measured 53.832 ms in that focused run; this is not a controlled comparison with
older cold measurements. Native tensor/head small numerical fixtures also
passed. Final native library SHA256:
276704d449c8f39eeba4ad9d23f05f1fc69dc523862185ed7498deb0a7e2eaee.

Final two-thread FP16 batch-1 photo run exited 0, preserving all baseline token
IDs. Peak RSS 2.8024 GiB; prefill 16.059 s; initialization 9.975 s; first output
26.034 s; wall 33.202 s. This run had zero major page faults, so its warm-file
startup must not be presented as a guaranteed cold-start speedup. Warm decode
performance and the final batch-2 result are recorded in metrics-optimization.json.

An optional fifth prepare_cpu_image.py argument now selects the pixel budget;
its default remains 65,536. Tested 16,384 with the same checkpoint/photo and FP32:
128x96, 12 visual tokens instead of 54; prompt 37 instead of 79 slots. Prefill
18.786 s, first output 27.848 s, decode 2.27 tokens/sec. However the output called
the headband a glowing orb and the sculpture metallic. This quality tradeoff is
not recommended as the default. The original-resolution FP16 profile preserves
more image evidence and is the recommended measured improvement.

Reproduce the recommended profile with the README command, setting
GARNET_OPENVINO_CPU_THREADS=2 and GARNET_OPENVINO_CPU_INFERENCE_PRECISION=f16.
Actual learned inference remains Garnet/xModel throughout. Further experiments are recorded below. Raw optimization logs retain
slower and quality-reducing experiments as well as successful improvements.

Final optimized batch 2 also exited 0 with the same 28 token IDs in both rows.
Both final photo runs used the rebuilt library above, original image size,
two threads, FP16 hint and the unrolled native output projection.

| Final measured profile | FP32 baseline B1 | Optimized B1 | FP32 baseline B2 | Optimized B2 |
| --- | ---: | ---: | ---: | ---: |
| Combined decode tokens/sec | 2.5079 | 4.0951 | 3.6106 | 5.7641 |
| Per-request decode tokens/sec | 2.5079 | 4.0951 | 1.8053 | 2.8820 |
| Peak RSS GiB | 4.6503 | 2.8024 | 4.8096 | 2.9481 |
| Prefill seconds | 29.081 | 16.059 | 38.345 | 23.576 |
| First output seconds | 42.436 | 26.034 | 51.978 | 35.721 |

Decode improved 63%/60% for batches 1/2 respectively; memory fell 40%/39%.
The final optimized runs had zero major page faults, unlike the earlier
FP32 samples, so first-output and startup comparisons include cache effects.
These remain one-photo, short-decode measurements with existing services active,
not a general quality/sustained-throughput guarantee. The strict pretrained
vision parity limitation from the earlier investigation remains unresolved.

## Post-reboot experiments: first output and decode speed

The initial resident chunk implementation exhausted RAM and the 2 GiB swap;
SSH stopped responding and the user manually rebooted the Pi. It produced no
completed inference result. `chunk-photo-b1.log` retains the failed attempt.
The replacement compiles one prefill layer at a time and releases it before
compiling the next. Chunk-4 padded fixtures passed unchanged numerical tolerances
for linear/full attention, batches 1/2, and pause/resume. The actual photo matched
all 28 baseline token IDs, with 2.81 GiB peak RSS. However, prefill took 37.02 s
(including 20.10 s compilation and 8.98 s graph inference), so this is an
experimental path, not the recommended one-shot latency profile. Its 75.99 s
first output includes cold initialization after reboot.

`measure_process.py` now supports stop-on-threshold guards for child RSS,
host available memory and elapsed time. These poll every 100 ms and kill only
the launched child's process group; they are not hard OS memory reservations.
Actual Linux allocation/time-limit probes verified termination. Subsequent
photo runs used 4096 MiB child RSS, 1024 MiB minimum available and 180 seconds.
No guard triggered in the completed runs.

The OpenVINO INT4-weight experiment completed execution but reached the
32-token limit without EOS, changed the caption, and retained about 2.81 GiB
RSS. It did not provide a useful overall improvement. It remains opt-in through
GARNET_CPU_WEIGHT_PRECISION=int4_fp16; default decoder weights are unchanged.
The decoder builder also fixes rank-one metadata indexing with group size -1.

The separate native Garnet Q4-weight/Q8-activation output-head experiment uses
GARNET_CPU_HEAD_PRECISION=q4q8 and GARNET_CPU_HEAD_THREADS=4, with unquantized
decoder weights and the FP16 backend hint. It completed the photo caption with
EOS at about 4.87 tokens/sec and 2.47 GiB RSS. First output was 28.01 s versus
28.29 s in the immediately preceding two-thread BF16-head control; startup/cache
variation prevents a first-output speedup claim. Its shorter caption identified
the statue, golden headband and cityscape, but omitted balcony/mountain detail.
This is a quality-changing experimental option, not general accuracy parity.
The default BF16 output head remains unchanged. One-thread FP16 also completed
all baseline token IDs at about 4.2 tokens/sec and 25.93 s first output; this
single sample does not establish a consistent advantage over two threads.

Logs preserve failed and slower experiments. Strict pretrained vision parity
remains unresolved; component/chunk success does not remove that limitation.


Four BF16-head workers passed the independent full-vocabulary batch-2 reference
at unchanged tolerances and preserved all photo caption IDs. Actual generation
slowed to about 3.49 tokens/sec, so two workers remain recommended. Rebuilt native
library SHA256: 24200af5a30258f72a93ab526960ba9404e360735efe4443ac004b84e215f3b3.
