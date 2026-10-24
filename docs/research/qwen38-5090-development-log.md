# Qwen3.8 RTX5090 research log

## Record policy

Append dated experiment entries; do not rewrite failed experiments into successes.
Use one fixed remote root `D:\CantorAI\5090-study-20261023`, one shared model cache,
and a flat `results` directory. Reuse build directories; keep accepted control
binaries and hashes separately from candidate output. Remove transfer archives
only after verified extraction. Keep compact raw JSONL and paper evidence; prune
rebuildable engines only when they are not needed by an active comparison.

For every future optimization record: experiment ID/time in UTC and local zone;
hypothesis; exact source diff and SHA-256; build flags/toolchain; model revision,
quantization, calibration and kernel dispatch; workload IDs and input hashes;
prompt/image token counts, output tokens, seeds, thinking/cache settings;
batch and client concurrency; control/candidate order; warmups and repetitions;
correctness and quality results; all raw latency/token/memory observations;
median/p95 and uncertainty; profiler configuration/overhead; failures;
interpretation, decision and limitations. Record power/temperature/clocks and
background load. Retain rejected candidates and interrupted runs as such.

Measurements, shape/memory estimates, hypotheses, and vendor claims are separate
evidence classes. Synthetic unit fixtures are not benchmark observations. GPU
profiling attribution is not production serving latency. No cherry-picked cohort
or changed precision/workload may silently support a 2x speedup claim.

## P000 — 2026-10-23: source and compatibility audit

Method: inspect the current Garnet working tree, retrieve pinned primary model
configs, compare architecture/quantization contracts and query target hardware
over SSH. The referenced chat exposed only message placeholders for prior
assistant claims, so no prior predicted token/s or TTFT is treated as evidence.

Source HEAD: `5b0dc7ff8e5c1a77807138db310605bcd12aa390`.
The original branch is `feature/gpt-oss-120b`, with substantial ongoing Qwen3.5
edits. A separate local worktree at `D:\CantorAI\work\qwen38-5090` uses branch
`qwen38-5090`. Current source was preserved through a per-file hashed snapshot;
its pre-existing dirty baseline is not attributed to this task or silently
committed. Future runs must retain both HEAD and complete source hashes.

Base source archive SHA-256:
`7c29e60a515827464a7fb0fa461d45bc0d5a91f93f6c00c7b7e476a1f8be9d99`.
Remote extraction verified that digest. `source-manifest.json` lists captured
Garnet/XLang3 files and exclusions; `preparation-manifest.json` supersedes
preparation files changed after the base snapshot. No credentials are included.

Observations: Qwen3.8-27B exists and uses Qwen3.5-family hybrid VLM architecture.
The 27B decoder, untied head and larger vision tower differ from the admitted
2B/4B models. NVIDIA's checkpoint mixes NVFP4 and FP8. A compressed-tensors
group128 symmetric W4A16 candidate is pinned, but not runtime-admitted. See the
implementation task and checkpoint evidence for exact revision/config hashes.

Target observation: Windows `zsdx-5090`, one RTX5090, 32607 MiB, driver617.14,
CUDA toolkit12.8. Existing `D:\CantorAI\Garnet` is deployment content, not a
buildable checkout. Preserved it and staged source under the fixed study root.
No target model weights were downloaded during preparation.

Outcome: 27B CUDA serving admission and packed NVFP4/FP8 integration are pending.
Current BF16 external projections and captured INT4 dequantization cannot be
treated as a working ModelOpt loader. No performance result or speedup exists.

## P001 — 2026-10-23: CPU preparation validation

Method: run `python -m unittest discover -s test2026/cuda/qwen38_5090 -p test_*.py -v`
locally with Python3.12.10 and remotely with `C:\Python312\python.exe` on
CD-RTX5090. Run offline preflight on the pinned NVIDIA config on both hosts.

Outcome: all9 tests passed on each host. Evidence-integrity checks validate
checkpoint-config hashes; negative tests reject incompatible model shapes and
bad timing records. Synthetic metric fixtures test correct decode token counts,
cohort throughput denominator, failed-request rejection and comparison against
the faster of both engines. These are CPU tests, not GPU inference validation.

Calculated B1/context4096 canonical-state sizes: attention KV536870912 bytes,
FP32 recurrent matrices150994944 bytes, convolution history7864320 bytes.
These omit weights, graphs, duplicated state, vision, activations and workspace;
they do not establish peak VRAM or prove fit on32GB.

## P002 — 2026-10-23: environment preparation

Method: start native CMake/Ninja/Visual Studio2022 Build Tools installation with
logs in the fixed remote root. User authorized WSL installation for reference
baselines. Installed WSL3.0.3 and enabled VirtualMachinePlatform without reboot.
Windows reports a restart is required before changes take effect. Native build
tools were still downloading at this entry; installation success is unverified.
Garnet remains a native Windows build. WSL is reserved for reference engines.

Outcome: baseline engines, native Release build, target GPU correctness,
quantized model fit and timed measurements remain pending. Do not publish
preparation timings as TTFT or claim the >=2x target is achieved.

## P003 — 2026-10-23: setup failure and alternate method

The initial CMake winget download failed with InternetOpenUrl error0x80072efd
(recorded installer exit -2147012867). Preserve its original log and status.
Stopped only this task's first setup process and its direct winget child to
avoid competing installers. Retry uses official PyPI CMake/Ninja wheels in
`tools/python` within the same study root, followed by native Visual Studio
Build Tools installation with `--norestart`. Retry success remains pending.

Remote `preparation-tests.log` records all9 CPU tests passing with exit0.
The source archive and subsequent preparation patch were verified and extracted
into the fixed study root. Preparation files are committed on the dedicated
branch; inherited Qwen3.5 source edits remain separate and uncommitted. No GPU
benchmark has run. A prior whole-tree whitespace check also reports inherited
Qwen3.5 issues; those are not attributed to this task. The preparation manifest's
own CRLF-only whitespace issue was fixed in a separate commit.

## P004 — 2026-10-23: reproducible inherited baseline and native build

Committed the previously inherited CUDA working snapshot separately as
`2355fc4b5`. This supersedes P000/P003's earlier uncommitted-baseline status;
the original checkout is unchanged. The commit is baseline preservation, not
a new optimization or performance result.

Local full-source configuration initially failed because Cargo was absent from
the build shell PATH. Adding the existing local Cargo installation allowed
configuration. Compilation then failed at XLang3 `sre_module.cpp:5109`:
`std::regex_constants::multiline` is unavailable in the installed MSVC STL.
These initial failures are retained as observed diagnostic excerpts here;
complete per-attempt raw logs were not preserved before the log path was reused.
Future build logs must append or retain each attempt before reuse.

Method change: an optional native-Windows CMake wrapper links Garnet against a
preserved existing Release XLang3 DLL/import library and matching public SDK.
It does not change or rebuild the VM. XLang3 executable SHA-256 is
`2e2db12a2dbc1134a3a4c662c080b91d500b37af488661630d4d580e7684dc10`;
runtime DLL SHA-256 is
`8dabab841d460120b4e0b5c1539851e9ff17dbb0f5113beece4017f9f3fac4b3`.
Record this binary dependency, not an unsupported claim that the captured VM
source was rebuilt successfully. The wrapper copies the executable alongside
Garnet for native package discovery. Its initial imported-executable path
caused three module-discovery failures; corrected the path and reran those checks.

## P005 — 2026-10-23: exact model and tensor-header contracts

Added the dedicated Qwen3.8-27B contract and logical projection shapes under
`xModel/qwen38/vl_27b`. The offline preflight uses that same contract. It checks
the gated attention shape, untied head, 64-layer hybrid schedule, larger vision
tower, multimodal rotary layout and token IDs. Runtime admission remains off.

Method: bounded official HTTP range requests fetched the three pinned NVIDIA
safetensors headers, without weight payloads. Preserve exact raw header bytes,
their SHA-256 hashes, immutable source URLs, index and audit result in
`test2026/cuda/qwen38_5090/evidence/nvfp4-headers`. All497 logical text projections
match their physical tensor layouts:193 NVFP4,208 FP8 and96 BF16. NVFP4 matrices
are U8 packed with half-width columns, FP8 block16 scales and FP32 scalar scale2;
FP8 projections have scalar FP32 scales. BF16 in_proj_a/b remain unquantized.
Indexed tensor payload is21921428072 bytes. This is checkpoint storage, not VRAM.
Scale payload values, packing nibble arithmetic and GPU kernels remain unverified.

## P006 — 2026-10-23: separate BF16 head correctness groundwork

Problem: the prior CUDA I/O head always projected with the embedding table.
The 27B checkpoint has an independent lm_head. Implemented a separate BF16 head
when present; both full logits and indexed proposal heads use it. Missing untied
heads and unsupported packed heads fail before embedding allocation, rather
than silently changing the model. Accounting distinguishes shared embeddings
and separate-head allocation. This does not yet implement the packed NVFP4 head.

Method: native Windows Release build, CUDA13.2.78, TensorRT10.16.1.11, SM89,
preserved XLang3 runtime above. Ran tiny independent exact-integer BF16 fixtures
on local RTX4080 device1 (driver617.14), not the RTX5090. Five checks pass:
distinct head logits/unchanged embeddings, tied behavior, reopening, packed-head
rejection and missing-head rejection. Synthetic fixture outcomes, binary/source
hashes and exact patch are in `evidence/native-head`. Tested Garnet DLL hash:
`816648a65f5d7b162e29db95e3e7990fd689f3357fb143548d0af6ccb50d1e14`.

Native API and serving-default CTests pass. Broader production capture does not
pass: its initial stale27-module inventory was updated for the two already
present helper modules; it now reaches29 imports and nine root captures before
failing in ASR prefill with `TypeError: object is not callable` at T.unary_op.
Retain the full failing gate log. Its baseline/runtime cause remains unresolved;
do not describe the entire regression suite as passing or infer pretrained
Qwen3.8 correctness from the small head fixtures.

Sixteen CPU tests pass, including actual pinned header layout/hash validation,
negative architecture and quantization cases, timing arithmetic and a synthetic
download integrity/budget test. None is a benchmark measurement.

## P007 — 2026-10-23: target setup and verified download route

CMake4.4.4, Ninja1.13.2 and Visual Studio2022 Build Tools17.14.42 installed on
CD-RTX5090. User explicitly approved restart; restarted and reconnected.
WSL3.0.3 now responds. Ubuntu24.04 installation is running under a retained SSH
job handle; the distro/GPU path is not yet verified. A pinned CUDA13.2/TensorRT
10.16.1.11 Python-package toolchain install is also running in the fixed tools
folder. It uses NVIDIA's official package index, including its nvidia.cn CDN.

Direct Hugging Face access from the target timed out; a mirror returned403 and
was not pursued. An official expiring public CDN URL resolved on the local
machine succeeds from the target with a206 response and exact8-byte header.
Prepared small metadata plus four official CDN artifact tasks for the pinned
NVFP4 checkpoint. Download helper verifies primary LFS SHA-256 values and byte
counts, uses partial files for resume, publishes only verified artifacts, and
keeps signatures out of permanent provenance. One shared model directory is
used; no BF16 base-model or duplicate framework model copies are requested.

No text/image TTFT, decode speed, concurrency throughput, peak VRAM, pretrained
quality comparison or >=2x performance result has been measured. These goals
remain active; the native head change is correctness groundwork only.

## P008 — 2026-10-23: download and SDK status correction

The official-CDN model download is now live, with one1.97GB shard and tokenizer
verified and both larger shards making progress. The permanent result is not
complete until every declared artifact passes its primary SHA-256 check.
Small metadata has been staged in the same shared model directory.

The first native SDK install exited0, but pip selected CUDA13.4.92 CRT/NVRTC/NVVM
components alongside the requested13.2 compiler/runtime. Installation success
does not prove a consistent toolchain. Added a complete requirements lock and
started replacement with `--no-deps`, preserving the first attempt log. Only
the SDK folder generated by this task is replaced; models and deployment data
are untouched. Validate final versions and compilation before any timed run.

## P009 — 2026-10-23: append-only evidence ledger

Added `run_logged.py` for build/setup/correctness commands. One fixed directory
contains an append-only JSONL ledger and raw output log with per-command byte
offsets, command arguments, UTC times, source identity/diff hashes, selected safe
environment fields, artifact hashes and actual exit codes. Process duration is
explicitly not model latency. Copied remote source uses external manifest hashes
and the study commit record. Never use this wrapper for authentication commands.

Tests verify that successful and failed commands both remain in the ledger and
that process duration cannot be relabeled as a benchmark. The initial ledger
test exposed an unclosed stdout stream; fixed it with process context ownership
and retained both attempts. All18 CPU tests pass in the latest recorded run.
The two commands printed by the ledger unit test are synthetic fixtures, not
new model measurements. Original unit-test/logging output stays under the
fixed local `out/study-evidence` directory.

The fully pinned SDK replacement has now exited0 on CD-RTX5090. Package install
success still requires a subsequent native compile/runtime test. Two model
shards and tokenizer are verified; the remaining shard continues downloading.
Ubuntu installation still has a confirmed live job handle and no terminal
result; do not start another distro installer solely because its log is empty.

## P010 — 2026-10-23: model ready, target validation and packaging fixes

The model downloader finished with exit0: all four CDN artifacts passed their
declared byte counts and primary LFS SHA-256 checks. The three weight shards
and tokenizer total21934506600 bytes. Small metadata remains a separate pinned
archive/provenance set. No inference or memory-fit claim follows from download.

Updated copied source verification reports6337 files with zero mismatches and
commit1dd69410e. All18 CPU tests also pass on CD-RTX5090, with raw output and
manifest identity retained in its command ledger. The additional18 copied
SDK/runtime files passed hash verification. CUDA compiler reports13.2.78.

Ubuntu installation finished with exit0. An actual root-shell GPU check reports
Ubuntu24.04.5 LTS and RTX5090/32607MiB/driver617.14 inside WSL. This establishes
GPU visibility, not framework baseline readiness or measured serving performance.

First target native configure passed MSVC/CUDA compiler checks but failed in
legacy FindCUDA because a backslash path was interpreted as an escape. Normalized
the CUDA path and retried. The next configure found CUDA13.2 but exposed a
runtime-packaging mismatch: official Windows wheels place cuBLAS DLLs under
bin/x86_64, while Garnet searched bin and bin/x64. Added the x86_64 runtime
discovery path, retaining the existing paths. Development import libraries
missing from the wheels were supplied from the matching local CUDA13.2 SDK and
checked by SHA-256. These are build/deployment changes, not an inference speedup.
Both failed configure attempts are retained in the append-only target ledger.

## P011 — 2026-10-23: image runtime and portable dependency closure

Completed the SDK's image dependency set with NPP13.1.0.48 and nvJPEG13.1.0.48,
matching the component versions in the local CUDA13.2 version.json. The wheel
merger checked member hashes, refused conflicting existing files, installed44
files and removed temporary wheel downloads. Its actual setup job exited0.
Added both packages to the SDK lock. The build/runtime discovery now also
includes cuBLASLt, NPP core and CUDA runtime DLLs, avoiding reliance on a global
older CUDA installation when loading the packaged Garnet runtime.

No model inference benchmark has run. The next target step is a fresh native
SM120 build, followed by the small head fixtures and quantized projection work.

## P012 — 2026-10-23: target image development imports

The SM120 configure attempt at commit36133e8ed verified6338 source files but
failed generation because CUDA::nvjpeg was absent. The official runtime wheel
contained nvjpeg.h and nvjpeg64_13.dll but no nvjpeg.lib. Added matching local
CUDA13.2 development import libraries nvjpeg.lib, nppc.lib and nppig.lib to the
existing three-library overlay. All six transferred files passed SHA256 checks;
the transfer archive was removed. A fresh native build is recorded under label
native-sm120-image-import-libraries. This records setup, not inference evidence.

## P013 — 2026-10-23: native SM120 build and RTX5090 head correctness

The native-sm120-image-import-libraries job completed exit0 after86 build steps,
including linking bin/garnet.dll. The rtx5090-native-head-fixtures correctness
job then completed exit0 and printed UNTIED_HEAD_NATIVE_CHECKS_PASS on the target
GPU with CUDA_VISIBLE_DEVICES=0. This validates distinct untied logits, preserved
embeddings, tied behavior, reopen behavior, and packed/missing head rejection
before embedding allocation. Results are retained in results/rtx5090-head-checks.json
and the append-only command ledger with the DLL hash.

These are tiny synthetic correctness checks, not full Qwen3.8 inference or
performance measurements. Packed NVFP4 output-head support is still pending;
there is no measured2x speedup, TTFT, throughput, or peak-VRAM result yet.

## P014 — 2026-10-23: packed NVFP4 projection correctness on SM120

Added a direct packed NVFP4 CUDA projection primitive. It consumes ModelOpt
row-major E2M1 pairs (even element low nibble), E4M3 block16 scales and positive
F32 scale2, with BF16 activations/output and F32 accumulation. It allocates no
full-size dequantized weight matrix. This is an experimental W4A16 arithmetic
path, not Blackwell W4A4 tensor-core execution or a completed model integration.
ModelOpt's published nvfp4_tensor.py pack/unpack implementation confirms nibble
order: https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/quantization/qtensor/nvfp4_tensor.py

Initial test CMake generation failed due imported CUDA targets being scoped to
the Garnet child directory; explicit parent CUDAToolkit discovery fixed it.
Target job nvfp4-projection-oracle then completed exit0. Its independent CPU
oracle uses arithmetic E4M3 decoding and double accumulation, compared after BF16
rounding with51 GPU outputs, all16 nibble codes, subnormal/zero/large scales,
multiple rows/batches and a non-power-of-two K=528. Invalid K alignment is rejected.
The exact test/kernel source hashes and raw output are retained by run_logged.
No performance number was measured. This primitive is not wired to the serving
factory yet; the production packed output-head admission remains intentionally
closed until integration and end-to-end tests pass.

Pinned candidate reference packages using official PyPI metadata retrieved this
session: vLLM0.32.0, SGLang0.5.22, uv0.13.5. Saved release filenames, hashes,
requirements and timestamps in evidence/*-pypi.json. Current primary recipes:
https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.8-27B.yaml
https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx
These upstream claims are not target measurements. WSL Python prerequisite
installation is active; reference package installation and model load remain
unverified. Separate engine environments share a uv cache within the single
study-owned WSL disk; neither script downloads a second model copy.

## P015 — 2026-10-23: integrated packed output head and actual checkpoint probe

Connected the direct NVFP4 W4A16 primitive to CUDATensorIO full-head logits and
prediction. Admission requires ModelOpt config, exact packed geometry, block16
E4M3 scales, finite positive scalar scales and nonnegative finite block scales.
Unsupported indexed packed-head execution explicitly rejects instead of falling
back to embeddings. Close/reopen clears packed state. Memory reporting labels
head_precision=modelopt_nvfp4_w4a16 and counts compressed weights plus scales.
This is an explicit W4A16 execution arm of the mixed checkpoint; it must not be
reported as equivalent to the checkpoint's W4A4 activation recipe.

Caught a source-sync timestamp defect before claiming the integration passed:
ZIP timestamps carry no time zone, and sender/target zones differ by15 hours.
Two logged builds therefore reported success without rebuilding the changed C++
object. Preserved those attempts. Forced affected source timestamps on the target;
nvfp4-head-target-clock-rebuild actually compiled cuda_tensor_io.cpp and linked
successfully. Source transfer now records epoch mtime_ns, restored after SHA256
verification, avoiding reliance on ZIP local time. Future evidence should retain
both source and binary hashes and verify behavior, not infer it from build exit0.

rtx5090-nvfp4-head-integration completed exit0: original five tied/untied/rejection
cases plus valid packed head exact logits and invalid scalar rejection. The18
CPU preparation/metrics/ledger tests also passed after these source changes.

actual-checkpoint-nvfp4-head completed exit0 against the pinned downloaded real
NVIDIA checkpoint. It opened the248320x5120 head without BF16 weight expansion,
computed full logits for a sparse four-coordinate hidden vector, and checked11
rows across the vocabulary against independently decoded checkpoint byte slices.
All11 passed the1% relative or1e-6 absolute criterion (BF16 output rounding);
raw expected/actual values and exact source bytes are retained in
results/real-head-oracle.json and results/real-head-checks.json.
Oracle SHA256: efdfca5e97984e7e36b778b25dc957924ac6fb56a4798b4f41887e756c6af36d.
Tested Garnet DLL SHA256: 212cb70e9e810d6ee325dbe00e715e8b746600e693e854548cc5538fc21ef2eaa.
Owned embedding allocation2542796800 bytes; packed head plus scales715161600
bytes. These are component allocations, not model peak VRAM. No full-model
quality, TTFT, decode speed, throughput or2x result is established by this probe.

WSL Python prerequisite setup completed exit0. vLLM0.32.0 installation is now
active in the study-owned WSL disk with shared uv cache; its terminal result and
GPU/model load checks are pending. SGLang installation has not yet run.
Remaining critical path: mixed FP8/NVFP4 text projection integration, Blackwell
W4A4 tensor-core path,27B model/vision admission and parity, then measured
reference engines and Garnet under the matched benchmark contract.

## P016 — 2026-10-23: reference VLM smoke and native packed FP4 execution

Both pinned reference environments installed successfully and passed actual CUDA
smoke tests on the target RTX5090 SM120. vLLM0.32.0 resolved Torch2.13.0+cu130;
SGLang0.5.22 resolved Torch2.14.1+cu130. Exact package freezes are retained under
results/references. They are independent engine environments, not identical Torch
builds. The uv cache is shared inside the study-owned WSL disk.

The vLLM initial VLM server loaded the shared pinned checkpoint, TP1, context4096,
max sequences8, BF16 KV and F32 SSM, eager mode, prefix caching disabled. Its
observed selected backends were CutlassNvFp4LinearKernel, CutlassFP8ScaledMM,
FlashAttention2, and FlashInfer GDN prefill. The server's model-loading report
was19.83GiB, not a measured process peak. Text smoke answered4; image smoke
answered red for a deterministic224x224 PNG with a small blue corner. Both
requests completed HTTP200 with26/93 prompt tokens and2 completion tokens each.
Raw requests/responses and image are in results/references/vllm/smoke. PNG SHA256:
ad17ed2fc2311d796ef00f78cad2545ac9b9a3626c4a60e1c18919319336665e5.
These are correctness smoke requests, not controlled timing or quality benchmarks.

The initial TensorRT10.16 Q/DQ projection implementation exposed two build-time
constraints: DynamicQuantizeV2 with a double scale rejected a vector block_shape,
and FP4 block DQ with F32 scales could not directly output BF16. Scalar block-size
DynamicQuantize and consistent scale/output types allowed execution; F32 DQ plus
BF16 casts also executed. NVFP4 W4A4 and FP8 W8A8 arithmetic each matched2048
exact synthetic outputs. However, actual engine inspection selected BF16 GEMM
(sm80_xmma_gemm_bf16bf16...) and materialized decoded weights. Consequently this
helper is retained only under the study tests as a numerical reference, not wired
into deployment or advertised as native FP4 GEMM. Preserved inspector variants
include modelopt-trt-bf16-fallback-nvfp4.json and modelopt-trt-f32-dq-nvfp4.json.
All failed attempts and actual source hashes remain in the append-only ledger.

Implemented a separate explicit cuBLASLt CUDA_R_4F_E2M1 path using block16 E4M3
scales in128x4 tiled layout, BF16 output and F32 compute. Tensor-global scales
are combined in alpha. The cold correctness helper currently creates descriptors
and selects a heuristic per call; it is not a tuned serving implementation.
Target job cublaslt-native-fp4 completed exit0 for batch1 and batch16, comparing
256 and4096 outputs exactly against an independent scale-layout oracle, including
multiple scale tiles and varying scales. This verifies packed FP4 cuBLASLt API
execution on SM120 without a caller-allocated full BF16 matrix. Kernel-level
hardware instruction tracing and performance validation remain pending.
Primary layout/API sources consulted:
https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
https://nvidia.github.io/cudnn-frontend/mxfp8-scale-factor-128x4-layout/
https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt/LtNvfp4Matmul

Stopped only this study's verified vLLM server PID2795 with SIGTERM after smoke;
its tracked job exited0 and device memory returned to106MiB. SGLang's initial
VLM load is now active on a separate loopback port, with matching4K/eager/cache
settings and explicit eight-state Mamba cache. Terminal/model-smoke results are
pending. Garnet currently retains F32 attention/conv state, so these BF16-cache
reference configurations are not yet a precision-matched2x comparison arm.
No controlled TTFT, decode tok/s, concurrency throughput, model peak VRAM or
measured2x result has been established. Next: activation quantization/scale
repacking and retained native projection contexts/plugins, then mixed checkpoint
loader and full27B decoder/vision correctness before the performance gate.

## P017 — 2026-10-23: GPU quantization pipeline and reference SDK repair

Added CUDA BF16 activation-to-NVFP4 quantization using per-block maxima,
E4M3 round/saturate block scales, the checkpoint tensor-global scale, E2M1
round-to-nearest conversion, and packed low-even nibbles. Added GPU repacking
of row-major checkpoint scales to128x4 tiled storage. Tiled allocations include
zero-filled outer/inner padding. Native FP4 compilation is guarded at CUDA12.8;
older SDK branches return not-supported (an older-SDK build is not yet tested).

Jobs dynamic-nvfp4-native-pipeline and guarded-native-fp4-pipeline both completed
exit0. The final native-fp4-pipeline-zero-blocks job rebuilt/linked Garnet and the
native test after the explicit CUDA runtime-header include, then completed exit0.
For each batch1 and16 it validated prepacked native cuBLASLt arithmetic and the
complete BF16 input -> GPU quantization/repacking -> native packed FP4 multiply
pipeline. Each case checked256/4096 outputs exactly against a CPU oracle. Added
all-zero activation cases and checked zero-filled padded scale storage. These
are synthetic correctness checks; kernel tracing, actual checkpoint layer parity,
retained descriptors/heuristics, TensorRT plugin integration and performance are
still pending. The tested code performs no full BF16 weight expansion.

The linked Garnet build also completed exit0 after moving the numerical-only
TensorRT Q/DQ helper out of production source. Obsolete study-created backend
copies were removed only after matching their recorded SHA256 hashes. Source
verification then checked6355 files with no mismatches at commit80847253d.
Subsequent source additions are separately hashed in the command ledger.

SGLang's first initial-load job terminated exit9: its JIT compiled a per-tensor
FP8 kernel but the Linux linker could not find -lcudart under CUDA_HOME/lib64.
Inspection found the installed wheel runtime under cu13/lib/libcudart.so.13,
with neither lib64 nor the unversioned linker entry. The guarded setup helper
created relative symlinks lib64->lib and lib/libcudart.so->libcudart.so.13, refusing
to replace differing existing paths. No library copy/package replacement occurred.
Setup job sglang-wheel-cuda-linker-layout completed exit0 and saved provenance
in results/references/sglang/cuda-layout.json, including actual nvcc13.4.92 output
and runtime SHA25696c42e418cec19054186b9429c321603cc1190bf26a18104e19408117a2a817b0.
The sglang-vlm-cuda-linker-fixed retry is active; it has reached shard loading.
Both attempts append to sglang-server.log. Its text/image smoke is not yet proven.

The integration/performance goal remains incomplete. There are no controlled
benchmarks or measured2x speedup results yet.

## P018 — 2026-10-23: consistent Linux reference compiler lock

The SGLang linker-fixed retry terminated exit9 after loading all three shards.
Its next FlashInfer JIT failed a CUDA compiler/header compatibility check:
automatic dependency resolution had installed nvcc/CRT/NVVM13.4.92 alongside
framework CUDA13.0 runtime headers. This was not a model architecture failure.
Verified official package availability and added an explicit Linux compiler lock:
nvcc/CRT/NVVM13.0.88 plus CCCL13.0.85. The alignment script preserves original
freezes and writes a separate freeze-cuda13.0.txt and nvcc-cuda13.0.txt after
successful completion. The installed framework runtime pin is retained.

Job sglang-align-cuda13-compiler is active, downloading the pinned compiler
components. Its final result and a fresh SGLang model retry are still pending.
The installer now applies the compiler lock and verified linker-layout helper
before freezing the environment. Future vLLM starts also expose their environment
bin and CUDA_HOME for FlashInfer discovery; vLLM's existing environment must be
aligned with the same compiler lock before that modified script is rerun.
Original smoke results remain valid for their earlier recorded configuration.
No controlled benchmark has been collected, and the2x target remains unproven.

## P019 - 2026-10-23: native packed projection integration

Strongly typed TensorRT NVFP4 plugin checks passed B1/B16 CPU-oracle comparisons,
serialization, explicit CUDA graph replay and invalid metadata rejection. Initial
weakly typed format negotiation failed; production uses strongly typed networks.
TensorRT10.16 asserted when stripping opaque INT8 packed constants. Mixed plans
currently embed compressed constants; sharing weights across engines remains pending.
The integrated mixed MLP passed exact outputs and graph counters (1 capture,
2 cached replays, 0 failures). Earlier three-call probes captured without replay;
the probe now uses five calls and asserts counters. No speed claim follows.

BF16 regression testing caught that isRefittable() cannot determine whether the
existing stripped BF16 engine needs refitting. Skipping refit returned residual-only
outputs. Unconditional refit then rejected the embedded mixed engine. An explicit
embeddedWeights flag now preserves existing callers' refit behavior; its fresh
remote two-precision regression is pending. The scoped Qwen TensorCompat fix assigns
the native factory before invocation. No VM change. ASR capture remains unresolved.

## P020 - 2026-10-23: reference startup recovery

Both reference compiler alignments completed exit0 with CUDA13.0.88/CCCL13.0.85.
SGLang then failed -lcuda discovery; LIBRARY_PATH now includes verified WSL driver
libraries. The retry loaded shards but exhausted guest RAM/swap during warmup/JIT:
MemAvailable=0, SwapFree=64kB, MemTotal=49139552kB, SwapTotal=12582912kB.
This is a startup failure, not a benchmark. Terminating only study Ubuntu completed
exit0; a fresh health probe is pending. No global .wslconfig or unrelated process
changes. The next script bounds compilation jobs to2 and disables FlashInfer startup
autotuning; record this separately from subsequent optimized baseline settings.

SGLang text/image smoke, fresh aligned vLLM smoke, full Garnet VLM, FP8 attention,
INT4/4B comparisons and controlled measurements remain pending. The2x target is
unproven. Source hashes, failed attempts and raw command logs are retained.


P019/P020 follow-up: explicit-embedded-weight-mlp-regression completed exit0 on
CD-RTX5090. Both profiles produced exact256 outputs (NVFP4=3.4296875,
BF16=3.3359375), each with graph_captures=1, graph_replays=2 and failures=0.
Both engines reported engine_refittable=false, confirming that the diagnostic is
not a sufficient refit policy. Raw ledger is results/native-projections/commands.*;
JSON results are native-mlp-checks.json and native-mlp-bf16-checks.json. Local18
CPU study tests passed. Recovered WSL memory probe completed exit0 with48035912kB
free RAM and all12582912kB swap free. SGLang recovery retry has started with the
explicit bounded-JIT/no-autotune configuration; readiness remains pending.


## P021 - 2026-10-23: native FP8 projections and actual NVFP4 MLP parity

Added native GPU BF16-to-E4M3 activation quantization with checkpoint tensorwide
input_scale, opaque E4M3 checkpoint constants and a strongly typed TensorRT FP8
plugin. cuBLASLt uses FP32 accumulation and BF16 output; the product of the two
scalar dequantization scales is alpha, with default unit matrix scale pointers.
NVIDIA documentation explicitly defines null scale pointers as unit scales:
https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmuldescattributes-t
No full BF16 attention-weight expansion is introduced. Mixed lowering now routes
FP8 attention matrices and NVFP4 MLP matrices separately; actual attention-layer
parity is not yet established. Descriptor/heuristic creation per call remains an
optimization opportunity; no hardware instruction trace has been collected.

Native SM120 build passed. Job native-fp8-plugin-capture completed exit0: CPU-oracle
comparisons (absolute0.0625 + relative1% tolerance), serialized-engine reload and
explicit CUDA graph replay passed at flattened rows1,16,4096, including signed,
small and saturated E4M3 input values. Synthetic FP8 captured MLP outputs exactly
3.375 across256 outputs, with one graph capture, two cached replays, zero failures.
This FP8 MLP is a lowering fixture, not the checkpoint's actual MLP precision.

Added an independent bounded-memory NumPy oracle over actual checkpoint layer0
NVFP4 MLP weights. It decodes packed E2M1 weights, E4M3 block16 scales, both global
scales, BF16 rounding and dynamic activation block scales, then applies SiLU/gating
and down projection. The oracle was checked against the independently GPU-verified
identity case and rounding ties. Job actual-checkpoint-mlp completed exit0: all5120
outputs matched exactly on five executions, with max absolute and relative L2 errors
both0, one graph capture and two cached replays. Predeclared acceptance bounds were
relative L2<=1% and maximum absolute<=0.05; no post-result tolerance change.
Oracle SHA256 e5858797b5ea50034bd5fca8af808e6d02c885088aeb0e6f86ebf24984f6d92d.
Raw results: results/real-mlp-oracle.json, real-mlp-checks.json and native-projections
command ledger. This validates one real MLP, not decoder/image quality or speed.

Added explicit Qwen3.8 CUDA decoder/vision factories with the exact architecture
contract and initial mixed-checkpoint batch1/capacity32..4096 guard. They are not
registered for general serving admission; full-model construction/parity is pending.
The CPU study suite now passes21 tests. Existing unrelated ASR capture failure has
not been resolved. INT4/4B, full VLM, controlled metrics and2x remain outstanding.

## P022 - 2026-10-23: SGLang FP8 JIT linker failure isolated

The bounded-JIT/no-autotune SGLang attempt loaded the model and started HTTP, but
both smoke attempts failed health503. Process inspection showed a defunct scheduler
while the parent listener remained alive; this was not a healthy baseline. The
parent later exited9 with the actual worker traceback: FlashInfer FP8 GEMM linked
-lcublas/-lcublasLt, but unified CUDA wheels provided only versioned .so.13 files.
Guest memory had40GB available and cgroup oom counters0, unlike the earlier memory
exhaustion. Do not characterize this later failure as another OOM.

Verified installed versioned libraries, extended the guarded relative-link helper
and saved a separate cuda-cublas-layout.json without replacing earlier provenance.
Setup job sglang-cublas-linker-names completed exit0, retaining hashes for cudart,
cublas and cublasLt. No packages or library binaries were replaced. Fresh job
sglang-cublas-linker-fixed is running; text/image smoke remains pending. Installed
SGLang explicitly rejects return_token_ids with streaming chat, so benchmark token
accounting must use its supported protocol rather than assuming vLLM's extension.

P021 oracle sanity: NumPy2.4.4; all5120 expected outputs are nonzero, range
[-2.6875,34.0], L2 norm35.74675615976486. Exact comparison was not a zero-output case.


## P023 - 2026-10-23: successful SGLang VLM smoke and first real measurements

The compiler/linker-fixed SGLang attempt completed startup warmup at23:02:56UTC.
Process evidence before readiness showed active bounded FlashInfer SM120 FP4
compilation, not a dead process or another OOM. Health/text/image smoke then
completed exit0: health200, text4, imageRed. Prompt/completion counts were26/2 and
93/2; the image response reports64 image tokens. Raw requests/responses are under
results/references/sglang/smoke. This does not establish broad model quality.

Added a real loopback streaming harness using continuous server cumulative token
usage, not SSE chunk counts. It retains raw timestamped events, request hashes,
errors, outputs, source identity and configuration provenance. Warmups run at each
concurrency. Deterministic prompt nonces and one changed image pixel avoid result
reuse while keeping identical requests reproducible across engines/workload seeds.
NVML samples adapter-wide memory; sample timestamps and observed gaps are retained.
The27 CPU tests include framing through a local HTTP fixture, failure-body retention,
backward/missing token counts, batched arrivals, stable/unique images and CPU oracles.

Pilot run1e1e765e98c9471ca4923342505d6e5d completed exit0 with two measured32-token
requests per case: text median TTFT235.5008ms/decode11.8206tok/s, image246.6652ms/
11.7469tok/s. These are exploratory measurements on the initial eager profile.

Expanded run below completed exit0, eight measured32-token requests per cohort,
one warmup wave at each concurrency, total context capacity4096. Configuration:
SGLang0.5.22, Torch2.14.1+cu130, mixed checkpoint, BF16 KV/conv and F32 SSM;
CUDA graph decode/prefill disabled, startup autotuning disabled. This is an initial
reference profile, not a tuned fastest baseline or a valid2x comparison with Garnet.
Garnet currently retains F32 state/cache arithmetic, so precision matching remains
an explicit outstanding gate. Existing WorldSense services were preserved.

Run ID: da2e6f2275164c5d8be1859f62fc26bf. All measured arrivals were single-token usage increments.

| Case | Concurrency | Median TTFT ms | Median decode tok/s | Aggregate output tok/s | Sampled adapter peak GiB |
|---|---:|---:|---:|---:|---:|
| text | 1 | 242.277 | 11.625 | 8.599 | 29.0486 |
| text | 2 | 315.826 | 11.420 | 20.756 | 29.0506 |
| text | 4 | 439.623 | 11.239 | 40.819 | 29.0506 |
| text | 8 | 370.987 | 11.442 | 82.926 | 29.0506 |
| image | 1 | 237.095 | 11.855 | 11.144 | 29.0525 |
| image | 2 | 393.318 | 11.228 | 20.304 | 29.0603 |
| image | 4 | 373.803 | 11.788 | 43.049 | 29.0857 |
| image | 8 | 462.719 | 10.996 | 77.242 | 29.0721 |

The sampler requested20ms; actual median gaps were about31ms. The text/concurrency1
cohort had a7.787s maximum sample gap and lower aggregate throughput; no outliers
were removed. Its sampled peak cannot bound an unseen transient in that gap.
Investigate repeatability/observer effects before treating these as publication
measurements. All raw rows remain in benchmark-initial; the manifest embeds the
frozen packages, compiler, startup script, checkpoint metadata and NVIDIA-SMI XML.

SGLang supplied exact26 chat input IDs and generated[19,248046] for the arithmetic
smoke. Stored text-token-oracle.json avoids guessing chat templates for native
validation. Verified owned parent PID368 by executable prefix, checkpoint and port,
sent SIGTERM; both signal job and server finished exit0. GPU returned106MiB idle.
No unrelated process was stopped. D: free-space check showed1,050,915,160,064 bytes;
all work continues under the fixed study root.

## P024 - 2026-10-23: real decoder-layer parity and grouped-query build fix

Added independent NumPy mixed-projection, GDN recurrence and full-attention
oracles for real layers0/3, two stateful tokens each. Every hidden value is checked;
state samples cover uniform positions plus every full-attention KV head's active
prefix. The improved oracle SHA256 is
fd7638cdbecf1db4814de0376514bc07fa82abde6cb086c02f2bdabcd3660107f.
The original oracle2e196c3c6775a55bbdfbaf85db64b6a06b600f5b8162ff3195c83c991b267e74
is preserved separately. These compare Garnet's declared mixed/F32 arithmetic,
not an assumption of equality to Transformers' complete BF16 model.

Layer0 passed at the first attempt. The original full-attention graph failed
TensorRT build at64MiB with requested67,127,472 bytes, then at128MiB with requested
134,236,336 bytes. Both failures remain in raw ledgers. Increasing the cap was
insufficient. The verified27B graph now groups its six query heads within each KV
head for matrix multiplication, instead of explicitly repeating K/V sixfold.
Other attention geometries retain their previously validated graph. Removed the
experimental cap override; final real-layer checks use the original64MiB limit.

Combined job native-head-and-grouped-layer-validation completed exit0 on native
Windows/SM120. Both decoder layers passed CPU hidden/state comparisons and replay:

- Layer0 (linear_attention): maximum hidden absolute error0.0009765625, maximum relative L21.09443898e-05; sampled-state max absolute1.86264515e-09; captures1, replays6, failures0.

- Layer3 (full_attention): maximum hidden absolute error0.0078125, maximum relative L27.63317594e-05; sampled-state max absolute4.76837158e-07; captures1, replays6, failures0.

No model-level speedup follows from this correctness gate. Full64-layer and vision
quality remain pending.

## P025 - 2026-10-23: native W4A4 output head and memory accounting

The existing packed head intentionally used BF16 activations (W4A16), which differs
from the reference ModelOpt head. Added an explicit enable_nvfp4_head mode using
checkpoint input calibration, GPU activation quantization, tiled packed scales and
native FP4 cuBLASLt. Default W4A16 behavior remains available and tested; the initial
Qwen3.8 factory explicitly enables W4A4. No full BF16 weight expansion is introduced.
Owned-buffer accounting now includes the separate embedding/head and native head
workspace, instead of counting only the tied embedding alias.

The combined native job passed exact synthetic head outputs at batches1/3/8,
mode disable/W4A16 preservation and invalid BF16-head mode rejection. Actual pinned
head checks passed11 sparse rows against an independent activation-quantized oracle;
its SHA256 is c89a092af70c2e8883fbd89b593ddedff49c163b92be3d2dd75bcc1f5151617d6.
This is not complete language-model quality. Full64-layer token smoke is prepared
and remains the next gate; no Garnet TTFT/decode/throughput benchmark exists yet.
INT4/4B comparisons, tuned reference profiles and the2x target remain outstanding.

## P026 - 2026-10-23: full native text and comparison artifact gates

Job native-full64-text-smoke completed exit0 on CD-RTX5090. All64 decoder layers
were constructed at batch1/capacity4096 with native W4A4 head. The saved26-token
SGLang prompt produced exact token IDs[19,248046], decoded text4. The I/O report
recorded whole-device usage28,230,221,824 bytes while live; this is a snapshot,
not peak VRAM. Compilation and smoke duration are not TTFT or decode measurements.
Full VLM, broader quality, serving and performance gates remain open.

The pinned Qwen3.5-4B download completed exit0 in the same fixed study folder:
three LFS artifacts totaling9,332,636,078 bytes verified against upstream hashes.
This establishes artifact integrity, not loaded VRAM or inference performance.

Correction to the earlier INT4 config-only interpretation: actual pinned index,
bounded safetensors headers and I64 shape payload show all496 text projections,
including GDN qkv/z/a/b/out, are packed INT4. The config ignore list naming parent
containers does not establish exclusion of their child projections. lm_head,
embedding, vision and MTP remain unquantized. Packed tensors are I32[n,k/8],
scales BF16[n,k/128], shapes I64[2]; no g_idx/zero_point tensors were found.
Shape payload SHA256429e9656efdf3b7b321624eb3ecf23e3c3a51b964e35fba8d94ac56dfb9eef5d.
The installed compressed-tensors0.18.0 CPU fixture passed offset-binary low-lane-first
packing/roundtrip, helper SHA2563fe695db3e38ee665d5c459ddce366df3e2dfed51be70058bcc6a9838117477d.
No actual INT4 weight numerical/kernel test or benchmark has passed yet.
This candidate has both model.safetensors and a sidecar index; a future loader
must open the index explicitly to include vision/MTP sidecars.

Comparison download admission now uses an explicit pinned registry and per-model
disk caps, rejects unknown revisions before network/output, and preserves public
hash provenance without signed transport URLs. All31 CPU study tests pass.
Vision preprocessing still needs its own correctness gate: the pinned processor's
minimum65536 pixels differs from Garnet's existing JPEG caller minimum1024.
The initial reference224x224 PNG yields64 image tokens; do not silently compare
against49 native tokens or substitute a different encoded image in a matched run.

## P027 - 2026-10-23: explicit checkpoint JPEG preprocessing bounds

Added prepare_image_profile(path,minPixels,maxPixels) to the current native
CUDA tensor I/O. The existing two-argument method retains its original1024
minimum. The Qwen3.8 helper validates pinned normalization/patch parameters and
pixel bounds before requesting minimum65536. This is an experimental factory
helper; general serving admission is still closed.

Native build plus job native-image-profile-check completed exit0. The same
constant224px JPEG produced legacy14x14 and checkpoint16x16 patch grids, hence64
merged checkpoint image tokens. All393216 normalized patch values were finite;
maximum error against decoded RGB[254,0,0] with mean/std.5 was5.955789603717676e-7.
Three invalid pixel bounds were rejected. This exercises GPU JPEG preprocessing,
temporal repetition and patch order on a constant fixture. It does not establish
general resize parity, PNG support, vision-tower quality or end-to-end TTFT.
All32 CPU study tests pass. Full vision and matched image workloads remain next.

## P028 - 2026-10-23: preserve failed full-vision F32 comparison

Prepared an independent Transformers5.19.0 CPU vision reference using all333
actual pinned vision tensors, strict state loading, eager attention and F32 math.
The deterministic256px JPEG was processed by the checkpoint's own image processor:
grid[1,16,16], patches[256,1536], merged output[64,5120]. Raw patches, expected
outputs, JPEG, source/config/index hashes and versions remain under results/vision-oracle.
Native job native-full-vision-oracle built and ran all27 vision layers but exited1:
maximum absolute error2.8424453735351562 exceeded the predeclared0.5 limit, although
relative L2 error0.011373065543350613 was below0.02. All four executions agreed.
This is a failed gate, not vision parity. The original result/oracle are retained.

Garnet uses BF16 input/weight/output projections with separate F32 bias, norm,
residual and attention math. A second diagnostic oracle retains Transformers'
independent vision operations while explicitly rounding only those projections
and patch convolution to BF16. It uses the same error thresholds and a separate
result location; it does not overwrite or relabel the failed F32 comparison.
Its result is pending. The replay fixture also needs five calls, including warmup,
to observe two actual graph replays; the original four calls produced only one.
No full VLM or performance success is claimed.

## P029 - 2026-10-23: mixed vision diagnostic also fails; INT4 bytes verified

The mixed CPU oracle completed exit0 with111 patched projections. Native job
native-mixed-full-vision-oracle completed exit1 against the unchanged error limits:
max absolute8.0, relative L20.02335669132205647. Raw full outputs are retained as
native-vision-mixed-checks.json.f32, alongside the JSON and both CPU oracles.
The second diagnostic does not explain away the first failure. Native vision
admission remains closed while intermediate arithmetic is investigated.
The mixed expected maximum magnitude was115.25390625, native116.75390625;
at element39834, F32 expected=-81.833984375, mixed expected=-74.75390625,
native=-82.75390625. Do not infer whole-VLM quality from these observations.

The pinned INT4 candidate download completed exit0: four LFS artifacts totaling
19,437,778,517 verified bytes. No duplicate checkout or checkpoint copy was made.
The extracted small metadata separately verified for all three checkpoints:
NVFP4=15 files,4B=11,INT4=12. Payload verification remains the downloader's full
SHA256 gate; the metadata audit does not imply runtime/quality/kernel support.

## P030 - 2026-10-23: prefix diagnostics and bounded F32 structural check

Explicit F32 accumulation on rounded BF16 CPU projection operands still failed
the retained full27-layer output comparison: max absolute5.0, relative L2
0.01854415012578696. No threshold was relaxed. Native prefix towers plus the real
merger also completed exit1 against the original absolute limit, while retaining
all three observations before assertion:

| Prefix depth | Maximum absolute error | Relative L2 error |
| --- | --- | --- |
| 1 | 2.0 | 0.0011233083247052845 |
| 3 | 4.0 | 0.001983842689693945 |
| 9 | 4.0 | 0.0029497819085522333 |

These do not establish the cause or whole-VLM quality. A bounded diagnostic
attribute now requests F32 vision projection constants/operators, leaving the
default BF16 path unchanged. It is restricted to batch1, at most256 patches,
hidden1152/out5120 and at most27 blocks. The existing captured strict-F32 policy
disables TF32. The diagnostic compares against the original untouched full-F32
Transformers oracle with predeclared tighter max0.05/relative0.001 limits. Its
result is pending; this extra F32 vision weight storage is not a serving profile,
an NVFP4/INT4 arm, or a performance claim.

The first native evidence archive preserves the full text smoke, JPEG gate,
both failed full-vision comparisons, CPU fixtures, binary hashes and raw command
ledgers. SHA256 b582054dbe3866a1e3c22b3f35c1e41a9ba3b564052693598d06fc53274e2781,
3,658,541 bytes,31 entries, verified after transfer. Later prefix/F32 structural
diagnostics are separate artifacts and are not silently represented as part of
that immutable archive.

## P031 - 2026-10-23: full F32 vision structure and CUDA replay pass

Job native-fp32-vision-structure completed exit0 on CD-RTX5090, with all27 real
vision blocks, the real merger and the original untouched F32 Transformers oracle.
Across five executions, maximum absolute error was0.000576019287109375 and relative
L2 error2.874012096435269e-6. The tighter0.05/0.001 limits and at least two actual
CUDA graph replays passed. This supports the structural vision implementation;
it does not retrospectively pass the separate BF16 comparisons or establish
whole-VLM quality. The only requested arithmetic change in this diagnostic is
F32 vision constants/projections; the ordinary BF16 path remains the default.
No vision timing, serving benchmark or2x result follows from this check.

Next: validate complete native image-to-text behavior against matched reference
requests, investigate/quantify low-precision output agreement and broader quality,
then admit the experimental batch1/4K serving profile and measure it. INT4/4B
runtime arms, tuned baselines, precision matching and the2x objective remain open.

## P032 - 2026-10-23: diagnostics archive transfer verified

The separate vision-diagnostics-evidence.zip contains24 entries and9,967,800
bytes: prefix failures/raw outputs, the explicit-accumulation CPU oracle,
successful full-F32 structural outputs, exact tested source snapshot, binary
hashes and command ledgers. Remote/local SHA256 matches:
6fe4ce494cfc12b512150f33ed1f5f76ba6fb25af51941f19a6ea125dba32d4e.
Local ZIP integrity check passed. Both immutable native archives and the initial
SGLang raw archive are available with an evidence index in the chat outputs.
No GPU benchmark/reference server was left running by these diagnostics.

## P033 - 2026-10-23: matched JPEG reference and full VLM probe

Restarted the same initial/eager SGLang profile only after native GPU diagnostics
had closed. Matched reference job matched-jpeg-vlm-reference-smoke completed exit0:
solid red/green/blue JPEGs returned Red/Green/Blue, and a red-top/blue-bottom JPEG
returned blue to a bottom-half question. Each image uses the same224px JPEG bytes
saved for native input, yields64 image tokens and requests disabled thinking.
The reference returns exact prompt/generated token IDs, image/wire/response hashes
and usage. This four-case color/spatial smoke is not broad VLM quality or timing.

Reference image content is explicitly before question text, matching Garnet's
current image-first request formatting. Earlier PNG measurements had text-first
content and must not be silently treated as these matched requests. The native
probe verifies full prompt tokenization/image hashes before model construction,
uses the normal BF16 vision tower and all64 mixed decoder layers, resets state
between cases and compares both generated token IDs and semantic colors. It
remains a correctness probe, not general serving admission.

A separate exploratory native-call text pilot reuses the same model load and
the eight exact text/c1 request bodies from SGLang run da2e6f2275164c5d8be1859f62fc26bf.
It records actual greedy IDs and synchronized per-token monotonic timestamps,
one excluded warmup, eight32-token requests, batch1 and4K capacity. This includes
native tokenization/reset/prefill/sampling and excludes HTTP; its TTFT/throughput
must not be represented as a matched client HTTP result or proof of2x. It is
intended to identify native bottlenecks before further optimization. Results pending.

The native supervisor records a separate pilot manifest keyed to this invocation
to reject stale results. NVML sampling covers the entire build/correctness/pilot
job; its sampled adapter maximum includes builder allocations and must not be
called a warmed serving peak. The measured direct-call subset remains separate
from the outer correctness process duration. Exact runtime binaries are hashed
after execution. The matched SGLang parent was signaled only after verified
ownership, observed to exit0, and the GPU returned to106MiB/0% before native work.

## P034 - 2026-10-23: first full native VLM observations and direct-call timings

Run11e0c7a7341b4862a8a4ccd1430b7b66, source57f432ef8, completed exit1 because
the strict reference-token assertion failed. All64 mixed decoder layers and the
normal BF16 vision tower executed on the RTX5090 at batch1/capacity4096. Exact
image hashes and full prompt IDs matched before construction. All four semantic
color/spatial checks passed; strict token parity passed two of four:

| Case | Garnet | SGLang | Exact generated IDs |
| --- | --- | --- | --- |
| red | Red | Red | pass |
| green | green | Green | fail |
| blue | blue | Blue | fail |
| bottom-blue | blue | blue | pass |

This preserves the failed strict gate; case-insensitive semantic success is a
separate observation, not a retroactive change of its acceptance rule. It also
does not establish broad VLM quality or explain the separate numerical vision
differences. General serving admission remains closed.

The independent direct native-call text pilot completed: one excluded warmup,
eight measured requests,256 actual generated IDs,32 tokens/request and41 prompt
tokens/request. Median first-token time1060.208299999431ms, median decode
37.318686598767705tokens/s, cohort output16.69191732112634tokens/s. These include
tokenization, state reset, per-token embedding/serial prefill and greedy sampling,
but exclude HTTP transport. They are not matched HTTP results or proof of2x.
The initially slow serial prefill is now an observed optimization target.

Whole-job sampled adapter maximum28,536,557,568 bytes;16,580 samples, requested
20ms interval, observed median31.18521ms/max218.7191ms gaps, no sampling errors.
Its window includes build, warmup, pilot and VLM correctness, not just warmed
serving. Pilot SHA256 f15cceb8bd3b6f3aa13f5cdf292f4902de1ff2ab237a09f70e6a9afc168d0b9e.
All raw timestamps, greedy IDs, requests, failures, source and binary hashes remain
saved. Importing the embedded runtime's hashlib emitted nonfatal unavailable
SHA3/SHAKE messages; SHA256 verification and subsequent execution succeeded.

Next optimization uses the existing Garnet fused GDN/attention chunk programs,
not another inference stack. Prepared real-layer0/3 checks against18 sequential
CPU steps at4K capacity, testing chunks4/16 plus a two-active-token continuation
and inactive padding, future KV guards and CUDA replay. One engine/weight set
must serve prefill and decode to avoid duplicating the compressed model in VRAM.
These component/whole-model optimized results are not yet available.

## P035 - 2026-10-23: fused GDN passes; four-token attention check fails

The18-step CPU oracle completed exit0, SHA256
9ccb239453c44da31d6d978628a2d78a6a66331583b68b38a741faf01a48e9d0.
Native job native-real-fused-chunks4-16 completed exit1. Layer0 fused GDN passed
four active tokens followed by two active/two inactive tokens, state continuation
and replay. Layer3's four-token hidden output failed max absolute0.25/relative
L20.015542778450663965 against the unchanged0.05/0.01 limits. The batch script
therefore did not run its16-token check. No optimized full-model timing exists yet.

This does not yet locate the discrepancy in the attention kernel: the earlier
serial real-layer gate covered only two steps. The next diagnostic broadens
serial coverage to six steps, records per-token and cache-state metrics before
asserting, and tests4/16 profiles independently. This distinguishes a fused-core
regression from batched projections, quantization/rounding or a newly exposed
baseline/reference difference. Strict cutoffs remain unchanged; raw failures stay.

The staged whole-model probe can select chunk4/16 explicitly, keeping one engine
and compressed weight set per layer for both prefill and decode. Its chunked
prefill uses the current Garnet batched embedding/prefill_sequence flow. This
profile is prepared, not admitted or measured, pending the component gates.

The full-vlm-pilot-evidence.zip transfer was verified:559,813 bytes,24 entries,
SHA256901e7baac7c7af515156393680ae089bc516bec0255c47247062910feb31f6c4.
It includes source snapshots, native raw token timings, NVML records, matched JPEG
reference requests/responses and the strict VLM failure. CUDA memory snapshot
29,194,911,744 bytes is also retained separately from the lower NVML sampled
maximum; the APIs have different measurement scopes and are not equated.

## P036 - 2026-10-23: broadened serial test also fails; audit FP8 oracle scaling

The serial6/fused4/fused16 diagnostic completed exit1 with all raw observations
retained. Serial GDN passed all six steps, but serial attention failed at steps3/4
(zero-based), with max absolute0.25/0.1899414 and relative L20.031518/0.0331484.
Fused4 shows the same large errors and sampled attention-state maximum
0.007527589797973633 as serial. Fused16 attention also fails; fused16 GDN fails
the absolute hidden limit at later steps, although relative L2 is below0.01.
These observations contradict an assumption that the discrepancy was introduced
only by the fused attention kernel. The original two-step serial result remains
valid only for its narrower coverage.

The CPU FP8 oracle currently dequantizes weights and activations separately
before its dot product, whereas the native cuBLASLt code combines the two F32
global scales in alpha after the raw FP8 accumulation. Both express the same
real-valued operation but can round differently before BF16 conversion. Added an
explicit optional alpha-after-dot CPU diagnostic; default historical arithmetic
is preserved and old oracles are not overwritten. A BF16 midpoint fixture checks
the native placement of the rounded F32 alpha; all34 CPU study tests pass. The
actual checkpoint diagnostic result is pending. No numerical limit was relaxed,
no native kernel changed, and no optimized full-model speedup is claimed.

## P037 - 2026-10-23: scale-order correction resolves the component gate

The new CPU oracle596bd1cf8909e0ebcdd1fdb3b7989a93bedcc9b23b3c257d17c80b709a0de6a9
uses the native FP8 descriptor's combined F32 alpha after raw FP8 accumulation.
Comparing retained, unchanged native outputs isolates that arithmetic change:
four-token attention max absolute error drops0.25 to0.000244140625, and its
two-token continuation0.18994140625 to0.00006103515625. Sixteen-token GDN max
drops0.5 to0.001953125. Original references, raw outputs and failed runs remain.

Job validate-native-order-chunk-oracles then completed exit0, re-executing serial
six-step layers0/3 and fused4/16 layers0/3 against the corrected reference. All
per-token hidden, sampled cache/state, inactive padding and CUDA replay gates
passed with the same max0.05/relative0.01 limits. No native kernel changed.
This fixes the reference mismatch, not the separate vision precision or VLM
capitalization differences. It does not prove any performance improvement.

The next full-model pilot selects the already existing fused16 path explicitly,
with batched embeddings and real chunked prefill. The same16-token engine serves
padded single-token decode, so no second compressed model copy is retained.
It preserves batch1,4096 capacity and the same eight text request bodies and four
matched JPEG smoke cases. Results are pending. General serving/tuned comparisons,
INT4/4B execution and the2x objective remain incomplete.

## P038 - 2026-10-23: full fused16 pilot measured; strict parity remains open

Run601a6376c8fa4c0cbf0e3b3790b9aa69 tested source e5adb6d9f0cda0c683f66386cd43a19b49ccee96 on the single RTX5090. Existing fused16 engines served chunked prefill and padded one-token decode using one compressed weight set. Batch1,4096 capacity,41 actual prompt tokens,32 generated tokens, eight measured requests and one excluded warmup match the earlier direct-call pilot request bodies.

Synchronized direct-call median TTFT was99.79560000010679ms, median decode36.489412479650696tokens/s, and cohort output32.951705388477315tokens/s. Earlier serial pilot values were1060.208299999431ms,37.318686598767705tokens/s and16.69191732112634tokens/s. These are exploratory single runs, exclude HTTP, and do not establish a matched2x advantage against either reference. No concurrent serving measurement was performed.

All four JPEG color/spatial semantic smoke checks passed. Strict token parity improved from2/4 to3/4 but bottom-blue differed (native6105 versus reference11855, followed by EOS), although both decode as blue. The overall job correctly remained exit1. General serving admission and broader quality remain closed; no assertion or tolerance was relaxed.

Whole-build/correctness/pilot sampled adapter-wide NVML maximum was27,810,942,976bytes over13,398 samples. Observed gaps median31.6873ms/max1.3155206s; no sampler errors. This is neither an instantaneous peak nor a warmed serving/process-only measurement. Source identity was unchanged before/after. Native DLL and VM hashes match the earlier pilot. Raw pilot SHA256 is47c86d5795f1ec0c1f67ef69970736d0704d3be9bb25602618623fb936bcf455f.

Next: resolve/characterize exact-token quality differences, expose true streaming events through current Garnet serving, and establish tuned matched CUDA-graph references before any2x claim. INT4 W4A16,4B and concurrency execution remain outstanding. The active goal is not complete.


## P039 - 2026-10-23: explicit matched JPEG workload and graph candidate

Added explicit benchmark image format/order options. Historical defaults remain PNG/text-first; new native-comparable runs must select JPEG/image-first on both engines. JPEG is encoded once into each saved request body; actual wire bytes remain available for hash matching. A new test checks MIME/decoded format, dimensions, base64 bytes, deterministic requests, changed nonce pixels and unchanged text requests. All35 CPU study tests pass. No earlier PNG result is relabeled as a JPEG result.

Installed SGLang0.5.22 field definitions were located under arg_groups/fields/exec_.py and schedule.py, rather than the earlier shallow scan. Source-hashed audit confirms full decode graph backend, explicit decode batch sizes and max_total_tokens. Added a separate candidate launcher capturing1/2/4/8 with32768 KV pool tokens (eight times4096), retaining eager prefill and all initial precision/cache/attention choices. It is not called tuned or faster before startup, correctness and measurements succeed. Metadata preparation now accepts the actual launcher and a fresh output path to avoid accidentally hashing the old eager launcher.

WSL direct health returned actual uptime; the slow shell inspection eventually completed and was not restarted. No native study xlang process was live before reference startup preparation. Graph candidate results remain pending;2x is unproven.


## P040 - 2026-10-23: actual SGLang graph replay; native stream transport groundwork

The graph candidate accepted its configuration, loaded all three shards, captured decode batches8/4/2/1 and reported cuda graph:True during actual requests. All four JPEG reference smoke cases passed before benchmark run81b4a8ebf3d14153b487d9ad40cdb4f4. Its first completed client cohort (eight requests,41-token text prompts,128 actual output tokens each) measured median TTFT141.90375ms, decode65.0572378tokens/s and cohort61.0921372tokens/s. These differ in length/transport from the native32-token direct-call pilot and are not a matched ratio. Remaining cohorts were still running at this entry. This stronger reference makes the earlier eager baseline unsuitable for the2x claim. Autotuning and prefill graphs remain disabled in this candidate, so it is not declared the fastest attainable reference.

A CPU-only native VM probe emitted an immediate JSON ready event, but sys.stdin iteration failed with invalid memory stream object (exit1). An unchanged-VM os.read/os.write framing probe then returned ready and the supplied value7 with exit0. Neither probe initialized Garnet/CUDA or ran inference. Added a bounded JSON-line transport using direct descriptors, handling split frames, partial writes, EOF and malformed/oversized requests. Added a research-only single-slot Qwen3.8 worker over the current64-layer mixed decoder and default BF16 vision. It emits token IDs/counts/timestamps immediately after synchronized predictions and retains full prompt/generated IDs; no buffered-final-text timing substitution. It uses one fixed JPEG input file and supports only the explicit image-first JPEG profile. The general ServeModel admission gate remains closed.

Worker full-model integration and HTTP transport are not yet validated. Five new CPU tests cover framing/partial writes and multimodal rotary boundaries, including non-square geometry; all40 CPU study tests pass. Python syntax compilation passed. No remote source resync or native GPU inference is performed while the reference benchmark is live. Next: preserve the completed reference evidence, stop only its verified-owned parent, then validate the native streaming worker against the retained request bodies before implementing HTTP measurements.


## P041 - 2026-10-23: eight graph reference cohorts completed

Run81b4a8ebf3d14153b487d9ad40cdb4f4 completed exit0: eight measured requests per cohort, two excluded warmup waves per concurrency, maximum128 output tokens,4K capacity, JPEG/image-first. Live server-info confirms full decode graph backend, capture1/2/4/8, max_total_num_tokens32768 and ready status; runtime logs confirm actual graph replay.

|case|concurrency|TTFT median ms|decode median tokens/s|cohort output tokens/s|actual output tokens|sampled peak bytes|
|---|---:|---:|---:|---:|---:|---:|
|text|1|141.90374999998312|65.05723780587802|61.092137216423644|1024|28030320640|
|text|2|264.3982000008691|63.168771583808265|111.93046739453078|1024|28036612096|
|text|4|338.55524999944464|60.707432695421744|210.35341859942818|1024|28036612096|
|text|8|315.3050000000803|60.48445245605198|423.5778487481452|1024|28101623808|
|image|1|143.87115000045014|61.819192916561335|57.0954999660933|910|28101623808|
|image|2|314.3764500000543|59.92573981179359|104.94914547720484|1024|28112109568|
|image|4|353.6734499994054|57.492526278726665|199.61835076684082|1024|28118401024|
|image|8|359.6580500006894|55.77625939567659|351.1103320328398|936|28279881728|

All measured arrivals carried single-token cumulative usage increments. Some image responses ended before128 tokens; actual counts910 at concurrency1 and936 at8 are retained without padding or discarded requests. Sampling errors were empty; actual per-cohort gaps and adapter-wide scope are in the manifest. These remain exploratory reference measurements until native transport, request hashes, lengths and quality are matched. Autotuning/prefill-graph tuning and vLLM graph baselines remain outstanding. No2x claim follows. The verified-owned parent369 received SIGTERM after the benchmark, with exit observation pending at this entry.


## P042 - 2026-10-23: real HTTP bridge ready for GPU integration test

SGLang graph parent exit0 was observed after verified SIGTERM; subsequent nvidia-smi showed106MiB and0% GPU utilization. No reference engine is left running before native startup.

Added a loopback-only CPython HTTP bridge to the native token-event worker. It drains real per-token events from the child pipe, checks request IDs, cumulative counts and final IDs, and streams immediate SSE with actual usage. A single native slot serializes concurrent clients; queue time stays inside client TTFT. It is not advertised as native batching. Raw child stdout/stderr are retained. Images use one fixed JPEG file. General ServeModel admission is unchanged and broad/reference quality gates remain open.

Two real child-process/HTTP CPU fixture tests validate separate token arrival times, cumulative usage, final IDs, Unicode prefix buffering/final text, non-stream replies and rejection of wrong input profiles. The first fixture failed because Windows wrote its Unicode source in the local encoding; explicit UTF-8 corrected the fixture. All42 CPU study tests pass, plus syntax checks. Fixture timestamps are never model performance evidence.

Added GPU validation driver: wait for real health; replay all four retained JPEG request bodies; require original native fused16 generated IDs and reference prompt IDs to match; retain the separate SGLang parity result without weakening it. Then require actual single-token HTTP arrival accounting on the seeded text/image requests, before eight-request128-token B1 timing with the same seed3/JPEG/image-first body generation as SGLang. Warmups and raw events are retained. Full-model startup/validation remains pending at this entry. No2x claim.


## P043 - 2026-10-23: replayed raw reference accounting while native build continues

Native HTTP worker10972 remains live under the retained server handle. CPU time increased60.8 to110.0 to136.2seconds and working set increased while its readiness-gated validator waited; silence was not treated as exit or restarted. At remote09:15:38 it was still constructing the model. The running source remains4023c78c7; local documentation/comparison edits were not resynced.

Updated the implementation task to remove stale pending statements for the completed fused16 and SGLang graph experiments. Added a raw HTTP comparison verifier that replays saved SSE cumulative usage/timestamps/text, recomputes request digests and per-cohort summaries, then checks identical request bytes and actual prompt/output lengths before presenting provisional ratios. It never infers model quality, arithmetic equivalence or an established2x win from accounting alone. Missing vLLM/SGLang coverage and unavailable single-token decode rates remain explicit.

Ran it against the verified SGLang graph archive directly: all8 cohorts and64 measured requests replayed successfully from actual request/event/result files, with summary agreement. The derived accounting report is exported separately; it does not replace raw evidence. Two corruption/matching CPU fixtures pass, and all44 CPU study tests pass. Full native HTTP output and paired ratios remain pending. General serving, native concurrency, tuned vLLM, INT4,4B and the2x objective remain incomplete.


## P044 - 2026-10-23: first native HTTP pilot completed; target missed

Source4023c78c73e30c33d5fac4430dfd5be813be3185 completed the full-model HTTP regression/streaming/pilot driver exit0. All four original JPEG bodies retain exact reference prompt IDs and original native fused16 output IDs; semantic4/4, SGLang exact parity3/4 remains unchanged. Both text/image actual single-token SSE usage gates passed before timing. Runcd8086d19d6a4118a888f93f598b9f6e used one native slot, capacity4096, seed3, JPEG/image-first,128-token maximum, eight measured requests and two excluded warmups per case.

|case|TTFT median ms|decode median tokens/s|cohort output tokens/s|actual output tokens|sampled peak bytes|
|---|---:|---:|---:|---:|---:|
|text|104.84710000037012|36.09678993226741|35.22800604463035|1024|27806748672|
|image|250.95925000005082|36.02435445492287|33.82502845918804|1024|27810942976|

Raw replay verifies both native cohorts and compares against the archived SGLang graph run. Text request bytes, prompt counts and actual output counts match exactly; provisional native/reference ratios are0.554846642 decode,0.576637316 output throughput and0.738860671 TTFT. Image request bytes match but output lengths differ (native1024 versus reference910 total tokens). Image ratios are therefore not length-matched. Native and reference precision/quality/repetition gates remain open, and vLLM graph data is absent. Neither2x speed nor halfTTFT is achieved; no win is claimed.

Native NVML errors were empty; median observed gaps about31.99ms/max33.87ms, with adapter-wide sampled maxima scoped to measured cohorts. Readiness/build/correctness memory is separate. The HTTP controller was stopped only after validation completed, using verified bridge PID15296/parent14396 and exact study command checks. Its parent ledger records intentional operator termination4294967295; this is not the benchmark's exit0. Native child10972 subsequently disappeared and GPU memory returned106MiB/0%. Original unrelated processes were untouched.

Verified native-http-b1-evidence.zip:87 entries,573373bytes, SHA256 bdd034d97d5beb7a65c8a1eaf161e15abcb1ee67566c6607a3d142036d8ff3ae. Contains raw requests/SSE/token IDs, controller/native logs, sampled memory, before-stop metadata, source snapshot and operator-stop identity. The separate replay/comparison report preserves all unmatched gates.

Next optimization is based on current Garnet's existing packed whole-decoder program rather than another execution stack. Inspection shows its current readonly projection input pool requires BF16; embedding the verified mixed compressed projection constants into one bounded program would reuse existing NVFP4/FP8 lowering and avoid64 per-layer graph submissions. This is a candidate, not demonstrated speedup. Start with a four-layer serial-versus-packed hidden/state/replay regression; only after it passes attempt full64 with a single16-token profile, preserving one weight set and4K capacity. Do not instantiate separate embedded prefill/decode copies. Native batching/memory admission, MTP, tuned vLLM, INT4 and4B remain required work.


## P045 - 2026-10-23: embedded compressed whole-program four-layer candidate

Added an opt-in embedded_projection_weights mode to the existing CUDAPackedDecoderProgram. Default external BF16 behavior is preserved. Empty external projection bindings route through current checkpoint lookup and native mixed NVFP4/FP8 lowering; BF16 joined input pools are rejected in embedded mode. No new CUDA kernel, VM source or full-model serving admission changed. No separate embedded prefill/decode models are created.

The GPU regression compiles actual layers0..3, one16-token program, capacity4096, and compares against current per-layer CUDADecoder with identical fused cores. It exercises16 active tokens then2 active tokens with14 inactive padded lanes. All hidden and every conv/SSM/K/V state word, including future KV slots, are snapshotted; diagnostic copies remain bounded to1M F32 words. Each chunk runs eager plus four CUDA-enabled executions and retains every snapshot/hash. Stats must show12 NVFP4 and13 FP8 projection weights, zero external BF16 weight inputs, graph replays and no capture failures.

A separate CPython comparator verifies snapshot hashes and compares hidden and each layer/state segment separately with unchanged max0.05/relative0.01 limits, avoiding a large cache masking a small hidden-output error. Two new CPU tests cover that masking case, invalid spans, nonfinite values and truncated bytes. All46 CPU study tests pass; syntax checks pass. This is a four-layer cross-program regression, not independent CPU math, full64-layer quality or a speed benchmark. GPU execution is pending at this entry. Original source/model folders remain fixed; source transfer explicitly includes the changed packed-program module.


## P046 - 2026-10-23: embedded four-layer GPU regression passes exactly

Job packed-embedded-four-layer-gate completed exit0 on source b454c1cdc67b4b5b857b800b14f6300903d8503d. All ten packed snapshots (five executions each for16 active tokens and a two-token continuation with14 inactive padded lanes) are byte-identical by SHA256 to their corresponding per-layer reference snapshots. CPython checked every hidden, conv, SSM and full K/V word in separate segments; all max absolute and relative errors are0. Each comparison spans10,952,704 F32 words; no future KV slots were omitted. Full files/digests and command/binary/source identities are retained. This is a cross-program regression using actual checkpoint weights, not independent CPU math, full-model quality or performance.

Actual program stats show12 NVFP4 projection weights,13 FP8 projection weights, zero external BF16 inputs, six CUDA graph replays and zero capture failures. No CUDA kernel or numerical threshold changed. All46 CPU study tests and syntax checks pass.

The next research-only factory uses existing CUDAPackedDecoder with embedded projections, enforcing equal prefill/decode chunk16 and rejecting BF16 joined pools so no second compressed model is retained. Full64-layer construction must report192 NVFP4 and208 FP8 projections and zero external BF16 inputs; the separate output head remains native W4A4. Ready metadata now records the selected decoder layout. A fresh named result directory preserves the original HTTP pilot; the same prompt/native-output/streaming gates precede timing. Canonical state/request buffers remain explicit. Corrected packed owned-byte accounting includes separate embedding/untied-head/native-head-workspace allocations consistently with the layer decoder; NVML remains the authority for sampled adapter-wide peak.

Full64-layer packed GPU construction and HTTP regression are pending. General serving admission, precision/broader quality equivalence, native batch/concurrency admission, tuned vLLM, INT4,4B and2x remain incomplete.


## P047 - 2026-10-23: full packed pilot completed; no decode improvement

Source ec6b3519b04b0cdce3a61449705a245ee3a36462 completed full64-layer packed construction and HTTP validation. All four saved native output-ID sequences and reference prompt-ID sequences remain exact; semantic4/4 and SGLang exact3/4 stay separate. Both streaming gates passed. Run08421c581fd44b85a9a653b85a5aea49 completed exit0 with the same seed3/JPEG/image-first128-token maximum, eight measured requests and two excluded warmups per case.

|case|TTFT median ms|decode median tokens/s|cohort output tokens/s|actual tokens|sampled peak bytes|
|---|---:|---:|---:|---:|---:|
|text|123.04409999978816|34.68293228334323|33.79564399165833|1024|24512122880|
|image|260.10540000061155|34.64965096686278|32.53912955051301|1024|24518414336|

This single pilot did not improve decode versus the preceding per-layer HTTP run36.10/36.02tokens/s. Sampled adapter-wide maxima are about3.29GB lower, but this is not evidence of a2x speedup or repeated statistical significance. Sampling errors were empty; raw per-cohort intervals and scope are retained. The packed path remains optional. Source and binaries were frozen during measurement; the large four-layer archive transfer finished and was verified before these timing cohorts. The component archive verified113835503bytes/17entries, SHA2561788367ae479f8045ae6046bf6ad53b582b973cca0b33ca0e32fd9343550de4a.

After benchmark exit0, only verified packed bridge2832/parent9016 was stopped, with command/parent identity recorded. Server parent records intentional operator termination4294967295. Child14360 disappearance and GPU106MiB/0% were subsequently observed. Protected unrelated processes were untouched. Packed raw/source/command archive SHA256724e7148d369f042b5468108c266d46bd6404db0a3b39cad2e1724d63ebaa91f; local download verification pending at this entry.

Prepared a separate vLLM decode-graph candidate using installed audited FULL_DECODE_ONLY mode, capture sizes1/2/4/8, bounded compilation workers and explicit4GiB cache budget. That budget alone does not prove eight simultaneous4K admissions: inspect resolved allocation and test admission before claiming it. Eager baseline is preserved. No vLLM graph performance is claimed before actual startup/quality/streaming/timing.

Read-only installed-source audits confirm vLLM returns prompt IDs at response top level and output IDs in choice.token_ids, unlike SGLang choice.prompt_token_ids/response_token_ids. Its shared protocol supports continuous_usage_stats. Extended the JPEG reference probe with an explicit engine selector, preserving default SGLang request bytes and omitting its unsupported prompt-ID request flag for vLLM. Two fixtures reject missing/wrong-layout/boolean/out-of-range IDs and context overflow. All48 CPU study tests pass. No new code was resynced into the live native run. Next: collect the vLLM graph reference, then profile projection/kernel costs and assess the current Garnet MTP path; removing64 host graph submissions alone did not deliver the target.


## P048 - 2026-10-23: lossy nonce collision discovered and corrected for future runs

Raw archived image bytes expose a cache-matching issue: the SGLang JPEG graph run has62 image requests but52 distinct encoded images (10 duplicate groups), and the original native HTTP B1 run has10 requests but9 distinct images. A one-pixel RGB nonce does not necessarily survive JPEG compression. This does not prove actual reference encoder-cache hits, but it invalidates the earlier blanket claim that the pixel nonce establishes uncached vision. Native code always recomputes vision; reference cache behavior remains unproven. Existing image timing results stay archived and exploratory; no source, metric or failure was rewritten as a matched uncached result. The separate uniqueness audit lists every duplicate request name/hash.

Added an explicit binary-grid image nonce:24 visible8x8 black/white cells in a64x24 bottom-right region encode the bounded24-bit seed. It is a different declared workload, not a relabeling of old data. Tests decode JPEG pixels at bit centers across carry boundaries and verify distinct encoded bytes AND decoded RGB for all62 seed3 image warmup/measured requests. Future benchmark CLI defaults to binary-grid; legacy pixel is explicit and marked unproven. The payload API retains historical defaults for fixture compatibility. The harness now prepares all request bodies before timed dispatch and saves encoded/decoded image hashes; binary-grid collisions abort before timing. Old and new preparation policies remain distinct in manifests. SGLang/native image cohorts must be rerun under the same new profile before a final image comparison.

The JPEG reference probe's installed-vLLM field mappings and new nonce checks pass; all50 CPU study tests pass, plus syntax checks. The vLLM graph driver waits for actual health and four JPEG/token-ID smokes before eight-request128-token text/image cohorts at concurrency1/2/4/8, explicitly selecting the new grid workload. The next source transfer/start is pending here; no graph result or eight-full4K admission is claimed.

The full packed archive also verifies locally (593213bytes/87entries). A terminal-wrap transcription duplicated one hexadecimal character in its initially logged digest; the corrected64-character SHA256, checked against four remote16-character pieces and the local file, is724e7148d369f042b5468108c266d46bd6404db0a3b39cad2e1724d63ebaa91f. Artifact bytes did not change. Raw packed event/request replay verifies both cohorts; compared with the prior native per-layer pilot, text decode ratio0.960831485/output0.959340246/TTFT1.173557495 and image decode0.961839608/output0.961983804/TTFT1.036444761. Both native runs have identical request bytes and output counts; broader quality remains separate. No2x claim follows.


## P049 - 2026-10-23: speculative streaming/count groundwork, no MTP GPU claim

The vLLM graph server and readiness-gated benchmark handles remain live on frozen source4b5aed30e. Logs show successful torch.compile and an SM120 FlashInfer GEMM JIT build in progress. No second GPU engine was started. Local changes below were not resynced into this run.

Inspected current Garnet MTP/verification code and pinned NVFP4 shard headers. The checkpoint contains15 BF16 MTP tensors, including fc[5120,10240], gated q_proj[12288,5120], k/v[1024,5120], o[5120,6144] and the5120/17408 MLP geometry. This is a header contract observation, not MTP numerical equivalence or performance proof. Native full-head token prediction supports the NVFP4 mode; indexed proposal-head prediction explicitly rejects packed NVFP4 heads, so that shortcut must remain disabled until separately implemented/tested.

Added optional MTP callbacks to existing CUDASpeculativeSession: first target prediction is exposed immediately after synchronized target prefill and before remaining draft work, using a copied token list; generate notifies only sampled current/target-verified accepted tokens, including EOS counts. Rejected draft tokens are never notified. Default calls without callbacks preserve behavior. These callbacks are prerequisites for real streaming; replaying final generated text as artificial token arrivals is not used.

Five CPU tests load the exact production control methods (isolated from native GPU imports) with deterministic verifier fixtures. They check rejection/non-emission, verified EOS counted once, emission before correction work, callback-failure poisoning/rollback, unmodified no-callback output, and first-token visibility before MTP work without mutating the target. They do not simulate model performance or prove GPU math. All56 CPU study tests and syntax checks pass.

Corrected decode accounting for a multi-token first arrival: use(N-first_arrival_batch_tokens)/(last-first), rather than always subtracting one. Direct/single-token streams retain the original formula. Invalid first-batch counts are rejected; actual request/token totals and TTFT are unchanged. Raw replay verification now checks saved first/largest arrival batches and single-token status against SSE. Replayed12 actual archived cohorts/96 requests: all had first batch1 and their published summaries remain unchanged. The currently running vLLM source still has the prior arithmetic; inspect its first-batch data and derive corrected summaries from raw events if necessary rather than silently mixing definitions. No MTP integration or2x result is established.


## P050 - 2026-10-23: guarded MTP prototype and confirmed reference link failure

Added opt-in packed-target MTP integration, default disabled. The checkpoint gate checks all15 BF16 draft tensor names/shapes/dtypes; only depths1/3/7 and the full native NVFP4 head are allowed. Streaming emits real target/verified tokens and checks callback IDs/counts against final output. Before timing, the new driver requires the original four VLM gates and16 complete same-target greedy/MTP token sequences to agree, retaining both raw responses and speculative counters. CPU control/layout/gate fixtures are not GPU equivalence evidence. All61 CPU study tests pass; git diff --check passes. No MTP GPU execution or speed is claimed.

The vLLM graph startup terminated exit1 after28 SM120 compilation objects at final linking: cannot find -lcudart. Existing pinned CUDA wheel contains libcudart.so.13 but no unversioned alias/lib64 layout. The original failed log, metadata and source snapshot remain intact. A narrow existing-library layout repair was launched; it remains pending, with no retry or reference measurement yet. The readiness-only benchmark waiter was intentionally stopped (exit4294967295); no benchmark cohort was produced. GPU allocations remained8984MiB/0% at the subsequent Windows observation, so process shutdown must be established before another engine starts. No reboot or broad package upgrade performed. Goal remains active and2x unmet.


## P051 - 2026-10-23: cached reference extension links successfully

The pinned CUDA-library layout repair completed exit0 and records versioned library SHA256 plus CUDA13.0.88. Relinking reused all28 compiled objects: exactly one final link, no object recompilation. The first standalone relink lacked the launcher's LIBRARY_PATH and failed on -lcuda; this operator setup error is retained. Repeating with LIBRARY_PATH=/usr/lib/wsl/lib (the existing launcher setting) completed exit0. Append-only remote ledger: results/vllm-link-repair-ledger/commands.jsonl and commands.log. This establishes successful extension linking only, not model initialization or reference performance. Original startup failure remains preserved separately.

No reference server or MTP GPU trial was started. Failed Linux server301/608 were absent in process inspection. Desktop applications now occupy roughly9.6-10GB VRAM (latest9999MiB used/22189MiB free), versus prior idle106MiB, leaving inadequate headroom for the measured27B configurations. User was asked to free GPU-heavy applications; they were not terminated. All61 CPU study tests pass; prototype commit2986ff882 pushed to qwen38-5090. Remote model source remains4b5aed30e until a controlled future sync. Goal active;2x remains unmet.


## P052 - 2026-10-23: deployed guarded MTP and prepared INT4 compatibility diagnostic

Remote sync verified6432 source files with no mismatches at24cca64fb. All61 CPU study tests passed again on CD-RTX5090 through the append-only mtp-cpu-gates-ledger, exit0. GPU inference remains off pending memory availability. The prior turn made progress by implementing/validating MTP gates and repairing the reference linker; this turn independently revalidated those states.

Prepared a separate TensorRT INT4 group128/BF16 compatibility target, not general serving support. Logical signed weights span all16 codes; offset-binary checkpoint bytes are converted by XOR0x88. Independent CPU logical values and distinct row/group BF16 scales supply exact arithmetic oracles for batch1 and16. Bounded128x256 matrices and64MiB builder workspace are used; detailed engine inspector JSON is required. A numeric pass alone does not prove a fused WoQ tactic or full checkpoint compatibility. Build, GPU arithmetic and tactic inspection remain pending at this source commit. No synthetic timing is reported.


## P053 - 2026-10-23: first INT4 graph rejected; physical-layout diagnostic

Source c733ec39c verified6433 files remotely. The standalone test compiled/linked, then TensorRT10.16.1 engine build rejected the first batch1 [N,K] INT4 DQ/BF16 MatMul graph: Myelin reports blockwise dequantization input types exclude INT4, followed by no implementation found. Exit1 retained in int4-group128-ledger; no engine, arithmetic pass, WoQ support or metric claimed. Official pinned-version documentation describes INT4 G64/128 BF16 WoQ but does not establish this concrete graph/tactic combination: https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html .

Added optional physical [K,N] storage with transposed scales and block axis0, MatMul non-transposed. The independent logical CPU oracle remains unchanged. This tests a possible tactic-layout restriction; it is an unproven hypothesis. The original [N,K] failure mode remains selectable and preserved. No package upgrade, dequantized BF16 fallback or checkpoint change.


## P054 - 2026-10-23: transposed INT4 failure and model-sized probe

The physical [K,N] graph also compiled but failed engine building with the same Myelin INT4 block-dequantization type restriction; exit1 retained in int4-group128-kn-ledger. Neither physical layout produces a passing engine or numerical result. Added kn-model diagnostic for a representative actual27B o_proj geometry [N,K]=[5120,6144], retaining BF16 scales/G128 and unchanged logical signed oracle. This distinguishes small-matrix tactic constraints from a broader installed-route limitation. Still only a compatibility probe, with no full-checkpoint loading, serving or timing claim.


## P055 - 2026-10-23: model-sized INT4 failure archived and verified

The representative [N,K]=[5120,6144] graph also failed TensorRT10.16.1 engine building with the same INT4 block-dequantization restriction. All three diagnostics compiled; none produced an engine or arithmetic pass. They preserve signed4-bit storage, G128 BF16 scales and BF16 activations; no BF16 weight-materialization fallback was substituted. This leaves the direct TensorRT route unproven on this installed Windows/SM120 stack. Alternative native WoQ implementation or a verified different tactic remains required; this evidence does not establish universal INT4 incompatibility.

int4-compatibility-evidence.zip contains all three append-only ledgers, build launchers, final selectable source, CMake, source transfer/state manifests and executable SHA256. Download verified SHA256/ZIP CRC and every archived source/log digest:21320bytes/14entries, SHA256c17db38730b3bed655416355e686541d59a836f84d418305bb8169844f9a75a6. No benchmark generated. Remote final tested source87254695e,6433 files verified. The prior continuation made concrete progress: remote61-test validation, three actual INT4 compatibility failures, and evidence archive. 27B/MTP/vLLM full-model runs remain pending memory availability; target2x, INT4 inference,4B comparison, corrected image cohorts and native concurrency are still incomplete. Goal active.


## P056 - 2026-10-23: native direct packed INT4 fallback primitive

Current remote GPU remains10023MiB used/22165MiB free/0percent utilization; no full27B benchmark was started. Previous continuation made progress via remote61-test validation and three concrete TensorRT INT4 engine-build failures, not a performance result.

Added a native CUDA research primitive that reads compressed-tensors offset-binary low-nibble-first INT4 weights directly with BF16 G128 scales and BF16 activations, accumulates FP32, and emits BF16. One warp handles each output row; the kernel uses no dequantized weight matrix or workspace. This is a straightforward correctness-oriented W4A16 implementation, not an optimized/tensor-core performance claim. It is not yet registered in decoder lowering or admitted for full-model serving. Test covers all16 signed codes with independent logical weights, distinct group/row scales, batch1/16 and K256/6144, plus one direct launch and two actual CUDA-graph replays per profile. Exact oracle uses independent double accumulation on dyadic fixtures. Compilation and GPU arithmetic are pending at this commit; no benchmark or real-weight equivalence is asserted.


## P057 - 2026-10-23: native INT4 fixture/graph arithmetic passes; actual slice prepared

Source a10a31864 verified6436 files remotely, then the standalone CUDA primitive compiled and exited0. All four profiles M1/16 x K256/6144 with N128 pass exact independent logical-weight/double-sum oracles for one direct launch and two actual CUDA-graph replays each. No workspace/full BF16 weight matrix is allocated. This is component arithmetic/capture evidence, not full-model support or a performance measurement. Raw build warnings/output and source identities remain in ct-int4-native-ledger.

Prepared an actual pinned checkpoint-slice gate: first128 output rows of layer11 self_attn.o_proj [5120,6144], BF16 G128 scales, M16 dyadic BF16 inputs. Reads bounded safetensors spans via explicit index (not DirectoryOpen preference), verifies dtype/shape/logical shape, saves packed/scales/input/expected bytes with digests. Independent Python double accumulation is rounded to BF16. Before GPU run the criterion is fixed to each finite output error <=max(1e-6,abs(expected)*0.0078125). Direct and two graph outputs are retained. The actual slice gate remains pending at this commit; it cannot establish full-layer, full-model quality or timing.


## P058 - 2026-10-23: actual native INT4 slice passes and evidence archived

Source ed464063d verified6437 files remotely. Native CUDA actual-weight test compiled and completed exit0: layer11 self_attn.o_proj first128/5120 rows, K6144, M16,2048 outputs each for one direct launch and two CUDA-graph replays. Every output exactly equals the independently generated Python double-sum/BF16 oracle: maximum absolute error0,relative L2 error0 on all three launches. This exceeds the predeclared bounded criterion without changing it. Fixture source index/packed/scales/input/oracle hashes are retained. This is only a bounded projection proof; decoder lowering, full checkpoint loading, end-to-end quality, true W4A16 full-model VRAM/speed and2x remain incomplete.

Verified archive ct-int4-native-evidence.zip includes both append-only ledgers, original packed/scale/input/expected slices, all actual output arrays, final source/CMake, launchers, source-transfer manifests and binary digest. Local SHA256, ZIP CRC and every archived file digest verified: 385488bytes/22entries, SHA25667dd385072e27abbf5b822184cfc711aa3c7c50ac6520a7fee853c3539c198bf. No synthetic or actual-slice timing is presented as model performance. The current continuation made concrete implementation/GPU arithmetic progress. Full27B runs still await GPU memory availability; goal active.


## P059 - 2026-10-23: native packed INT4 plugin candidate

Current remote GPU still10023MiB used/22165MiB free; full27B/MTP/reference timing remains deferred. Prior continuation made concrete progress with native CUDA packed INT4 arithmetic and an exact actual-checkpoint slice result, preserved in verified evidence. Goal is not complete.

Added a dedicated serialized TensorRT projection plugin around the tested CUDA primitive. It accepts BF16 activations, opaque INT8 bytes carrying offset-binary packed INT4 weights and BF16 G128 scales, produces BF16 with FP32 accumulation, and requires zero plugin workspace. Constructor/deserializer/shape bounds reject unsupported geometry; no activation quantization or full BF16 weight materialization. The standalone graph test retains the independent logical oracle, builds/serializes/deserializes an engine, and saves its detailed inspector. Decoder lowering/admission has not yet changed. Build and plugin arithmetic are pending at this commit; standalone kernel success is not substituted for plugin validation.


## P060 - 2026-10-23: serialized INT4 plugin passes; decoder lowering candidate

Source051107cc8 verified6440 files remotely. Native INT4 plugin test built/serialized/deserialized engines and matched independent exact logical-weight outputs at M1/16,128/2048 outputs respectively, exit0. Inspector JSON retained. Plugin constants retain compressed bytes and BF16 group scales; no generic TensorRT INT4 DQ route or BF16 expanded weight matrix is used. This does not establish decoder integration or speed.

Added candidate compiled-graph recipe recognition for the specific symmetric compressed-tensors G128 W4A16 format; explicit shard index is used for this recipe. Lowering validates I32 packed shape/bytes, I64 logical shape, positive finite BF16 scales, and rejects zero-point/g_idx metadata. Missing logical .weight names map to their packed triplets only in this precision profile. MatMul dispatch reaches the dedicated plugin; immutable prequantized engines are retained without stripping/refitting. Per-graph INT4 counts expose actual path use. Added research-only batch1/capacity32..4096 decoder factory requiring all496 packed projections and BF16 head, with explicit index preserving sidecars. General ServeModel admission remains closed. Native main-library compilation and decoder/layer correctness remain pending. New CPU recipe tests reject changed weight/activation arithmetic and unsupported profiles.


## P061 - 2026-10-23: independent INT4 layer oracles prepared during live build

The main-library build/captured-projection handle is confirmed live and has reached final link after all17-source build steps began; no resync/restart occurred. Local-only preparation adds a bounded-memory NumPy packed-INT4 projection oracle to the existing independent layers0/3 arithmetic, two stateful serial tokens each at capacity4096. CPU dequantization is only an oracle, in128-row blocks; native GPU serving retains compressed weights. The original hidden/state thresholds are unchanged. Probe accepts an explicit index path and verifies actual CT profile/count8 or7, no NVFP4/FP8/external BF16 paths, and a non-refittable retained engine before checking arithmetic. Full states are not exhaustively compared; scope remains all hidden values plus sampled state entries/active-KV prefix samples. Oracle generation and layer GPU checks are pending; full27B, chunked layer quality, benchmarks and2x remain unproven.


## P062 - 2026-10-23: main Garnet INT4 capture/lowering passes

Main source731cfcea4 verified6443 files; complete Garnet library build and actual projection capture/lowering job exited0. The full5120-row layer11 o_proj is compiled through the explicit-index loader/native plugin; CPU oracle covers its first128 rows (2048 outputs for M16). Five direct/graph-enabled runs meet the unchanged criterion and all full-projection outputs are byte-identical across execution modes. Stats require actual native_int4_g128_projection_weights1, zero NVFP4/FP8/external BF16 paths, non-refittable engine and successful graph replay. Detailed actual errors/stats/raw outputs are retained. This remains one projection, not decoder/full-model correctness or performance.

Runtime Python3.14 emitted missing optional hashlib algorithm warnings, retained in raw logs; required SHA256 and the probe completed successfully. No global Python/SDK repair was made. Process audit distinguished live post-build SDK comparisons from a failed link: read traffic increased2.799GB to4.076GB over52seconds, with zero writes, before final build completion. This setup I/O duration is not TTFT. Independent CPU INT4 layer oracle generation also completed exit0 on its separate retained handle, using the artifact-hashed helper from23277bd3f against unchanged reference code. Local layer probe now saves every full hidden array and sampled state outputs before assertions, preserving failures as well as passes. Stateful layer GPU gates remain next; no full27B benchmark or2x claim.


## P063 - 2026-10-23: stateful INT4 layer gates pass and integration evidence verified

Remote source5e4023565 verified6444 files. The unchanged native C++ source/binary from731cfcea4 is reused (Git source diff empty, binary SHA256 compared with preserved capture provenance). Stateful layers0/3 GPU job exited0: two serial tokens per layer, capacity4096, five direct/graph-enabled hidden checks each, plus sampled recurrent/conv/KV state checks. Actual projection counts8/7 and CT profile are asserted; NVFP4/FP8/external BF16 fallbacks are zero. All63 CPU study tests also pass remotely. Raw hidden and sampled-state arrays are saved before assertions; full-state/full-model and chunk16 numerical coverage remain incomplete.

Machine-derived layer error summary: [{"layer": 0, "passed": true, "native_int4_weights": 8, "hidden_max_abs": 0.00390625, "hidden_max_rel_l2": 9.846876721895467e-05, "state_max_abs": 4.76837158203125e-07, "state_max_rel_l2": 1.5137746306454617e-07}, {"layer": 3, "passed": true, "native_int4_weights": 7, "hidden_max_abs": 0.00048828125, "hidden_max_rel_l2": 5.905684291976985e-06, "state_max_abs": 4.76837158203125e-07, "state_max_rel_l2": 4.134340554310304e-08}]

ct-int4-integration-evidence.zip verified locally by archive SHA256, ZIP CRC and every archived source/log/raw-result digest: 2597964bytes/93entries, SHA256575d53d904b4c9ab36e0a3ef1df7e7c4454b400ebcf43155e63fadc9514331b6. Includes plugin build/load evidence, actual captured projection, source snapshots/provenance, CPU oracle, actual stateful layer outputs, all command ledgers and binary hash. Captured projection's checked2048 values have max_abs0 for all five runs; complete5120-row outputs are byte-identical through replay. No model benchmark or2x claim. This continuation made concrete decoder mapping/compilation/selected-layer progress. Full INT4/27B VLM/MTP and4B comparisons, corrected reference/native image cohorts, concurrency and2x remain open; GPU memory is still held by desktop apps.

Optimization candidate found by source inspection only: current qwen35_attention_core.cu schedules scores over full capacity and all padded query rows, then normalizes over full capacity even for a short active prefix. Compact per-query score/softmax/context scheduling may reduce wasted blocks; it requires an independently gated A/B experiment and real timing. No speedup inferred from this inspection alone.


## P064 - 2026-10-23: compact attention A/B candidate and predeclared gates

Prior continuation made concrete INT4 capture/selected-layer progress; full-model comparisons remain incomplete. Current GPU9903MiB used/22285MiB free/0percent utilization, so no27B model or reference was started.

Added a separate unwired compact FP32 attention function. It combines visible-prefix score/softmax/context in one query block, preserves FP32 dot/context reduction ordering, and retains full KV copy/update semantics. The original function/plugin remain selected by serving. Shared declarations total33828 logical bytes/block (compiler padding/residency unmeasured), no global scratch; one block/query can underutilize the GPU for long prefixes, so no speedup is assumed. Standalone comparison covers actual27B H24/KV4/D256, padded T16 decode prefixes32/128/512/4096, full active T16 prefix32, ragged/inactive B2 with non-multiple capacity33/D64, and full8192 prefix. Independent CPU double attention, full KV bit equality, untouched inputs, buffer guards, exact legacy/compact contexts and three graph replays must all pass before timing.

Correctness and measurement are separate invocations. Subsequent component-only CUDA-event intervals enclose20 graph launches/sample, six samples/variant in ABBA order repeated3 times after five warmups/variant, retaining raw samples/contexts/source hashes, device XML and VRAM samples. Context/KV copies are included; CPU oracle and downloads are outside event intervals. Warm component caches and co-resident desktop GPU memory make this exploratory and distinct from full-model HTTP TTFT/decode/concurrency. Test/build/metrics are pending at this commit; no2x claim or serving default change.


## P065 - 2026-10-23: compact attention passes but long-prefix regression measured

Source0140b1142 verified6446 files remotely. Standalone SM120 build and gated component study completed exit0. All seven profiles passed independent CPU attention, bit-exact legacy/compact contexts, entire KV outputs, untouched inputs, guards, invalid-profile rejection and three graph replays. Largest CPU maximum absolute error1.81607902e-8. No serving plugin/factory was switched.

Machine-derived medians and component ratios, retaining every raw sample including the8192 baseline first-sample outlier: [{"case": 0, "capacity": 4096, "prefix": 32, "all_active": false, "baseline_ms_median": 0.117227204, "compact_ms_median": 0.02282399965, "component_ratio_baseline_over_compact": 5.136137653244313, "cpu_max_abs": 6.51925802e-09}, {"case": 1, "capacity": 4096, "prefix": 128, "all_active": false, "baseline_ms_median": 0.1180775985, "compact_ms_median": 0.03146640025, "component_ratio_baseline_over_compact": 3.7524978250411727, "cpu_max_abs": 2.56113708e-09}, {"case": 2, "capacity": 4096, "prefix": 512, "all_active": false, "baseline_ms_median": 0.1211975995, "compact_ms_median": 0.06582640114999999, "component_ratio_baseline_over_compact": 1.841170068280423, "cpu_max_abs": 1.86264515e-09}, {"case": 3, "capacity": 4096, "prefix": 4096, "all_active": false, "baseline_ms_median": 0.1537336035, "compact_ms_median": 0.382966399, "component_ratio_baseline_over_compact": 0.4014284383732579, "cpu_max_abs": 6.05359674e-09}, {"case": 4, "capacity": 4096, "prefix": 32, "all_active": true, "baseline_ms_median": 0.1407656, "compact_ms_median": 0.03713999875, "component_ratio_baseline_over_compact": 3.7901347533028655, "cpu_max_abs": 7.4505806e-09}, {"case": 5, "capacity": 33, "prefix": 32, "all_active": false, "baseline_ms_median": 0.015332000299999999, "compact_ms_median": 0.014500800500000001, "component_ratio_baseline_over_compact": 1.0573209596256425, "cpu_max_abs": 3.7252903e-09}, {"case": 6, "capacity": 8192, "prefix": 8192, "all_active": false, "baseline_ms_median": 0.335959986, "compact_ms_median": 1.0311143999999999, "component_ratio_baseline_over_compact": 0.3258222230239438, "cpu_max_abs": 1.81607902e-08}]

Short prefixes benefit, but full4096/8192 prefixes regress. Therefore this candidate is not a general default and does not establish any model speedup. Timings are warm core CUDA graphs including KV copies; CPU work/downloads excluded, six samples/variant,20 launches/sample, ABBAx3, desktop memory co-resident. Full HTTP metrics and2x goal remain unmet.

Archive verified locally SHA256/CRC/all file digests: attention-compact-evidence.zip,1267878bytes/60entries,SHA25693e6a686997a67e8a628c6b24f71b7e1a5ba94f9319eca66a31692b64628c9b9. Raw contexts, all timing samples, CPU/graph gates, build ledgers, CMake cache, device XML, VRAM samples, source and binary hashes are retained. Added a separate output-tiled candidate to test whether duplicating scores across32-dimension tiles increases useful GPU parallelism for long prefixes; no performance assumed. New three-way experiment predeclares ABCCBAx3 (legacy,row,tiled), same six samples/variant and unchanged numerical gates. Current serving path remains legacy; new tiled build/run pending.


## P066 - 2026-10-23: output tiling does not fix long-prefix cost

Source39a393c8a verified6447 files; three-variant build and independent-gated ABCCBA study completed exit0. All seven profiles retain bit-exact legacy/row/tiled contexts, full KV states, CPU tolerances, three graph replays, guards and untouched inputs. No serving path changed. Machine-derived complete component medians: [{"case": 0, "capacity": 4096, "prefix": 32, "all_active": false, "baseline_ms": 0.1171040015, "row_ms": 0.02279360035, "tiled_ms": 0.02156960035, "baseline_over_tiled": 5.429122450106036}, {"case": 1, "capacity": 4096, "prefix": 128, "all_active": false, "baseline_ms": 0.117706399, "row_ms": 0.0311136004, "tiled_ms": 0.02895120065, "baseline_over_tiled": 4.065682816508683}, {"case": 2, "capacity": 4096, "prefix": 512, "all_active": false, "baseline_ms": 0.121114403, "row_ms": 0.06579440085, "tiled_ms": 0.062076801449999996, "baseline_over_tiled": 1.9510412935426782}, {"case": 3, "capacity": 4096, "prefix": 4096, "all_active": false, "baseline_ms": 0.1535487995, "row_ms": 0.382740796, "tiled_ms": 0.371903196, "baseline_over_tiled": 0.41287303027102784}, {"case": 4, "capacity": 4096, "prefix": 32, "all_active": true, "baseline_ms": 0.13956160099999998, "row_ms": 0.0330456011, "tiled_ms": 0.06735279785, "baseline_over_tiled": 2.0720980487078604}, {"case": 5, "capacity": 33, "prefix": 32, "all_active": false, "baseline_ms": 0.012026399850000001, "row_ms": 0.0116408002, "tiled_ms": 0.0106392, "baseline_over_tiled": 1.130385729190165}, {"case": 6, "capacity": 8192, "prefix": 8192, "all_active": false, "baseline_ms": 0.335593611, "row_ms": 1.029478375, "tiled_ms": 1.103306355, "baseline_over_tiled": 0.3041708311378303}]

The output-tiled variant is still slower than legacy at full4096/8192, and slower than row-compact with all16 short-prefix queries active. No cause is asserted from timings alone, and no general default or model speedup follows. Archive attention-tiled-evidence.zip verified SHA256/CRC/all file digests: 1676775bytes/74entries,SHA256eb889e12355d4e561717e07d52f7655ee06b3705e8f5ec88c695bed9a72797ce. Negative results and every raw sample/context remain retained.

Prepared a separate split-key candidate:64-key blocks produce FP32 local maxima, denominator and weighted-context partials, followed by stable global merging. It retains full KV copy/update and masks, reduces fixed-capacity score block count, and uses explicit bounded global scratch. Reduction order differs; predeclared CPU/legacy tolerance remains2e-5+2e-4*abs(reference), not bit equality. Row/tiled paths must still be bit-exact; all states/graph replays/guards remain strict. Four-way ABCDDCBAx3 gives six samples/variant after qualification, with no omitted sample. Build and measurements pending; no2x/model performance claim.


## P067 - 2026-10-23: split-key component improves4K profiles; full-model claim remains open

Sourceac1be2ba6 verified6448 files remotely. Standalone build and gated four-way study completed exit0. All seven independent CPU/context/full-KV/guard/input/invalid-profile/graph gates passed for four variants. Row/tiled remain bit-identical to legacy. Split changes FP32 reduction order and meets the unchanged2e-5+2e-4*abs(reference) tolerance; all KV values remain exact. Raw context replay independently verifies these comparisons and recorded maximum legacy/split differences. Every sample retained; no outlier was removed.

Machine-derived component medians/ratios and raw replay errors: [{"case": 0, "capacity": 4096, "prefix": 32, "all_active": false, "baseline_ms_median": 0.117073603, "split_ms_median": 0.03101280055, "component_ratio_baseline_over_split": 3.775009058316083, "split_max_abs_vs_baseline_replayed": 6.51925802230835e-09, "split_workspace_bytes": 25362432}, {"case": 1, "capacity": 4096, "prefix": 128, "all_active": false, "baseline_ms_median": 0.11760479600000001, "split_ms_median": 0.0320864003, "component_ratio_baseline_over_split": 3.6652536557676747, "split_max_abs_vs_baseline_replayed": 3.6088749766349792e-09, "split_workspace_bytes": 25362432}, {"case": 2, "capacity": 4096, "prefix": 512, "all_active": false, "baseline_ms_median": 0.1214168035, "split_ms_median": 0.033210398599999996, "component_ratio_baseline_over_split": 3.6559875406012146, "split_max_abs_vs_baseline_replayed": 2.7939677238464355e-09, "split_workspace_bytes": 25362432}, {"case": 3, "capacity": 4096, "prefix": 4096, "all_active": false, "baseline_ms_median": 0.153206393, "split_ms_median": 0.0659863986, "component_ratio_baseline_over_split": 2.321787463030298, "split_max_abs_vs_baseline_replayed": 5.587935447692871e-09, "split_workspace_bytes": 25362432}, {"case": 4, "capacity": 4096, "prefix": 32, "all_active": true, "baseline_ms_median": 0.1395248025, "split_ms_median": 0.0453040004, "component_ratio_baseline_over_split": 3.079745745808354, "split_max_abs_vs_baseline_replayed": 8.847564458847046e-09, "split_workspace_bytes": 25362432}, {"case": 5, "capacity": 33, "prefix": 32, "all_active": false, "baseline_ms_median": 0.01151680015, "split_ms_median": 0.0131176002, "component_ratio_baseline_over_split": 0.8779654795394664, "split_max_abs_vs_baseline_replayed": 3.259629011154175e-09, "split_workspace_bytes": 50688}, {"case": 6, "capacity": 8192, "prefix": 8192, "all_active": false, "baseline_ms_median": 0.33610400549999997, "split_ms_median": 0.166548796, "component_ratio_baseline_over_split": 2.0180512472753027, "split_max_abs_vs_baseline_replayed": 1.7695128917694092e-08, "split_workspace_bytes": 50724864}]

For tested H24/KV4/D256,B1,T16,capacity4096 profiles, split improves the warm attention core including KV copies at every tested prefix. It regresses the small capacity33/B2/D64 profile, so it is not a universal default. Split workspace is25,362,432bytes at B1/T16/4K versus legacy6,291,456bytes; model memory admission must include this tradeoff. Component clocks/caches and desktop co-residency do not establish HTTP or model performance, and no2x-vs-SGLang/vLLM result is claimed. Existing serving plugin remains legacy; model-level integration/greedy/quality/precision gates remain required.

attention-split-evidence.zip verified SHA256/CRC/every file digest and raw output/error replay: 2084195bytes/88entries,SHA256fe53ab6b822bec6a3873c232ca9c210b0ff6a2a3f0629db30a7b41b06a6d577a. Includes every raw timing/context, qualification/build ledgers, CMake/device/VRAM metadata and source/binary hashes. Earlier compact/tiled failures remain separately archived. This continuation made measured implementation progress, not goal completion. The main Garnet DLL predates these unwired attention candidates; do not attribute them to the earlier native HTTP figures.

Next: qualify an opt-in attention plugin/model profile before any model speed claim, and continue the4B comparison through current Garnet factories. Full27B/MTP/vLLM/INT4 runs remain pending GPU memory release;2x, actual native concurrency and fully matched corrected-image/precision/quality comparisons remain incomplete. Goal active.


## P068 - 2026-10-23: full4B smoke prepared through current Garnet Model API

Previous continuation made measured attention-component progress; it did not finish model integration or establish2x. GPU memory remains about9.9GB used/22.3GB free, so full27B/reference/MTP runs are still deferred. The smaller downloaded4B checkpoint can now be qualified through existing runtime.cuda_serving.Model and vl_4b factories, avoiding a second inference implementation. Prepared B1/capacity4096,prefill/decode16,fused GDN/attention and BF16 vision16x16 profile. Tests record actual arithmetic/color outputs, repeated text/image IDs, all graph/memory statistics and a3968-token input with128 allowed outputs. Early EOS/actual lengths are retained; configured capacity is not a claim of4096 occupied KV slots. Source, checkpoint config/image/binary hashes and preserved binary origin are recorded. Internal API timings remain raw exploratory data, not HTTP benchmarks or matched comparison. Independent4B quality/reference and broader accuracy remain pending even if this smoke passes.


## P069 - 2026-10-23: user-authorized Cantor shutdown; full4B qualification live

Previous goal turn made progress by stopping the explicitly authorized Cantor application and verified WorldSense processes, while preserving the study xlang3 process. Remote process identity evidence is retained in results/cantor-authorized-stop.json and copied locally. Immediately afterward NVML reported4907MiB used/27281MiB free, including the running4B qualification, not an idle memory baseline. The same4B job handle remains live with increasing CPU/file activity; do not restart for quiet output. Source synchronization and other GPU engines remain deferred until it is terminal. No4B correctness or benchmark result yet.

Prepared optional vLLM log/output/metadata paths for a distinct linker-repaired retry, retaining the original failed startup evidence and CUDA object cache. The driver refuses any nonempty output directory before health checks or mutations. Python compilation, CLI preservation check and diff whitespace checks passed. This preparation does not establish vLLM startup, quality or performance; full-model2x remains unmet.


## P070 - 2026-10-23: real4B token event path prepared, GPU qualification pending

The previous continuation made progress with explicit Cantor shutdown evidence and committed/pushed attempt-preserving vLLM retry paths. The same full4B smoke remains live and advances CPU/file/memory allocation, so no restart or concurrent GPU engine is justified. Remote source remains frozen at its run identity.

Added optional on_token(row,token,sampled_count,prompt_count) to current runtime.cuda_serving.Model target-only inference. Events are emitted from synchronized decoder predictions before the next decode, include actual EOS, and skip stopped batch rows. Defaults preserve existing calls/results. Noncallable callbacks and speculative callback use fail explicitly; the separate27B verified-MTP worker is unchanged. Consumer errors propagate rather than fabricate completion. Added research4B worker using the current Model, exact final-vs-streamed ID/length checks, BF16 weights/FP32 residuals, B1/capacity4096 and real text/JPEG inputs. The existing HTTP bridge accepts an explicit pinned4B model route while preserving its27B default. It is still one queued slot, not native concurrent batch serving.

All68 CPU study tests passed, including exact production-method event control with staggered row EOS, length boundary, consumer failure, existing real-pipe HTTP token accounting and explicit checkpoint-route rejection. One initial direct module test invocation omitted the test import path; rerunning via the repository test discovery path passed. These CPU fixtures establish control/protocol behavior only, not GPU numerical quality, HTTP timing or throughput. The new worker has not been loaded onGPU; independent4B reference/quality and matched benchmarks remain pending. No2x claim.


## P071 - 2026-10-23:4B native load/text pass; image policy failure diagnosed and preserved

Initial full4B job8d3131d9904d4123938703a7c93df530 finished exit1 after compiling32 decoder layers and vision. Text arithmetic returned4 with2 sampled tokens including EOS. The next image call failed before vision inference because default native prepare_image uses min_pixels1024/max65536, giving14x14 for the224px fixture, while the static profile preloaded16x16. No full4B quality/long-input/performance result exists. Whole-build adapter memory sampled maximum15542145024bytes includes Cantor's initial co-residency, transient compilation and all background allocations; it is not warm serving peak. All raw failed logs/report/source snapshots are retained in qwen35-4b-initial-failure-evidence.zip.

Actual native image-only probe completed exit0: default14x14, explicit min=max65536 gives16x16. Added opt-in imageMinPixels to existing Model using existing prepare_image_profile; absent option retains previous resize behavior.4B study profiles explicitly choose65536 to match the intended64 visual tokens. Probe supports a distinct output directory so retries preserve failures. All68 CPU study tests and whitespace checks pass; full corrected4B GPU inference remains pending.

User raised concern about another person's restore/change. Read-only remote integrity audit verified6449 retained source hashes with zero mismatches and unchanged qualified Garnet DLL. Remote source was intentionally frozen atc4b981638 during the job; latest local62eacd429 and current image fix were not silently deployed. Original local D:/CantorAI/Garnet remains feature/gpt-oss-120b with existing dirty/untracked edits, untouched. No attribution of authorship or modification time is inferred. Cantor desktop app has since restarted (observedPID5584), but GPU was464MiB used after the study exited. Dedicated branch/root remain authoritative for our runs.


## P072 - 2026-10-23: authorized primary-source replacement; repaired vLLM graph retry live

User explicitly requested D:/CantorAI/Garnet use our code. Preserved all original dirty/untracked files via stash commit plus stable refs/codex-backups/garnet-before-qwen38-5090-20261023, original branch/head, binary patch and per-file hashes in D:/CantorAI/backups/garnet-before-qwen38-5090/replacement-record.json. Then switched primary checkout to detached1e19927bc; all9992 tracked files matched dedicated qwen38-5090 worktree byte-for-byte. Ignored build/env assets retained; no claim those primary binaries were rebuilt. Local user-facing Garnet-source-replacement-record.json preserves recovery identity. No third-party authorship inferred.

Remote1e19927bc source transfer verified6451 files/zero mismatches. Initial4B failure ZIP132354bytes/13files was downloaded and independently verified by ZIP CRC and allmanifest digests. Native image-only probe confirms14x14 default versus16x16 explicit resize policy. After terminal4B job and freeGPU verification, started separate vLLM linker-repaired server and gated HTTP driver, keeping original failure logs untouched. New results/vllm-graph-linkfix-* paths retain fresh metadata, raw command ledgers and frozen source snapshot. Server/read-only WSL probe are live/healthy atprocess level; model loaded, AOTcache reused, FlashInfer GEMM/SM120 FP4 JIT compilers observed consuming CPU. HTTP health, graph capture, quality, benchmark completion and admission remain unproven. Do not restart or overlap another model merely due quiet JIT output.

Next: finish or diagnose this actual reference run; then corrected4B GPU streaming/correctness, strict27B MTP/fullINT4 and corrected native/SGLang matched cohorts.2x remains unmet; goal active.


## P073 - 2026-10-23: corrected4B retry now checks actual GPU callback sequences

Previous continuation made source/plan progress and verified the same reference processes live. Current vLLM server/gated-driver handles remain running; Linux compiler processes and newly completed .ninja_log FP4 objects prove active JIT work, not HTTP readiness or benchmark completion. No restart, cache deletion or concurrent GPU model was performed. Reference source remains frozen at1e19927bc with separate source-snapshot.zip; local documentation/source preparation does not change that running source.

Prepared corrected4B smoke to enable vision CUDA graph mode, collect actual target-only Model callbacks on text/image, preserve raw events/results before assertions (including partial events on inference exceptions), compare callbacks with complete sampled IDs/EOS/prompt counts, and compare repeated calls with callbacks disabled. Existing repeat/near-capacity/independent-quality limitations remain. The4B research HTTP worker enables the same vision graph setting. CPU study suite68 tests passed after the callback change, both scripts compile, and whitespace checks pass. These additions are not yet run onGPU and produce no performance claim.


## P074 - 2026-10-23: vLLM corrected-grid graph pilot completed and independently replayed

Reference source1e19927bc, run cf140177a4f24bf3ad845ad32f8cb4c0 completed exit0, after native4-JPEG token/color gates passed. Actual FULL decode graph capture sizes1/2/4/8 completed. Resolved cache37774 tokens/log-estimated9.22x capacity at4096; this is not an actual eight-full4K-input admission test. BF16 KV/FP32 SSM, mixed checkpoint, prefixcache disabled. Actual workload short41-token text/104-token image, max128, seed3 JPEG/image-first/binary-grid,8 measured requests plus2 warmup waves/cohort. Every measured cohort generated1024 actual tokens, all measured arrivals single-token.

[
  {
    "case": "text",
    "concurrency": 1,
    "ttft_median_ms": 152.36114999970596,
    "decode_median_tokens_per_second": 68.48187553704147,
    "output_tokens_per_second": 63.62260897505279,
    "generated_tokens": 1024,
    "sampled_peak_adapter_used_bytes": 27909734400
  },
  {
    "case": "text",
    "concurrency": 2,
    "ttft_median_ms": 257.50035000055504,
    "decode_median_tokens_per_second": 63.639642380250216,
    "output_tokens_per_second": 113.24198465156229,
    "generated_tokens": 1024,
    "sampled_peak_adapter_used_bytes": 27909734400
  },
  {
    "case": "text",
    "concurrency": 4,
    "ttft_median_ms": 272.6720500013471,
    "decode_median_tokens_per_second": 61.885123191550846,
    "output_tokens_per_second": 219.89214333320064,
    "generated_tokens": 1024,
    "sampled_peak_adapter_used_bytes": 27909734400
  },
  {
    "case": "text",
    "concurrency": 8,
    "ttft_median_ms": 296.779250000327,
    "decode_median_tokens_per_second": 63.7942554305161,
    "output_tokens_per_second": 445.7684580249254,
    "generated_tokens": 1024,
    "sampled_peak_adapter_used_bytes": 27909734400
  },
  {
    "case": "image",
    "concurrency": 1,
    "ttft_median_ms": 166.51065000041854,
    "decode_median_tokens_per_second": 68.49756713461639,
    "output_tokens_per_second": 63.21622392036499,
    "generated_tokens": 1024,
    "sampled_peak_adapter_used_bytes": 27916025856
  },
  {
    "case": "image",
    "concurrency": 2,
    "ttft_median_ms": 331.79550000022573,
    "decode_median_tokens_per_second": 64.02234111212088,
    "output_tokens_per_second": 110.64390951956912,
    "generated_tokens": 1024,
    "sampled_peak_adapter_used_bytes": 27928608768
  },
  {
    "case": "image",
    "concurrency": 4,
    "ttft_median_ms": 312.0722500025295,
    "decode_median_tokens_per_second": 61.236407185715734,
    "output_tokens_per_second": 214.92384591196142,
    "generated_tokens": 1024,
    "sampled_peak_adapter_used_bytes": 27932803072
  },
  {
    "case": "image",
    "concurrency": 8,
    "ttft_median_ms": 327.7791000000434,
    "decode_median_tokens_per_second": 63.69880718536221,
    "output_tokens_per_second": 439.86431903929963,
    "generated_tokens": 1024,
    "sampled_peak_adapter_used_bytes": 27993620480
  }
]

Archive vllm-graph-linkfix-evidence.zip: 1187004bytes, SHA256 8002dc620d88f1234e6a34548fa1f0d64ad71d0c13c8255e74fce9fcd98ca930. Independently verified ZIP CRC/allfile hashes and replayed64 measured request bytes/SSE timestamps/token counts/text/summaries. All62 image warmup/measured JPEG requests independently decoded and encoded/decoded hashes verified:62distinct encoded and62distinct RGB, fixing the legacy pixel-nonce collision. These checks do not prove broad pretrained quality or reference/native precision equivalence. Output-head FP4 tuning buckets for4/8 rows remain uncovered (fallback tactic=-1), so this is a candidate, not an optimized-fastest-reference claim. ServerPID301 was identity-checked/signaled; supervised server exit0 and GPU464MiB used/31724MiB free confirmed before another engine. Original failed startup logs remain intact.

Prepared chart from archived manifests comparing historical text-only Garnet/SGLang with this vLLM candidate. Chart explicitly labels different run/request-preparation profiles and native C2/4/8 unmeasured. No2x claim; current Garnet historical decode is slower. Full matched corrected native/SGLang runs, arithmetic/quality/repetition and concurrent admission still required.

Synced corrected4B sourcec8d0e21dc, verified6451 files/zero mismatches, started qwen35-4b-grid16-native-smoke in separate fixed results/ledger paths after vLLM shutdown. It remains live with growing CPU/file activity; no premature success claim. Prepared explicit MTP3 packed native launcher using the qualified DLL without forced rebuild, and validator --launcher provenance plus nonempty-output preservation guard.68 CPU study tests and CLI preservation check pass; MTP GPU self-greedy/quality/stream/benchmark remain pending. No concurrent GPU models or source mutation during the4B job.


## P075 - 2026-10-23: corrected full4B GPU smoke passes; actual27B MTP3 load started

Corrected4B run c2e3b157aa34452a8ea5ec13a5837c52, sourcec8d0e21dc, completed exit0. Native32-layer decoder plus BF16 vision, B1/capacity4096, min=max65536/profile16x16, text4 and dominant-red image both correct, repeats exact. First text/image calls used actual Model callbacks and matched full sampled IDs includingEOS/prompt counts; repeated calls without callbacks matched generation.3968 actual input tokens plus128 allowed outputs generated78 sampled tokens before natural stop; no padding to128. Decoder33graphs report33captures/11060replays/zero capture failures. Vision graph requested,1capture/0replays across two image requests; this does not establish warmed vision graph replay or benefit. Allocator tracking disabled, so no claim of exhaustive retained model memory. Whole build/correctness NVML peak11937824768bytes is not warm serving peak. Independent reference/broad quality/HTTP/concurrency timings still pending.

qwen35-4b-grid16-evidence.zip 88689bytes SHA256 beb80c6e2aa69997eac5a96b03fa53ca8cd58674b654cb51a49652cceeb03c1b. Independently verified ZIP CRC/allfile hashes and actual callback/final-ID/long-input fields after downloading. Initial failed image-grid run remains separately preserved.

After verified terminal4B and GPU464MiB used/31724MiB free, synchronized3e26eb8cc source with6452 verified files/zero mismatches. Started native-mtp3-http-server and native-mtp3-gated-http handles, using explicit packed/MTPdepth3 launcher and separately pinned existing qualified DLL. Native process24524 is live loading/building; no MTP readiness, numerical pass or timing result yet. Do not restart for quiet output, overlap a model, or change its running source. Installed-reference MTP source audit was invoked read-only; observation timeout does not establish that it failed or completed, so inspect its artifact/process before retry. Full final comparison must include feasible supported vLLM/SGLang MTP candidates as well as ordinary decode; generic MTP implementations do not prove this mixed-quantized VLM checkpoint is supported.2x remains unmet.


## P076 - 2026-10-23: actual MTP3 passes strict greedy gates,72tok/s candidate remains far from2x

Source3e26eb8cc, MTP3 runfdac816cfa4645ce845e3e2fafc4b3ed completed correctness+benchmark exit0. Four retained native VLM/prompt/semantic checks passed, reference exact3/4 remains the known bottom-blue whitespace difference. All16 complete128-token self-greedy text/image pairs matched exact prompt/generated IDs, actual verified_mtp mode and real metrics; real stream gates passed before timing. Measured16 requests generated2048 actual tokens, no fabricated arrivals. B1/capacity4096/short41text104image/max128/seed3/binary-grid image-first JPEG.

[
  {
    "requests": 8,
    "generated_tokens": 1024,
    "ttft_median_ms": 111.15090000021155,
    "decode_median_tokens_per_second": 72.11572324956046,
    "output_tokens_per_second": 67.2679320557427,
    "requests_per_second": 0.5255307191854899,
    "sampled_peak_adapter_used_bytes": 27616133120,
    "samples": 482,
    "requested_interval_seconds": 0.02,
    "observed_sample_gap_median_seconds": 0.03192420000050333,
    "observed_sample_gap_max_seconds": 0.0339968999978737,
    "scope": "NVML adapter-wide used memory, including background allocations; sampled maximum, not instantaneous or process-attributed peak",
    "case": "text",
    "concurrency": 1,
    "single_token_arrivals": true,
    "prompt_tokens": [
      41,
      41,
      41,
      41,
      41,
      41,
      41,
      41
    ],
    "request_keys": [
      "fdac816cfa4645ce845e3e2fafc4b3ed-s3-text-c1-r0",
      "fdac816cfa4645ce845e3e2fafc4b3ed-s3-text-c1-r1",
      "fdac816cfa4645ce845e3e2fafc4b3ed-s3-text-c1-r2",
      "fdac816cfa4645ce845e3e2fafc4b3ed-s3-text-c1-r3",
      "fdac816cfa4645ce845e3e2fafc4b3ed-s3-text-c1-r4",
      "fdac816cfa4645ce845e3e2fafc4b3ed-s3-text-c1-r5",
      "fdac816cfa4645ce845e3e2fafc4b3ed-s3-text-c1-r6",
      "fdac816cfa4645ce845e3e2fafc4b3ed-s3-text-c1-r7"
    ]
  },
  {
    "requests": 8,
    "generated_tokens": 1024,
    "ttft_median_ms": 269.9392500016984,
    "decode_median_tokens_per_second": 71.65596627478999,
    "output_tokens_per_second": 62.18879488124505,
    "requests_per_second": 0.48584996000972697,
    "sampled_peak_adapter_used_bytes": 27620327424,
    "samples": 521,
    "requested_interval_seconds": 0.02,
    "observed_sample_gap_median_seconds": 0.031915049999952316,
    "observed_sample_gap_max_seconds": 0.03375939999750699,
    "scope": "NVML adapter-wide used memory, including background allocations; sampled maximum, not instantaneous or process-attributed peak",
    "case": "image",
    "concurrency": 1,
    "single_token_arrivals": true,
    "prompt_tokens": [
      104,
      104,
      104,
      104,
      104,
      104,
      104,
      104
    ],
    "request_keys": [
      "fdac816cfa4645ce845e3e2fafc4b3ed-s3-image-c1-r0",
      "fdac816cfa4645ce845e3e2fafc4b3ed-s3-image-c1-r1",
      "fdac816cfa4645ce845e3e2fafc4b3ed-s3-image-c1-r2",
      "fdac816cfa4645ce845e3e2fafc4b3ed-s3-image-c1-r3",
      "fdac816cfa4645ce845e3e2fafc4b3ed-s3-image-c1-r4",
      "fdac816cfa4645ce845e3e2fafc4b3ed-s3-image-c1-r5",
      "fdac816cfa4645ce845e3e2fafc4b3ed-s3-image-c1-r6",
      "fdac816cfa4645ce845e3e2fafc4b3ed-s3-image-c1-r7"
    ]
  }
]

Native MTP3 raw SSE and requests replayed independently after archive download. All16 measured streams confirmed actual verified_mtp mode/positive verification calls from saved final frames. Recomputed all16 complete teacher/trial checks from raw responses, not only saved booleans. Archive477332bytes/162entries SHA256 38b2eb3dc5cdf4a155c992fe616c0d8c508809fb165a79144a4cd71045c2acb0 verified CRC/allfile digests. See native-mtp3-evidence.zip and native-mtp3-raw-replay-comparison.json.

B1 request bytes and prompt/output counts match the completed corrected-grid vLLM run. Provisional native/vLLM decode ratio text1.0531,image1.0461; aggregate output ratio text1.0573,image0.9837; native/vLLM TTFT ratio text0.7295,image1.6212. Thus image TTFT worsens and2x is not established. These are single-run arithmetic comparisons; SGLang corrected-grid, broad quality/precision equivalence, repetition, reference MTP/tuning and native batched concurrency remain pending. Earlier Garnet~36tok/s used a different profile, so do not attribute a controlled2x native improvement without matched ordinary-target timing.

Intentional shutdown after benchmark: first guarded process lookup matched both supervisor and bridge and refused to act. Narrowed to the actual bridge excluding run_logged; recorded identity5596 and stopped only its tree (cmd15268,xlang24524). Server supervisor exit1 is the documented intentional force-stop, not benchmark failure; gated driver exit0. GPU464MiB used/31724MiB free confirmed. Source stayed fixed throughout. Prepared separate MTP7 launcher, identical packed target/4K/pixel/arithmetic profile with depth7 and own temporary image path; requires same strict gates, no assumed gain.


## P077 - 2026-10-23: opt-in B2/B4 packed backend qualification prepared

Previous continuation made measured MTP3/archival progress and started the specific MTP7 worker/gate handles. Those handles remain live; do not restart for quiet full-program compilation. Remote running source stays5bf5c6cc0.

Prepared a separate research-only batch2/4 profile validator/factory, sharing the exact existing embedded mixed program/head/statistics construction with B1. Default validate_cuda_profile and ordinary factory continue requiring B1; general serving admission is unchanged. Research profile explicitly rejects unqualified batch/shape/recipe/model combinations. Current70 CPU study tests pass; this does not prove batch GPU support or memory fit.

probe_native_batch.py prepares actual B2/B4 ragged text/image/early-EOS groups from saved B1 oracle prompt/generated IDs, snapshots each reused vision output, pads inactive prompt rows, decodes with per-row active/limit controls, saves all raw generated/expected IDs before comparison and repeats after reset. It uses retained prompt IDs directly, so its eventual scope is backend batching, not tokenizer/frontend parity or HTTP transport. No throughput is calculated. It must pass actual GPU numerical/memory checks before implementing native HTTP batch scheduling or adding concurrent Garnet points to the chart. MTP batch support and B4 GPU admission remain unproven.


## P078 - 2026-10-23: optional resident vision binding targets observed graph invalidation

Previous continuation prepared explicit native batch qualification; MTP7 same supervised handles remain live and have entered strict greedy checking. Remote source remains5bf5c6cc0; no local follow-up was deployed into that run.

Source inspection found CUDAImageEncoder.encode_patches passes every newly allocated JPEG patch tensor directly to graph.run_device. TRT EnqueueWithCudaGraph resets captured graphs whenever binding pointers/shape signatures change. This is a concrete candidate for extra image latency; the earlier4B run's vision1capture/0replays does not quantify its TTFT contribution. Added default-off enable_resident_input(caller_io), one fixed-profile FP32 GPU patch buffer, native copy_into before each encode (existing identical dtype/shape and GPU fence checks), explicit byte count, and release of borrowed IO/buffer after graph close. Caller IO remains owned by decoder. Default direct path is unchanged. Research worker may explicitly request GARNET_STUDY_RESIDENT_VISION=1 and reports actual policy/extra bytes; no serving profile is silently changed.

Added CPU tests against actual encoder methods: replacement-image freshness/stable binding, shape-copy failure cannot run stale image, default direct behavior, double-enable rejection and caller-owned IO survival after close.73 CPU study tests pass; scripts compile and whitespace checks pass. Prepared real vision component qualification to compare six changing-color direct/staged full output byte arrays, save raw outputs before assertions, require actual graph replays/zero new capture failures and reject wrong patch shape before graph execution. Inputs remain freshly JPEG-preprocessed; this is a stable allocation, not a decoded-image/answer cache. GPU qualification and any end-to-end TTFT effect are pending; no speed claim.


## P079 - 2026-10-23: MTP7 exact-output pass but throughput regression preserved

Source5bf5c6cc0, run6cdfab09ff6c4a11a4e5ac9acab18f07 completed gate+benchmark exit0. Four retained VLM/prompt/native-semantic checks passed (same3/4 external exact parity), all16 full self-greedy pairs and real stream gates passed before timing. B1 corrected-grid cohorts each generated1024 actual tokens. Text TTFT124.0028ms, decode69.04852930tok/s, aggregate63.36420868tok/s; image TTFT270.6552ms, decode64.01375195tok/s, aggregate57.54462006tok/s. Sampled adapter peaks27616133120 and27620327424bytes, not instantaneous/process-attributed peaks.

475036byte/162entry native-mtp7-evidence.zip SHA256 8362130e731710274df9401d6f73a9aa81872f77586aec25fb857fe4beb17971 independently verified CRC/allfile hashes, both raw SSE cohorts and16 complete raw teacher/trial pairs. Request bytes/counts match MTP3; MTP7/MTP3 decode ratios text0.95747,image0.89335, aggregate0.94197/0.92532. This one run is a negative result, not a faster candidate. MTP3 remains current best measured native option;2x versus tuned feasible references is not established. Preserve all depths/results rather than selecting only favorable numbers.

Study-owned bridge26056/cmd19040/xlang27004 intentionally stopped after completed timing; stop identity record retained. Server supervisor exit1 reflects intentional shutdown, gated driver exit0. After archive finalized, synced sourcecbb2388b3 with6457 files/zero mismatches and verified GPU464MiB used/31724MiB free. Started vision-resident-input-qualification through raw command/NVML supervisor; GPU output/replay qualification is pending, no end-to-end image latency claim. B2 backend qualification is next after it is terminal, followed by real batch/concurrency performance only after correctness gates. Source line endings restored to original CRLF so cumulative core diff is limited to the actual stable-input change; numerical code unchanged by that cleanup.


## P080 - 2026-10-23: resident vision full-output/actual replay gates pass; B2 GPU qualification live

Sourcecbb2388b3, vision run4a5f9921263744338ed9ee68cd3433b5 completed exit0. Six changing red/green/blue inputs passed full direct-versus-staged byte equality; first three output arrays differ, ruling out constant/stale image output in this fixture. Direct path with retained distinct patch allocations had0captures/0replays; fixed input path1capture/4actual replays/zero capture failures. Extra buffer1572864bytes. Wrong patch shape rejected before graph replay. Whole-build/correctness NVML peak3178020864bytes is not warm serving peak. No component clocks or model TTFT were measured.

vision-resident-input-evidence.zip 11037147bytes SHA256 d1b644441a7b5905131928822dfaf2a9b84faec63ff47bff642953c8e13827ca independently verified CRC/every file digest and all six complete direct/staged rawF32 pairs after download. Includes raw arrays, device/command/NVML/source/binary metadata and source snapshot. This qualifies the mechanism for this vision profile only; real end-to-end image TTFT must be repeated before claiming benefit.

After terminal vision process and GPU464MiB used/31724MiB free, started native-batch2-backend-qualification using same frozen sourcecbb2388b3, raw correctness/NVML supervisor and distinct results path. It is live loading/building (observed nativePID9464), no actual B2 output/concurrency/performance claim yet. No source synchronization or another GPU model during this job. Final scope still includes fully matched/repeated stronger references (including feasible MTP), native GPU concurrency,4B HTTP/reference, full INT4 and broad quality/precision gates.2x unmet.


## P081 - 2026-10-23: actual B2 full-ID gate fails; row-isolation diagnostics prepared

B2 runaaa53455393341ebb9c5753970af1f0b, sourcecbb2388b3, terminated exit1 after preserving first group's full generated/expected IDs. Inputs are text41/image104 tokens. Text row first divergence is token0; image row token8; both generated128 tokens. Do not infer corruption versus ordinary shape-dependent numeric differences from IDs alone. No batch throughput, frontend support or B1-equivalence claim is justified. Whole build/correctness adapter sampled peak27022639104bytes is not warm serving peak. Original strict full-ID gate stays unchanged; failed run and its B1 request/response oracles are archived before any resync.

Prepared separate same-B2-engine prefill row-isolation diagnostic: text with image/text/short-color companion, image with text/image companion, each repeated after reset. Saves complete last-hidden and normalized hidden bytes before checks, along with first prediction IDs versus B1. Fixed-B2 row independence/repeat is checked separately from the still-failed cross-B1 equality gate. This tests ragged padding/row/state isolation without weakening the original qualification. Further math/head/kernel diagnosis depends on actual results.

Prepared research batch transport component requiring explicit study_batch_v1/physical-capacity handshake before healthy status. Bounded request grouping routes real worker events through per-request queues, preserves counts and rejects unknown IDs/wrong capacity. CPU subprocess/HTTP fixtures prove two clients group and receive distinct outputs, partial batches dispatch, invalid worker protocol is rejected, and unknown IDs fail clients. All77 CPU study tests pass. These synthetic fixtures are not model performance; a compatible real worker and numerical/HTTP gates remain prerequisites. This component is not deployed during the failed GPU job and no batch performance is measured.

B2 failure archive verification completed locally:130306bytes/23entries,SHA2560d79e863108f3efa9ffb1af90b777ec1f1b71d37c6ecf3dad654bf1eb89abca4; ZIP CRC/every file hash/raw ID divergences verified. After archive finalized, source3ef07624c transfer verified6460 files/zero mismatches and GPU464MiB used/31724MiB free. Started specific native-batch2-prefill-controls handle with distinct raw ledger/output; only diagnosis, no admission/performance claim.


## P082 - 2026-10-23: explicit next resident-vision experiment; B2 diagnostic verified live

Revalidated actual remote job native-batch2-prefill-controls through its supervised handle: running, native PID21048 present and report still starting with no captured cases. This is a verified wait, not a successful isolation result. Frozen remote source remains3ef07624c; no deployment/restart or competing GPU workload. Primary local checkout was independently verified at8e7f16377, matching the dedicated branch before this follow-up.

Added a separate serve_native_mtp3_resident_worker.bat launcher that explicitly selects depth3 and GARNET_STUDY_RESIDENT_VISION=1, with its own temporary image path. It otherwise preserves the measured MTP3 worker configuration. This makes the already component-qualified vision mechanism an explicit reviewable next end-to-end candidate. It is not deployed while the B2 diagnostic is running. Existing sixteen full greedy sequences, VLM checks, actual stream gates and corrected JPEG request preparation remain mandatory before timing; no latency or throughput result is inferred from launcher preparation.2x remains unmet.


## P083 - 2026-10-23: stronger vLLM MTP3 candidate prepared, not measured

Installed-source audit identifies method=mtp and num_speculative_tokens fields, Qwen3_5 MTP architecture rewriting and the installed predictor implementation. Added separate serve_vllm_mtp3.sh using the same pinned local checkpoint, explicit4GiB KV budget,4096 capacity,8 sequences and existing decode graph request, with speculative methodmtp/depth3 and a distinct default log. Mixed NVFP4/FP8 VLM compatibility, graph support, full generated-ID equivalence to the same reference engine without speculation and actual runtime performance remain unproven. Do not claim a tuned feasible MTP baseline from this file or launch timing before correctness gates.

Local Git Bash syntax check passed. Remote copy was staged only under fixed-root tmp for syntax inspection; WSL bash -n observation timed out, which is not a terminal result and did not justify restart. Read-only process inspection confirms WSL processes still present, native diagnosticPID21048 alive, and about50GiB free system RAM. Native supervised handle still running with no captured prefill cases; frozen source is unchanged. No reference GPU model was started. The observation timeout is retained as a setup limitation, not a model failure.2x remains unmet.


## P084 - 2026-10-23: same-reference MTP full-ID gate prepared

Added reference_mtp_equivalence.py to capture16 ordinary same-engine greedy sequences (8text/8corrected-grid image, seed3,128 output limit) with actual API token IDs, raw request/response files and hashes. Compare mode requires a passed16-case ordinary oracle of the same engine and matches semantic request digest, complete prompt IDs and complete generated IDs before declaring this limited sequence check passed. Partial/failed results are saved and cannot become a valid ordinary oracle. Engine identity/request mismatches and changed/empty IDs are rejected by CPU tests;79 study tests pass. This is not GPU compatibility, broad quality, server-configuration verification or a benchmark; actual server launch metadata remains independently required. No automatic timing follows these checks.

Native B2 prefill diagnostic remains live on its original handle/PID21048; latest read-only sample shows native CPU time advancing and GPU23013MiB used, with no completed case yet. This changes the observation from early loading to later model allocation/build activity, but does not prove correctness. Frozen remote source remains untouched; no restart or competing reference GPU model.

Follow-up request-contract inspection removed streaming-only stream_options from non-stream reference capture requests. A real local HTTP/subprocess fixture now executes all16 ordinary captures and16 exact comparisons, then injects a changed generated ID and confirms failed status plus raw response preservation. This is synthetic transport/gate coverage, never GPU/model evidence. All80 CPU study tests pass; whitespace check passes. Native B2 diagnostic remains live with no case output at this observation.


## P085 - 2026-10-23: B2 row isolation passes; original B1 mismatch remains

Run1b0b5c75d7af4266bfb0d08711a79e1b source3ef07624c completed exit0. Ten prefill captures covered five companion-row groups, each repeated after reset. Eight row-isolation/repeat checks passed full row0 hidden and normalized-byte equality. Packed decoder recorded1capture58replays and finalnorm1capture18replays, zero graph capture failures. This fixture finds no companion-row interference; it does not establish why B2 textfirstID2 differs from saved B1firstID8160, and original full B1-ID qualification stays failed. Whole-build/correctness sampled adapter peak27041513472bytes is not warm serving memory.

Archive738120bytes/43entries SHA2561418c36b0694f4bee20c0029a273ee9a42be87d9cd2a2bd1f2afc0bab35fa4f5 independently verified ZIP CRC, every file hash and16 complete raw row0 comparisons. Raw arrays/source/oracles retained before any sync. Added next bounded head-only diagnostic: replay the saved normalized vectors identically asB2 and individualB1 through native NVFP4 head and existing dequantized packed-head path, checking original replay and repeat stability. This can isolate head-shape dependence without rebuilding/loading the64-layer decoder. It is diagnosis, not an accuracy oracle or throughput result.


Head-only run59ecec36f0c544ab984dbbbf04c07984, source44f2f17a9, completed exit0. Saved group0 normalized vectors reproduce original[2,1919] and group4[1919,1919]. Native head fullB2, individualB1 and repeatedB2 predict identical IDs for both saved groups; existing dequantized packed-head path produces the same IDs in both shapes. This limited first-token replay does not identify the upstream decoder difference, but supplies no evidence of a head-only batch-shape argmax issue for these vectors. All raw inputs/results retained; no timing. Whole-process sampled adapter peak4794925056bytes is not serving memory. Next useful check is freshB1 decoder with the same retained-output configuration asB2, since saved MTP3 ordinary target used retain_linear_inputs=True while batch-study defaultFalse; this configuration difference must be isolated before assigning a numerical cause. Original B1 full-ID gate remains failed.


Head-only evidence archive95479bytes/18entries SHA2569a1de16e12a67ef04f0b90410ed86e5926b86e55d540e49252c1eb2edfb66765 independently verified CRC/all hashes and all retained full/single native/dequantized prediction equalities. Remote GPU returned to464MiB used and no studyxlang process before next experiment preparation.

Prepared probe_native_b1_configuration.py as a separate diagnostic, leaving original batch qualification assertions intact. Explicit retain0/1 selects the freshB1 factory configuration. Four saved prompts each repeat full generation twice; stores prefill hidden/normalized arrays, all actual/expected generated IDs and same-configuration B1-vs-savedB2 normalized maxabsolute/relativeL2 differences without asserting a numerical tolerance. Repeat-ID equality remains mandatory. Script compiles; actual results pending. Default next runretain0 isolates batch size from the earlier saved MTP3 target's retain1 setting.


## P086 - 2026-10-23: fresh same-configuration B1 diagnostic live; measured chart refreshed

After terminal head diagnostic/archive verification, deployed sourcef8587f5ee:6466files/zero hash mismatches, GPU464MiB used/31724MiB free. Started native-b1-no-retain-control through actual raw-command/NVML supervisor. Explicitretain0 and B1capacity4096; same raw prompt/vision preparation as prior batch diagnostic. Specific handle is live with initial known optional hashlib warnings, no result yet. Frozen source is not overwritten while running.

Created/exported/viewed Garnet-MTP-vs-vLLM-B1 PNG/SVG and dataJSON directly from archived completed cohort manifests for MTP3,MTP7 and corrected-grid ordinaryvLLM. Shows text/image decode and TTFT, honest dashed2x/halfTTFT thresholds against that ordinary reference only. All source archive hashes/runIDs are in dataJSON. It states referenceMTP/further tuning pending, one run per configuration/no uncertainty estimate, unqualified native concurrency beyondB1, and configured4Kcapacity not4Kinput. No SGLang image result is mixed into this corrected-grid comparison; original historical chart stays separate. Render visually checked; target remains unmet.


## P087 - 2026-10-23: opt-in smaller MTP draft profile prepared

Source inspection confirms MTP3 correction spans at most4 positions while existing draft program/buffers use targetT16. Added optional draft_chunk_size, defaulting to existing target width. Bounds require supported4/8/16, <=target profile and >=depth+1. Target verifier/64-layer engine remainT16; one existing BF16 MTP engine may explicitly useT4. Prefill normalizes at target width, then feeds existing shifted hidden/embedding/position/cache logic in smaller slices. Correction copies only the bounded prefix into the smaller draft buffers. No second draft engine or weight copy. New research launcher explicitly selects depth3/chunk4/directvision; older MTP3/7 and resident candidate launchers now explicitly pin chunk16 and their intended vision policy to prevent inherited-environment ambiguity.

83 CPU study tests pass, including production-constructor target16/draft4 shape separation, wide-prefill slicing/next-token position shift/last inactive slot, defaults/bounds, and existing verified-token event checks. No native GPU qualification or speed claim forchunk4. Frozen remoteB1retain0 diagnostic remains live/sourcef8587f5ee; this follow-up is not synced into it. Full16-sequence greedy/VLM/stream gates remain required before timing the smaller-draft candidate.2x remains unmet.

Added optional actual-worker profile requirements to the MTP pre-benchmark validator: requested depth/chunk/vision policy must match health before any quality/timing work. CPU tests reject wrong or absent fields, preventing a chunk4 launcher from accidentally timing a still-running chunk16 server. Existing default invocation remains compatible. Updated implementation plan to actual P087 evidence and outstanding full scope; remote live source unchanged.


## P088 - 2026-10-23: fresh B1 retain0 matches saved B1; B2 hidden divergence persists

Run46a76dcdc368483d9044df55b659483d sourcef8587f5ee completed exit0. Four prompts each repeated twice produce full generated IDs identical to saved MTP3 retain1 ordinary-target outputs; freshB1 repeats also match. Thus this fixture does not attribute B2 mismatch to retain_linear_inputs alone. B1/B2 normalized hidden relativeL2 differences text0.1570641748,image0.6951661019,red0.4529177477 (maxabs1.6928241/6.1347985/9.3186012). These are measured final-layer differences, not proof of harmless rounding or root cause. Original batch gate remains failed; next batch diagnosis needs intermediate layer/projection isolation before admitting concurrency.

Archive513658bytes/42entries SHA256842eff7efbdb69a38a760bcca548ef7d03ea4a623f79b1611f4318890896522a independently verified CRC/every file digest, all8 full-ID equalities and recomputed six raw normalized-vector comparisons. Whole-build/correctness sampled peak24898224128bytes is not warm serving peak. Raw inputs/arrays/source/binary identities preserved before deployment changes. Next serial GPU experiment is the prepared opt-in MTP3 draftT4 candidate with unchanged targetT16, directvision and mandatory actual-health profile/full16selfgreedy/VLM/stream gates before any timing. No2x claim.

After B1 terminal/archive verification, transferred sourcebd0ae1eb3 with6468files/zero mismatches and verified GPU464MiB used/31724MiB free/no studyxlang. Source-check command overall exit1 came from final no-match Get-Process query; verifier itself reported zero mismatches. Started actual native-mtp3-chunk4-server and native-mtp3-chunk4-gated-http on session95279, both verified live on their owning supervisor. Gated driver explicitly requires depth3/chunk4/directvision and will run all retained quality/stream checks before timing. No candidate output or performance result yet. Remote source stays frozen.


## P089 - 2026-10-23: checkpoint-prefix batch diagnosis prepared while MTP4 loads

Both original MTP3chunk4 supervised handles remain live, frozen sourcebd0ae1eb3; no output/performance claim yet. No competing GPU model is launched. Added a separate probe_native_prefix_batch.py for the next batch investigation: direct experimental packed graphs truncated explicitly to checkpoint prefixes1 and4, B1/B2 identical41-token text rows, capacity4096/T16/retain0, two reset repeats, saves complete last-hidden F32 arrays before checks. It requires same-profile repeat and duplicate-row equality, then records cross-B1/B2 exactness/maxabs/relativeL2 diagnostically. It avoids head prediction and can distinguish an earlyGDN/projection issue from a later first-attention-stage difference before loading64 layers. This is a truncated-graph investigation, not an architecture/profile admission or replacement for the failed full-model gate. Script compiles/whitespace passes; GPU execution pending after serial ownership release. No numerical root cause assumed.

Read-only installed vLLM qwen3_5_mtp.py retrieval and fixed-root snapshot both terminated exit0 despite recurring systemd-root-session warning. Downloaded11766bytes hashfbf31b1dd2b63eaf1911e325e3b43db602cf1d4268dfb2d388727e44caffadc2 matches prior installed-source audit exactly. It explicitly forces BF16 mtp.fc for modelopt_fp4 and implements multimodal embedding interfaces. This strengthens feasibility evidence only; it does not prove actual mixed-checkpoint runtime or establish an MTP baseline. Source snapshot/verificationJSON retained in outputs. Native chunk4 server/gated handles remain live; nativePID22080 CPU time advancing, no quality or timing output yet. No source resync or competing GPU model.


## P090 - 2026-10-23: MTP3 draftT4 full-ID pass, no measured throughput gain

Runf3b69318278b44a1930cf82f958cd47c sourcebd0ae1eb3 completed gated driver exit0. Actual health confirms targetT16/draftT4/depth3/directvision. Four retained VLM checks pass (same3/4 external exact parity), all16 full self-greedy sequences and actual stream/count gates pass before timing. text B1 TTFT126.01495000126306ms, decode71.3191551786732tok/s, aggregate66.83188282586441tok/s, sampled adapter peak27614035968bytes. image B1 TTFT276.00675000030606ms, decode71.85527027059194tok/s, aggregate61.84760889292747tok/s, sampled adapter peak27618230272bytes. Each cohort8measured/1024actual output tokens. Matched request bytes/counts vs priorT16MTP3; decode ratios text0.9889543/image1.0027814, aggregate0.9935177/0.9945137, TTFT1.1337286/1.0224773. Single-run result supplies no clear throughput benefit; keep defaultT16, preserve this negative candidate. No2x claim.

Archive481479bytes/162entries SHA256ffc081c5ecc66bbae0fc39cfa23e8130f66db42de45ff858a1a5f2f5303a55c8 independently verified CRC/everyfile digest,16 raw full greedy pairs and16 raw measured SSE event replays plus recomputed summaries. Bridge14412/cmd15368/xlang22080 identity checked and intentionally stopped after timing; server supervisor exit1 is shutdown, gated driver exit0. GPU returned464MiB used/31724MiB free. Next serial job is the small prepared checkpoint-prefix B1/B2 diagnosis to isolate the earliest batch divergence before native concurrency admission.84 CPU study tests passed on current local source.


## P091 - 2026-10-23: prefix1 exact across B1/B2, prefix4 diverges

Runedfaf2f5fccf4b8fac90bf6a21eb49eb source6692700cb completed exit0. Both reset repeats and B2 duplicate rows are byteexact for each tested prefix. B1/B2 prefix1 last hidden is byteexact; prefix4 maxabs0.037109375/relativeL20.011781515236205036. This bounds divergence to after the first layer and by the fourth, not yet specifically to attention: layers1/2 are additional GDN layers and layer3 is first full attention. No whole-model quality/admission or numerical-cause conclusion. Whole-build/correctness adapter sampled peak6187433984bytes is not warm serving memory.

Archive179312bytes/24entries SHA2567093c0aa054b4d63533073c8995434441acc0f4af40367c8f02401b596f940e3 independently verified CRC/allfilehashes, full raw duplicate-row/repeat comparisons and recomputed cross-batch differences. Added explicit unique prefix selection1..4 to the separate diagnostic, preserving default1/4; next fixed output and launcher select2/3. Original full64 batch gate remains failed. Script compiles/whitespace passes; new GPU run pending after source verification.


## P092 - 2026-10-23: GDN-only prefixes already differ; capture complete chunk states next

Run1b02eeb9b4594aa6a03530edaf7830db source260bc6e1f completed exit0. Prefix2 last-valid-token B1/B2 maxabs0.001953125/relativeL20.0013205276481965491; prefix3 maxabs0.025390625/relativeL20.008917359311986546. All same-profile reset repeats and duplicate B2 rows remain byteexact. Differences exist before first full attention layer; this does not identify whether earlier token/state differences precede the final-token prefix1 equality. Clarification: earlier prefix1 exactness covered only last-valid-token hidden, not every token or complete recurrent state.

Archive180722bytes/24entries SHA256b2a21c2c96e797502f56b8229de341dc0373706fac0027a4d2444f43527d1253 independently verified CRC/allfilehashes, all raw last-hidden repeat/duplicate comparisons and recomputed metrics. Added optional --internals to bounded GDN-only prefix diagnostic: save every chunk entire16-token hidden row plus each layer complete convolution/recurrent state row, before equality checks, with per-file digests. Per-row bounded readers remain <=1048576FP32words; no new native API or arithmetic change. Same-profile full internal repeat/duplicate equality is required, cross-B1/B2 differences remain diagnostic. Next selectprefixes1/2 in a fixed output; raw data is bounded to about200MB, with free disk previously measured~796GB. Script compiles/whitespace passes. Not a benchmark, batch admission or root-cause claim; original full batch gate remains failed.


## P093 - 2026-10-23: complete chunk capture finds tiny first-layer hidden differences

native-prefix12-internals source4ded85dec completed exit0. All full internal same-profile repeats/duplicate rows pass. Remote archive CRC/allfilehash/raw comparisons independently recomputed with NumPy: first-layer convolution and recurrent states byteexact across all3 chunks and both prefix engines, but prefix1 hidden differs9/81920words at chunk0 (firsttoken0,maxabs3.0517578125e-5),8words atchunk16(maxabs6.103515625e-5),2words atchunk32(maxabs1.52587890625e-5). Prefix2 chunk0 state remains exact; layer1 state diverges by chunk16. This shows why final-token-only prefix1 equality was insufficient. It suggests inspecting output projection/norm/MLP/residual arithmetic; no specific cause or harmlessness conclusion yet.

Raw archive170511846bytes/114entries finalized; remote integrity/full-array analysis passed. Original170MB archive is being downloaded through session95279, actual local partial-file size advancing. Local independent verification is pending; do not claim it complete. Remote raw sources/arrays/archive are retained and source changes cannot alter the finalized archive. Archive helper's initial wrong diagnostic-directory suffix was corrected before archive creation; no raw files changed.

Prepared isolated post-attention norm and three-native-NVFP4 MLP component B1/B2 diagnostics using identical real prompt embedding rows, not actual post-GDN residual activations. Three same-profile repeats and duplicate-row equality required, raw arrays/statistics before comparisons. No timings or CPU accuracy-oracle claim; graph projection count prevents silent BF16 fallback. It may expose a batch-shape arithmetic difference but alone cannot assign whole-model root cause. Script compiles/whitespace passes. Next serial diagnostic can run after remote evidence finalization while the immutable archive download continues; this is not a performance run.


## P094 - 2026-10-23: isolated norm and native MLP match across batch shapes

native-norm-mlp-batch-diagnostic on session48577/source9dd0d5025 completed exit0. For identical first16 real prompt embedding rows, isolated layer0 post-attention norm and norm+native MLP both produce byteexact B1/B2 outputs; all three reset-independent graph executions/duplicate rows match. Actual MLP graph admits3nativeNVFP4 matrices/zero externalBF16 fallback. This fixture does not use actual post-GDN residual activations, so it cannot rule out their norm/MLP arithmetic or establish whole-layer correctness. Next investigation should include the actual residual-add/branch computation; no root cause or fixed batch profile is established.

Norm/MLP raw archive finalized4160002bytes/28entries remotely, with source/commands/NVML/outputs; local archive verification pending behind the170MB internal-state download on session95279. That transfer remains active (local partial size grew past100MB), not a failed/missing job; do not restart it. Full prefix12 internal archive was already CRC/hash/raw-array verified remotely, with local independent verification pending. Current local source has84 passing CPU study tests; no new numerical backend default changed by these diagnostics.2x remains unmet.


## P095 - 2026-10-23: residual branch materialization experiment prepared

Norm/MLP4160002byte archive independently CRC/hash/full-array verified on remote: norm and native MLP each B1/B2 equal, all3 repeats and duplicate rows equal. Local download queued behind the still-advancing original170MB internal archive on session95279; local verification pending. Neither archive is modified by source updates.

Prepared separate probe_native_residual_batch.py. Same real16-token embeddings plus a deterministic flattened roll13 of recorded first-layer hidden as a synthetic second operand feed residual=x+operand, then out=residual+nativeMLP(norm(residual)). Compare default graph and existing explicit FP32 barrier after first addition at B1/B2, three repeats, full raw outputs, actual3nativeNVFP4/zeroexternalBF16 lowering. Saves the derived operand and source/derived hashes. Records whether barrier changes either shape; no correctness or speed assumption. Inputs are not actual attention output/post-GDN residual and there is no independent full-model oracle, so any result is limited arithmetic diagnosis. Existing backend/model defaults remain unchanged. Script compiles/whitespace passes; actual GPU run pending. This tests the remaining residual/branch hypothesis without changing native binaries.


## P096 - 2026-10-23: residual fixture negative; all large evidence now locally verified

native-residual-batch-diagnostic source2ac8e555a completed exit0. Both default branch and explicit residual FP32-barrier branch are byteexact B1/B2 for this deterministic synthetic second operand; barrier changes neither shape. Three repeats/duplicate rows and actual3nativeNVFP4 lowering pass. This does not reproduce the real layer difference and does not justify a backend barrier fix. Stop drawing whole-model conclusions from these substitute activations; further native batch diagnosis needs actual layer intermediates, while full stronger reference/vision/INT4/4B work remains outstanding. No numerical default changed.

Local transfer/independent verification is now complete for all three archives:
Full prefix12 internals:170511846bytes/114entries,SHA25632586aea53c0fd77b1d96935f700abe0061bd24f2cc1915400454929aac10564.
Isolated norm/MLP:4160002bytes/28entries,SHA2565b2f142947aed78b0fdd6e0420cb33d78b6d0e0877dca3eab92e1213cc9a81df.
Residual branch fixture:4493141bytes/30entries,SHA256213a70094ae95fca9c0fe4c10e1ee076e73fe7d9214d07c557f65aaf24d73003.
ZIP CRC/everyfile hash and complete saved row/repeat/state arrays independently checked locally. Derived residual operand reconstructed from retained source and roll13 recipe; all branch outputs independently compared. Full-state first-layer9/8/2 hidden-word differences and exact layer0 convolution/recurrent state were reproduced locally. Archives/raw failures/negative results retained. The170MB download finished normally, no restart. Best native measuredMTP3 remains~72tok/s;2x unproven/unmet. Next priority includes actual stronger referenceMTP feasibility/correctness and the remaining end-to-end vision candidate, not another unvalidated synthetic fix.


## P097 - 2026-10-23: stronger vLLM reference full-sequence capture setup

Native diagnostic jobs are terminal and GPU464MiB used/31724MiB free, no studyxlang, fixed output vllm-mtp-ordinary-oracle absent. All prior raw archives locally verified. Added run_reference_greedy_candidate.py: bounded readiness observation without server restart, actual advertised model-ID check, four retained JPEG VLM/token checks,16 actual full greedy captures/comparisons, launcher/config/script hashes and immutable startup-log snapshot. No automatic timing; actual resolved precision/speculation/graph config must still be reviewed independently. Existing request/count/full-ID gate reused unchanged. Script compiles/whitespace passes. Next ordinary vLLM run uses same qualified4GiBKV/cap4096/maxseq8/decode-graph candidate and a distinct server log; captures an actual same-engine ordinary oracle before trying feasible MTP3. No new reference speed claim or2x result.
