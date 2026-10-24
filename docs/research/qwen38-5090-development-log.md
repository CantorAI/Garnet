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

Deployed source99badc0d6,6472files/zero mismatches, GPU464MiB used/31724MiB free and no Windows18080 listener. Overall source-check shell exit1 came from final no-match listener query, not verifier. Started original handles vllm-mtp-ordinary-server and vllm-mtp-ordinary-greedy on session95279; both verified live. Distinct log vllm-mtp-ordinary-server.log/output vllm-mtp-ordinary-oracle; no speculative-config flag in ordinary launch. Driver performs only readiness/quality/full-ID capture; no timings yet. Actual resolved startup config remains to review. No source sync or competing GPU model while live.


## P098 - 2026-10-23: reference config observed; wider-prefill sharing constraint identified

Original vLLM ordinary server/greedy handles remain live on session95279. Startup logs resolve v0.32.0,Qwen3_5ForConditionalGeneration,4Kcapacity,4GiBKV/BF16 KV/F32SSM,maxseq8,FULL_DECODE_ONLY sizes1/2/4/8 and prefixcacheoff. Engine631 loading3 checkpoint shards through the existing Windows9P path; no clone/new model copy or restart, no new quality/timing result. Actual complete resolved engine configuration still must be retained/reviewed after readiness; no speculation baseline claimed from startup alone.

Read-only native-source inspection confirms external_weights currently requires BF16, while modelopt NVFP4 lowering explicitly retains embedded opaque constants until external packed bindings can share weights. Recorded concrete shared-compressed-pool/wider-static-prefill candidate in implementation plan: exact packed bytes/scalars, guarded external bindings/ownership stats, one pool for prefillT128/decodeT16, component and full quality gates before clocks. Current41/104 prompts require3/7T16 target calls; this arithmetic does not prove their latency contribution or any speedup. Candidate is not implemented; active remote source remains99badc0d6, no sync while reference loads.


## P099 - 2026-10-23: actual ordinary vLLM oracle complete; MTP capture buckets corrected

vllm-mtp-ordinary-greedy source99badc0d6 completed exit0: fourJPEG VLM/token checks pass; all16 ordinary captures contain128 actual generated IDs each. Startup snapshot confirms speculative_config=None,quantization=modelopt_mixed,FlashInfer NVFP4+FP8 kernels, BF16KV/F32SSM, four actual FULL graph captures and37774 resolved KV tokens (not real full4K concurrent admission). No new clocks were measured. Identity-guarded LinuxAPI368 SIGTERM after completed quality capture, server supervisor exit0, GPU back464MiB/31724MiB free. Raw requests/responses/startup/source evidence being archived before any deployment.

Installed-source graph audit (four exact file hashes retained) shows uniform_decode_query_len=1+num_speculative_tokens and capture sizes rounded to that multiple. MTP3 therefore verifies4 queries/request; previous unrun candidate's max8/sizes1/2/4/8 would only retain4/8 token buckets. Corrected separate MTP3 candidate to max32/sizes4/8/16/32 for C1/2/4/8 coverage, preserving ordinary launcher. This is source-driven candidate configuration, not proof of actual graph coverage or runtime compatibility. Git Bash syntax/whitespace checks pass; actual pinned mixed checkpoint MTP loading/quality/performance still pending.2x unmet.


## P100 - 2026-10-23: ordinary oracle archive verified; actual vLLM MTP3 live

Ordinary archive111448bytes/67entries,SHA25672aaf93192c7dd27e90dd6f68c9051db015dcb7811ef25a05087d0e85d161518 independently verified locally: ZIP CRC/everyfile hash,16 raw request/response actual full-ID pairs,128 outputs per sequence, semantic request digests, and startup evidence speculationNone/mixedNVFP4+FP8. Not a timing run. Actual ordinary reference remains preserved for strict same-engine MTP comparison.

After ordinary terminal/freeGPU and archive verification, deployed source69d4c953f,6472files/zero mismatches, GPU464MiB used/31724MiB free. Started vllm-mtp3-server and vllm-mtp3-greedy-equivalence on session95279; both observed live. Candidate explicitly methodmtp/num3,maxcapture32/tokenbuckets4/8/16/32; quality driver uses saved ordinary full oracle, fourVLM checks and no automatic benchmark. Actual checkpoint load, resolved speculation/graph configuration and exact output checks remain unproven. Preserve any failure; no source mutation/otherGPUmodel while live.2x unmet.


## P101 - 2026-10-23: MTP checkpoint loaded/compiling; local packed-pool API prototype unqualified

Actual vLLM MTP server379/engine743 progressed through mixed checkpoint plus draft loading (log20.66GiB model memory) into torch.compile. Original server/equivalence handles remain live; no output-quality or timing result yet. This establishes loading progress, not complete MTP support. Frozen remote source69d4c953f remains untouched; no native rebuild/otherGPUmodel during reference work.

Began local uncommitted CUDATensorIO shared compressed projection pool prototype in cuda_tensor_io.cpp/.h. Explicit opt-in metadata/query/materialization APIs validate named bounded NVFP4/FP8 checkpoint matrices, finite positive calibration/product and block scales, retain raw opaque bytes, repack block scales once, wrap readonly allocation owners, cache per name, clear on close and account compressed bytes separately. Current SDK exposes INT32/UINT8, not INT8 storage; prototype packs raw bytes into opaque INT32 words without numeric conversion, requiring matching external plugin bindings later. No caller or model default uses this API. External TensorRT binding/plugin support, build, malformed-fixture/resource/numerical/lifetime GPU tests and full profile qualification remain unimplemented/pending. Whitespace checks only; do not label this built/tested/deployed or a speedup. These two C++ files intentionally remain uncommitted locally until qualification.

Added a preflight guard to local source-transfer helper: refuses any uncommitted source before writing an archive, so the live reference cannot accidentally receive this WIP or claim a clean commit. Direct invocation rejected the dirty checkout as intended before packaging. Do not sync/build the local prototype into the active reference run.2x remains unmet.


## P102 - 2026-10-23: actual vLLM MTP3 strict greedy gate failed; repeat stable

Frozen remote source69d4c953f MTP3 server started and passed four JPEG/token/semantic VLM checks. Quality driver completed exit1 on first text sequence; no timing was admitted. Local raw ordinary and MTP reports confirm identical semantic request and all41 prompt IDs,128 actual generated IDs each, first generated divergence at zero-based26 (27th token): ordinary494 versus MTP6197. Repeating the exact saved request against the same live MTP server returned identical prompt and complete128 generated IDs to the first MTP response. This establishes repeat stability for this one case only; numerical/kernel or algorithmic cause remains unproven. Do not label MTP unsupported or relax exact-ID correctness gate from this evidence.

Preserved raw first response, repeated response, failed report and startup snapshot with SHA256 in outputs/vllm-mtp3-first-divergence.json. Actual server supervisor vllm-mtp3-server was polled running on95279; live model source unchanged and local compressed-pool C++ prototype remains uncommitted/unbuilt. Next investigation should compare ordinary versus speculative target numerical behavior at the shared prefix before any performance run. No new benchmark or2x claim.


## P103 - 2026-10-23: MTP3 logprob repeats and complete failure archive verified

Two untimed identical requests with logprobs/top5 enabled preserved the original full128 MTP generated IDs. Complete reported logprob arrays were identical across repeats. At generated index26 selected token involved had logprob-1.192812442779541, next candidates from/comma-1.567812442779541, of-1.692812442779541. This is the MTP distribution only; ordinary same-prefix probabilities remain unavailable, so no numerical-root-cause or harmless-rounding claim.

Archived first gate failure, four VLM checks, request/response, logprob repeats, startup snapshot, source/config/ledgers and preparation identity: vllm-mtp3-failure-evidence.zip70967bytes/36entries/SHA25699f359b0200ffec6a8c12f4d7166b4842d6655c98d296180598c574d04a09356. Local independent CRC/everyfileSHA/full-ID/logprob-array verification passed; outputs/vllm-mtp3-failure-verified.json records details. Identity-guarded study server shutdown requested after archive to enable ordinary diagnostic comparison; terminal/freeGPU not yet verified at this log entry. No timing/2x claim.


## P104 - 2026-10-23: shared compressed pool object builds; plugin storage admission passes locally

Native MSVC/CUDA13.2 local build compiled cuda_tensor_io.cpp prototype successfully. Extended existing NVFP4/FP8 plugin opt-in storage admission to INT32 opaque byte containers alongside legacy INT8 constants, keeping BF16 activation/output and unchanged16-byte/version1 metadata. Physical dimension checks preserve exact raw byte capacity (NVweightK/8, FP8K/4, NVscalesbytes/4 for INT32); null buffers/wrong dtype/layout rejected before kernel dispatch. No numerical conversion added.

Both updated test executables built and --formats-only passed without GPU tests. Added default GPU test branches for real external INT32 inputs versus legacy embedded constants, serialized-engine CPU-oracle/direct/graph checks; these new GPU branches are compiled but NOT executed. Local compressed-pool-format-validation.json records source/binary hashes and raw CPU admission results plus retained build logs. Source patch snapshot retained as compressed-pool-candidate.patch. Six code/test files intentionally remain uncommitted and undeployed until GPU/resource/alias/lifetime/full-builder qualification; no full runtime or speed claim.

Remote MTP3 study supervisor completed exit0 after identity-guarded stop, GPU10MiB used/32178MiB free observed. Started separate ordinary logprob server+health-waiting diagnostic using frozen69d4c953f source and same saved request; actual API1114/engine1241 checkpoint loading observed. No source sync and no restart due observation timeout. Ordinary probabilities still pending. Full2x objective remains unmet.


## P105 - 2026-10-23: ordinary/MTP shared-prefix probability ranking difference verified

Ordinary logprob diagnostic completed exit0; both repeats preserve saved ordinary128 full IDs and identical reported logprob arrays. At common generated prefix26 (same41prompt IDs), ordinary selected from494 with logprob-1.3076276779174805, tied with involved6197 at identical reported logprob; MTP3 selected involved with-1.192812442779541 versus from-1.567812442779541 (gap0.375). Both engines are individually repeat-stable for this case. This establishes a reported target distribution/ranking difference at shared prefix, not its numerical/kernel/algorithmic cause; do not call harmless rounding, MTP unsupported, or relax strict full-ID gate. No timing admitted.

Ordinary evidence archive47166bytes/12entries SHA2564646a20c255eae7420a6eb66380eea0167a2f092deaa05bb83058ea7a58fac52 independently verified CRC/everyfileSHA plus twofull-ID/logprob repeats and actual startup speculative_config=None. Comparison with independently verified MTP archive recorded in outputs/vllm-mtp3-ordinary-logprob-comparison.json. Source remains frozen remote69d4c953f; local six C++/test prototype files uncommitted/undeployed.2x unmet.


## P106 - 2026-10-23: opt-in shared compressed bindings build/link;84CPUtests pass

Added external_compressed_weights checkpoint-name dictionary to captured Qwen operations: each value [packed] FP8 or [packed,tiled_scales] NVFP4, declared INT32 network inputs only. Builder independently checks pinned metadata geometry/calibration/tiled-scale dimensions, rejects cross-weight storage aliases, conflicting tensor IDs, unused bindings and mixing embedded/external copies of one checkpoint weight. Registers logical projection metadata without compressed constants; separate external input/byte counts exposed in compiled graph stats and reset on close. Pool API remains caller-owned/readonly and separate from graph-owned byte counts. No existing factory/default opts in.

Local full SM89 Garnet library build/link succeeded;84CPUstudy tests passed. Prior packed format admission tests pass; expanded external INT32 GPU oracle/serialization/graph test branches compile but have NOT run. First builder object compile found newly inserted dictionary block in wrong Qwen branch; moved to branch with external lookup, rebuilt successfully. Initial CPU invocation from chatcwd lacked xModel import path; reran from repository root successfully. Both failed logs preserved, no implementation assertion weakened. Local build/source hashes in outputs/shared-compressed-build-validation.json; raw build/test logs retained.

Plan update: commit experimental opt-in code now, before GPU qualification, so transfer uses a clean reviewable revision and existing dirty-source transfer guard remains effective. This supersedes earlier intention to leave candidate uncommitted until GPU tests; qualification remains pending and no primary/deployed asset replacement is authorized by test status alone. Next: actual5090 external buffer/plugin/pool resource+numerical/lifetime tests, then two-engine profile integration and full VLM/HTTP gates before timing. Current remote ordinary diagnostic server completed exit0, GPU10MiB used/32178MiB free; no live benchmark. Full2x objective unmet.


## P107 - 2026-10-23: actual5090 external compressed plugin components pass

Verified idle GPU10MiB/32178MiB free and ordinary diagnostic server terminalexit0. Before source update, preserved existing qualified study DLL in fixed build/qualified-before-shared-compressed/garnet.dll and verified SHA2560040b5a8e96a4a865dbbb82e7fda777259a7575619b4d8759ea2f5c211dae688; primary D:/CantorAI/Garnet untouched. Synced clean experimentala218f47b46814b5dfc0689897c565b81e1c410c2;6472sourcefiles verified zero mismatches. First verifier invocation found missing root helper; uploaded same retained helper and source verification then passed. No live GPU run/source mutation overlap.

SM120 NVFP4/FP8 component targets built exit0. GPU correctness supervisor completed exit0: all10cases passed. Synthetic256x256 NVFP4 M1/16, FP8 M1/16/4096, each embeddedINT8 and externalINT32. Existing CPU oracle checks, serialized engine execution, direct execution and CUDA Graph replay exercised. NVFP4 exactBF16 oracle; FP8 existing .0625+1% tolerance unchanged. This qualifies external byte-container plugin path on5090 for these fixtures only, NOT real-checkpoint pool/builder integration, dual-engine sharing/lifetime, full model, TTFT or throughput.

Archive shared-compressed-plugin-evidence.zip110067bytes/17entries SHA256f7bc20ae33298b33434ce86ed5f995a9705aa56d4c69d9ac35546bd2863fc71f includes actual raw build/GPU command ledgers, executable hashes/binaries, source/tests, source/backup manifests. Independently verified local CRC/everyfileSHA,10successmarkers and terminalexit0 records. outputs/shared-compressed-plugin-verified.json retained. Main experimental study library build started separately on frozen same source, originaljob shared-compressed-library-build still pending at this log entry; do not restart for observation timeouts. No benchmark fabricated;2x remains unmet.


## P108 - 2026-10-23: committed actual checkpoint pool byte/lifetime probe; build dependency copy verified live

Added probe_compressed_pool.py (initial005c5d082, detailc3090bb2b), py_compile passes. It requires actual NVFP4 layer0MLPgate and FP8 layer0GDNout metadata, compares full packed GPU bytes to pinned safetensors data, independently repacks/compares every NV block-scale byte, checks metadata-only/no-resident-allocation behavior, invalid matrix admission, readonly flags/writable-copy rejection, repeated resident accounting, and retained tensor ownership after pool close. No timing or projection accuracy claim; actual run pending. Uses existing model files and fixed outputfolder, retains hashes rather than duplicate model data. Root runner/probe staged separately, not in frozen Garnet source tree; ledger must record probe SHA plus actual DLL hash when launched.

Main5090 build original supervisor shared-compressed-library-build remains running. Actual cmake25420 build and cmake23976 -E copy_if_different command observed, comparing existing TensorRT multiple architecture resource DLLs and CUDA dependencies into existing build/bin; no link.exe then, and no failure/timeout inference/restart. Source remains frozena218f47b4; current root probe staging does not mutate compilation sources. Do not run checkpoint probe before build terminalexit0 and binary hash capture.


## P109 - 2026-10-23: dual-profile MLP gate prepared; corrected SGLang quality capture started

Committed probe_compressed_mlp.py37549f13d, py_compile passes. It compares full actual layer0MLP outputs from same-profile embedded and external weights at B1/T4 and B1/T128, then retains both external engines together on one three-matrix pool, checks native/external counts+logical byte accounting, graph replay, and interleaved outputs. It is unexecuted; no sharing/performance claim. This explicitly tests differing prefill/decode shapes without assuming cross-shape bit equality.

Main build originalhandle remains live; actual dependency comparator23976 read counter progressed2266302112 to2633071712 bytes, write0 (existing file comparison). Verified wait, no restart or failure inference. GPU10MiB used/32178MiB free observed. To advance full baseline scope during this CPU/I/O wait, started separate sglang-corrected-grid-server and sglang-corrected-grid-quality on95279 against frozena218f47b4 scripts, no source mutation. Root launcher matches original ordinary graph configuration except dedicated log and pre-load occupied-port bind guard; retained/reviewable serve_sglang_corrected_grid.sh. Read-only ss inspection observation timed out before any server launch, not a server failure/restart.

Quality driver only: fourVLM cases and16full greedy captures on corrected binary-grid JPEGs, no automatic benchmark. Do NOT time while main build active; do NOT start native pool/MLP GPU probes while SG owns GPU. Actual SG loading/quality unproven at this entry; original launcher handles must be polled. No2x claim.


## P110 - 2026-10-23: opt-in shared mixed weights connected through packed decoder; GPU qualification pending

Added explicit external_compressed_weights flag through packed decoder/program/chunk. Only enabled mode creates INT32 packed/scales symbols from one caller IO pool, supplies per-layer compressed binding dictionaries to existing native mixed lowering, and leaves known BF16 GDNin_proj_a/b as small checkpoint constants; unexpected BF16 projection fallback rejected. BF16/joined and embedded modes are mutually exclusive with this new mode. Existing defaults unchanged in configuration; GPU regression remains unproven. Per-matrix symbol slots keep packed/scales bindings separate and tracked. Decoder retained-memory accounting includes compressed pool/scales once across engines, separately from graph logical input-byte counts.

New explicit create_cuda_shared_compressed_decoder candidate factory uses B1 existing capacity validation, prefill32/64/128 and decode4/8/16, shared canonical state/output storage. It requires every engine192NVFP4+208FP8 native projections and400external bindings/zeroexternalBF16, exactly400resident matrices with one physical pool byte count. Not wired into serving worker/default factory; full checkpoint numerical/greedy/VLM/HTTP and speed tests still pending.

py_compile passes and84CPUstudy tests pass (raw outputs/shared-compressed-decoder-cpu-tests.log). Retained source hashes in shared-compressed-decoder-validation.json. Preserved original cuda_chunk.py CRLF convention; whitespace verification uses cr-at-eol plus all normal trailing-space checks. No remote source sync while main build/reference active. Main comparator23976 actual read progressed5546413472bytes, SG Linux455 observed live at02:18elapsed/RSS642424KiB; quality still waiting, no new timing/2x claim.


## P111 - 2026-10-23: native shared compressed library build completed and binary archived

Original shared-compressed-library-build completed exit0, no restart. Actual native5090 garnet.dll5912064bytes/SHA256a959ec2c2da3f5eae6cc17cf197b3dd84f81db867e61ef3503d7709d998ac99b from frozena218f47b4, runtimeDLL19659776bytes/SHA2568dabab841d460120b4e0b5c1539851e9ff17dbb0f5113beece4017f9f3fac4b3 matches prior retained runtime identity; original qualified Garnet DLL0040b5a8e9 backup unchanged. Current source4acea4a37 Python whole-decoder candidate is local/pushed only, not deployed into live reference source.

Build/source/binary archive shared-compressed-library-evidence.zip2292432bytes/19entries/SHA2568bb25a20532a84b3d893d9b3e4e0b1c28d3554c07a4e7514e6e53f63c7dfebde independently verified local CRC/everyfileSHA and actual included DLL against recorded identity. outputs/shared-compressed-library-verified.json retained. No actual pool or full-model execution of new DLL yet. SG corrected server became ready, actual first JPEG prefill93tokens/cached0 and200response observed; quality handle still live, no timing. Keep GPU exclusive to SG until completed/guarded shutdown before native probes.


## P112 - 2026-10-23: corrected SGLang quality complete; matched HTTP benchmark started after build terminal

sglang-corrected-grid-quality completed exit0: allfour JPEG/token/semantic checks and16ordinary full greedy captures (8text/8image,128generated IDs each) passed script invariants including actual41/104prompt counts. Actual server_info retrieval completed exit0; filtered metadata retained at results/sglang-corrected-grid-metadata.json with raw-info/config/index hashes. ActualSGLang0.5.22,context4096,BF16KV/F32SSM,maxseq8,total32768,full decode graphs1/2/4/8,prefillgraphsdisabled,radixoff,FlashInferautotune disabled,speculative_algorithmNone. Server args quantization field is null (auto-config argument, not evidence of BF16 fallback); startup logs show fused SiLU+FP4 quantization enabled. Resolved per-layer kernel/tuning audit still separate; do not call this fastest tuned baseline.

After main native library build terminalexit0 and quality completion, launched sglang-corrected-grid-benchmark originalhandle95279. Same corrected binary-grid JPEGseed3/image-first workloads as vLLM, C1/2/4/8,8measured requests/cohort,2warmup waves,128outputs. Own benchmark harness records rawSSE/actualcounts/cohortTTFT/decode/aggregate/NVMLpeak; outer setup ledger process duration is NOT a model metric. No nativeGPUprobe/source sync/rebuild/heavy remote hashing/download while timing active. Actual benchmark completion/metrics/rawstream validation pending. No2x claim.


## P113 - 2026-10-23: localhost transport mismatch identified and measured; comparison guard added

Completed SG corrected-grid trialf948ed75a0af457199c313f349ba19e5 exit0/8cohorts, but client used http://localhost:18081 while authoritative archived vLLMcf140177a used http://127.0.0.1:18080. TextB1 reported~2185.87ms TTFT despite~65.30decode; imageB1~2215.93ms. Excluded entire trial from matched speed-ratio/TTFT claims pending controlled IPv4 rerun. Rawtrial remains preserved, not relabeled as model speedup.

After benchmark terminal, paired TCP diagnosis (no model requests) completed exit0: median localhost2.061246299999766s versus literalIPv40.0151025999984995s. Actual resolution includes ::1, refused after2.0430976999996346s, followed by immediate IPv4 success. This independently establishes hostname/IPv6 fallback overhead explaining most observed additional TTFT; actual baseline rerun still required, no synthetic subtraction/corrected metrics. Raw outputs/sglang-loopback-transport-diagnostic.json retained.

Timing CLI now requires literal127.0.0.1. Raw-run verifier records transport scheme/hostname/path (ports deliberately engine-specific); comparator suppresses ratios when transport evidence differs or is missing. Added regression fixtures for differing hostnames, matching host/different port and missing evidence.85CPUstudy tests and py_compile pass; original remote sourceA218 untouched while trial/server live. Source hashes/legacyraw archived separately; no automatic2x claim. SG actual cache settings show enable_prefix_mm_cachefalse and enable_mm_global_cachefalse/radixdisabled; preprocessing cache confignull, not assumed as proof of cache-free preprocessing. Native pool probes remain queued until SG releases GPU.


## P114 - 2026-10-23: corrected IPv4 reference verified; real compressed pool passed

SGLang IPv4 run25c22636d2534ae4bc965bb364feb4f1 completed exit0. Archive752147bytes/440entries SHA256c83557d551ccca61d0092566e9e7ad236a356ef4adda1ccbb5549e545acb820b independently verified CRC/everyfileSHA,64raw measured SSE,8recomputed summaries and recorded-window NVML maxima. Actual payload bytes, prompt/output counts and literalIPv4 transport match archived vLLM and native MTP3 B1. TextB1 TTFT153.89045ms/decode65.29871/aggregate61.02186; imageB1 TTFT156.38365/decode61.99053/aggregate57.96135. C8 aggregate text435.06180/image404.83384. Not fastest tuned/MTP reference, broad quality or real full4K concurrency admission.

Native MTP3 versus best supplied ordinary references: B1 decode ratios1.05306text/1.04611image, TTFT ratios0.72952text/1.72613image.2x unmet. Updated chart Garnet-MTP-vs-References-B1.png/.svg and source/data retained; rendered/viewed. Original localhost trial retained and excluded; see separate transport-correction note for verified archive and TCP diagnosis.

After archive verification, identity-guarded SG455 SIGTERM, server supervisor completed exit0, GPU10MiB/32178MiB free. Started actual-compressed-pool-qualification on frozena218 source/newDLL a959ec2c2da3f5eae6cc17cf197b3dd84f81db867e61ef3503d7709d998ac99b. Completed exit0 with passed report: actual layer0NVFP4MLPgate and FP8GDNout packed bytes exact to checkpoint, independent tiled-scale repack exact, readonly flags/copy rejection, metadata/invalid-lookup guards, repeated resident accounting and retained GPU tensors after pool close exact. Loaded76021760packed+5570560scale bytes/2entries; closed counters0. This is resource/byte qualification, not projection/model timing. Optional embedded-Python hash-provider warnings persist; actual SHA256 checks succeed.

Pool archive47569bytes/15entries SHA2563f3492c8489e5372fb8b0de809dc343d36b09201610d9b51866708be75b64117 retained; independent local archive verification pending at this entry. Next originalhandle actual-compressed-mlp-qualification started on95279 with same DLL/source: same-profile embedded vs external full-output checks for B1/T4 andT128, two resident engines/one three-matrix pool. Outcomes pending. No remote source sync or other GPU task overlaps this qualification.

## P115 - 2026-10-23: real two-profile MLP exact; raw evidence independently verified

Pool archive independently passed CRC/everyfileSHA, recorded exit0, readonly/cache/closed-counter report invariants and actual DLL artifact identity. GPU raw byte/lifetime checks were executed by retained hashed probe; offline report verification is not a second GPU execution.

actual-compressed-mlp-qualification completed exit0 on frozen a218 source and a959 native DLL. Actual layer0 MLP at B1/T4 and B1/T128: full outputs match same-profile embedded reference exactly, repeated direct/captured/replayed outputs remain exact, and interleaved runs of two simultaneously resident external engines match. One shared three-matrix pool retains 150405120 physical packed/scale bytes across both engines; each graph's external input byte count is a logical reference, not a second allocation. No complete decoder, VLM or performance qualification implied.

Archive actual-compressed-mlp-evidence.zip5212285bytes/25entries SHA256eb0cc33ce3a618d9e2eadbf8202478da4262d1e0f820740f97e9b6b18b5b3a24 retains all eight full F32 output files, source, command/executable identities and report. Independently verified CRC/everyfileSHA and every output byte for both profiles against corresponding embedded raw reference. Structured verification retained in outputs/actual-compressed-mlp-verified.json. Next: deploy the committed opt-in Python factory, qualify mixed decoder layers and full64-layer behavior before any new timing.2x remains unmet.

## P116 - 2026-10-23: complete source update and full64 shared candidate smoke

Before deployment, caught cuda_chunk.py missing from the old manual transfer list for the new Python change. Prior a218 experiments remain valid for their own revision. Export helper now includes all tracked changes from2355fc4b5, rejects deletions and dirty trees, and records the changed-path list. Actual preflight covered183tracked updates/189patchfiles, including the missing dependency;28native/CMake file hashes match the tested library package. Remote installer verified6472old source files and both preserved/current DLL identities, then installed Python/sourceb60e21b75. Remote verifier checked6476files/zero mismatches and complete183path coverage. Tested a959 DLL was reused with unchanged native source; no unnecessary rebuild or production asset replacement.

Committed standalone full-model probe d61f57883 and resource monitor f91a334ad; actual root copies are hashed in the command ledger. Full64 B1/cap4096 shared T4decode/T128prefill constructor passed its192NVFP4+208FP8/400external-binding contracts for both engines and one400matrix pool. Six real inputs, each repeated twice, passed: text/image128outputs and four VLM semantic cases. Repeated full generated IDs and5120word normalized vectors agree. Only6of12raw sequences match oldT16 IDs, so these smokes do not establish cross-profile arithmetic/quality equivalence or authorize timing/MTP claims.

Sampled adapter peak during construction plus correctness was26311544832bytes (about24.5GiB), not a serving/process/instantaneous peak. Native child and resource supervisor exited0. Archive shared-decoder-full-smoke-evidence.zip319282bytes/32entries/SHA2563eac82f8a0200531cde9be8b184201563da52049cbe4d628e5bdde642405cc7b independently verified CRC/everyfileSHA, all12normalized outputs, repeat IDs, VLM answers,400matrix counters and raw sampled memory maximum. Detailed verification/report/resource files retained under outputs.

Implementation plan updated to actual opt-in status. Next: controlled same-profile complete arithmetic/quality comparison, verified MTP target equivalence and streaming gates, then matched HTTP pilot. Reference MTP/tuning, broader quality and cache precision, INT4/4B, actual long-context/concurrency admission, native batching and repetitions remain required. No new shared-candidate benchmark exists;2x target remains unmet.

## P117 - 2026-10-24: explicit shared streaming profile and full arithmetic probe

Committed8524fce5b adds an opt-in shared compressed worker profile: packed B1/T4 decode/T128 prefill, default MTP draft4 when selected. Legacy defaults remain T16. Invalid shared/nonpacked and draft/target combinations fail before vision/model allocation. Readiness now reports actual decoder chunk/prefill/shared attributes. Qualification can require these fields and rejects stale or missing values. All89 CPU study tests passed, including new incompatible-profile and stale-health fixtures. An initial targeted test invocation omitted the study directory on PYTHONPATH; corrected discovery passed. These CPU results do not qualify new GPU streaming or performance.

Committed6434d8071 adds a separate-process same-profile full64 arithmetic comparison. On the previously verified frozen remote b60 application and a959 native DLL, the embedded reference job shared-program-embedded was started in the fixed study directory, with probe and DLL hashes retained by run_logged. It prepares saved real text/image B1/T128 inputs, zero initial state and identical masks/positions/slots; each mode hashes every F32 word of the packed hidden/state output in bounded chunks for three repeats. External mode must match all reference chunk hashes and input identities. The reference process is live at this entry; neither arithmetic success nor benchmark admission is claimed. No source synchronization occurred while it ran. New worker changes remain unqualified on GPU until deployment and strict MTP/SSE gates pass.2x remains unmet.

## P118 - 2026-10-24: embedded full arithmetic reference completed

Reference job shared-program-embedded subsequently exited0. Both real inputs completed three repeated full packed outputs, each167 bounded hash chunks covering hidden plus173932544 state words; repeated hashes agree and graph capture failures are0. Downloaded embedded-report.json and independently checked these report invariants locally. This proves producer-recorded reference repeat consistency; complete raw-state bytes were not retained for independent GPU re-execution. The archive will retain all hash tables and the first1048576-word raw chunk per output.

Started the external shared-weight comparison only after the embedded process terminated. Job shared-program-external is confirmed live and uses the exact saved input bytes and same frozen b60/a959 application/library. Do not claim cross-storage equivalence until it exits and its full-output comparison passes. No new timing has been performed. Dedicated shared MTP3 launcher and explicit shared=0 pins in all legacy MTP launchers were committed16a9e67e1; four CPU profile tests, including launcher pinning, pass. These launchers are still not deployed during the live arithmetic test. Archive/integrity-check helpers are prepared and reject live or incomplete jobs.

## P119 - 2026-10-24: same-profile full arithmetic equality and MTP deployment

External job shared-program-external exited0. All six full output chunk tables (text/image, three repeats each) exactly match embedded reference, using identical saved inputs, B1/T128/cap4096 and zero initial state. This qualifies the recorded same-profile initial-prefill arithmetic comparison for compressed external binding, not cross-T16 precision, subsequent decode state, generation quality or speculative correctness.

Archive shared-program-equivalence-evidence.zip34684477bytes/34entries SHA256d133f8a7d6a40bc9b0af60410b8a4183c4717707faeb854ed4f6ca5d812a3cc9 was downloaded and independently verified for CRC/all33file hashes, contiguous full output coverage, repeat and cross-storage hash equality, retained first-chunk raw hashes, input hashes and both terminal exit0 ledgers. Complete later-state raw outputs are represented by producer hashes, not independently re-executed GPU results. Structured offline verification retained in outputs/shared-program-equivalence-verified.json.

Only after both probes terminated and archive creation succeeded, installed source383f3e085 to the same fixed study root. Installer verified6476 prior files and28unchanged native/CMake sources plus preserved/current DLL identities; reused tested a959 library. Post-install verifier checked6482files/189tracked updates with zero mismatches and removed the transfer ZIP. Original full arithmetic archive retains old b60 manifests and source before replacement. No production checkout changed.

Started native-mtp3-shared-server and native-mtp3-shared-validation with separate append-only ledgers. Qualification explicitly requires actual target4/prefill128/shared=true/MTPdepth3/draft4/direct vision health, then four VLM semantic cases,16full128-ID self-target comparisons and real SSE/count gates. The existing driver admits its B1 HTTP pilot only after every gate succeeds. At this entry both jobs are live, no shared MTP result or new timing is claimed. Reference tuning/MTP, broad quality/cache precision, long context/concurrency, native batching, INT4/4B and repeated measurements remain outstanding;2x remains unmet.

## P120 - 2026-10-24: preserved historical rejection, strict shared goldens, first pilot

Historical validation exited1 before timing: bottom-blue was semantically correct but new shared output IDs10025/248046 (Blue) differ from oldT16 IDs6105/248046 (blue). All four QA semantic checks passed. The old gate was not edited or silently waived. Diagnostic producer188cf41db then completed20exact same-profile target/MTP pairs, including16full128-token text/image pairs. Bottom-blue also exactly matches previously retained shared-profile smoke output. New separately committed driverab156e640 pins the original shared smoke report SHA256634e6200658aa74467b575c8d4b5e8dde58bb7522d4de1e5ca813da7576acf4b, verifies all six retained profile goldens, exact requests,20target/MTP pairs, explicit health fields and real SSE IDs/counts before its pilot. This is an explicit versioned-profile admission policy, not cross-T16 bit equivalence or broad model quality. Both new producers ran as hashed root tools without modifying live frozen383f3e085 Garnet source.

Pilot123b80fa6f764ae8969ea232ec2f4bde completed0: B1,4096capacity, short41/104-token prompts,128outputs, eight measured requests/cohort and two warmup waves, same corrected JPEG/image-first/binary-grid seed3 requests and literal IPv4 transport as references. Text: TTFT71.27175ms, decode69.96160017tok/s, aggregate67.10492122tok/s, sampled adapter peak28495437824B. Image:105.07610ms,71.88701571tok/s,66.14653356tok/s,28499632128B. Versus best supplied ordinary vLLM/SGLang metric, provisional decode ratios1.02160754/1.04948276, aggregate1.05473388/1.04635376 and TTFT0.46778165/0.67191231. Versus old nativeMTP3, text decode regressed to0.97013x, image1.00322x; TTFT0.64122x/0.38926x. Text half-TTFT holds for this ordinary-reference short-prompt single trial only.2xdecode/concurrency remains unmet; fastest tuned references and broad quality are unproven.

Identity-guarded study bridge/task tree intentionally stopped only after successful terminal pilot; server supervisor exit1 is recorded intentional termination, separate from historical gate exit1. Archive native-mtp3-shared-evidence.zip529453bytes/199entries SHA256282f879d3e70772a15c4e03099676f6bdc32827bfaa5411ca1147fdb3ef9d03b retains both failures and successful diagnostic/pilot, current source manifests, producers, raw40nonstream responses/requests,16measured SSE plus two gate streams, raw NVML samples and stop identity. Offline verifier checked CRC/allfileSHA, exact producer bytes against committed versions,20pairs/sixgoldens, full streamed IDs, all raw SSE metrics and both sampled-memory maxima. Request bytes/counts/transport match both reference archives; ratios remain provisional. Structured outputs/native-mtp3-shared-verified.json and summary retained. New PNG/SVG/data/source chart visually checked; it shows the TTFT gain and unmet decode target without plotting excluded localhost reference.

All92CPU study tests pass from repository root. One initial discovery attempt from the chat directory failed to locate the study package, then corrected. FullINT4 repeat/VLM probe c59949491 is prepared and syntax checked, not GPU-qualified at this entry. Next optimize measured decode and native batching, complete INT4/4B and long-context/reference/quality/repetition scope. Goal remains active.

## P121 - 2026-10-24: full INT4 correctness job started

After the shared pilot archive and local raw verification completed, confirmed only39MiB adapter use/32149MiB free and launched int4-full-qualification under run_logged plus resource monitor. Root probe bytes correspond to committedc59949491; live application remains frozen383f3e085 and native a959. Producer requires all496packed G128 projections, BF16 head, repeat IDs/normalized vectors and four VLM semantics; mixed NVFP4 IDs are diagnostic because checkpoint quantization differs. Construction/correctness memory is sampled separately from future serving metrics. Original job handle is confirmed running; no INT4 full-model success or benchmark result is claimed. Fixed study/model directories reused and no live source synchronization performed.

## P122 - 2026-10-24: accepted-head reuse candidate and real stage-trace preparation

Replayed counters from verified shared pilot archive, using untimed strict-greedy diagnostic responses. Per128outputs, text/image average44.25/43.625target verifications,23.625/23.625prefix replays,67.875/67.25target head evaluations and zero head reuse. Draft acceptance62.97%/64.77%;2.893/2.934emitted tokens per target verification. These counts establish redundant work candidates, not GPU latency attribution. Reproducible outputs/shared-mtp-work-counters.py checks the input archive SHA and writes the structured counter report.

Committedbba459c00 allows existing default-off reuse_verified_head with graph-prefix replay, without requiring native prefix commit/register128 arithmetic. Accepted hidden rows retain authoritative predictions; masked normalization is still refreshed, active zero-prefix rows recompute, missing caches reject/rollback, and full-prefix behavior is unchanged. Five CPU control tests pass for accepted prefixes, active-zero/inactive-empty rows, cache failure rollback and invalid/full commits. No GPU numerical or performance qualification of this candidate yet; no production default enabled.

Committed6d75c7c54 prepares probe_shared_head_reuse.py: before tracing, real full64 shared T4/T128 MTP3 with graph replay compares reuse off/on for20retained ordinary-target sequences, requires exact IDs and unchanged proposed/accepted/verification/replay work, checks head-call reduction, and keeps QA semantics. Afterwards a warmed ABBA device-wide-synchronized stage wall trace separates draft/target verification/prefix commit/correction. Its timing includes CPU/GPU/synchronization and altered scheduling; it is not HTTP throughput or isolated kernel duration. Probe syntax check passes, actual execution pending. INT4 original process21180/job int4-full-qualification remains live with increasing CPU time; no restart or live source replacement. Goal/2x target unchanged.

## P123 - 2026-10-24: complete-graph INT4 candidate prepared while original builds

Original INT4 job remains running, not restarted: observed process21180 CPU increased71.19to96.13seconds, adapter use9038to11608MiB. Report remains constructing_decoder. These observations show live activity, not inference success or a latency measurement. Terminal archive helper is staged separately and requires finished ledger/resource records before retaining success or failure; no source replacement occurred.

Committedcfd37e617 adds a default-off create_cuda_int4_packed_decoder alternative built from existing whole-program code/native INT4 kernel. B1/cap32..4096 and static shared chunk4/8/16, explicit shard index, exactly496native G128 projections, no NVFP4/FP8/externalBF16 fallback and BF16 head are required. Existing per-layer factory is unchanged. Prepared probe_int4_packed_full.py requires successful prior full per-layer INT4 evidence, exact requests and all12generated-ID/normalized-vector matches plus repeats/VLM checks before qualification. This addresses complete-model launch/layout overhead as a candidate; no build-time, numerical or performance gain is claimed before actual GPU execution.

All98CPU study tests pass, including invalid packed profile/index admission and verified-head cache transaction tests. Both optimization candidates and full probes remain unqualified on GPU. Next preserve the original terminal INT4 evidence, install a fully hashed Python-only update only after GPU ownership is released, then run the prepared head-reuse and complete-graph INT4 qualifications. Main2x/native-concurrency/reference/quality/4B/long-context/repetition scope remains unchanged.

## P124 - 2026-10-24: full INT4 smoke passed and installed reference shape audit

Original int4-full-qualification completed child/supervisor exit0, without restart. All64per-layer engines plus final norm constructed, with496native G128 projections and BF16 head asserted by the pinned factory. Twelve runs passed: text/image128outputs twice and four VLM color/spatial questions twice, with exact repeated generated IDs and5120word normalized vectors. All65decoder graph capture failure counters are0. This establishes full-model repeated smoke only; no broad quality, cross-quantized equivalence, actual4Kprompt or HTTP performance claim.

Sampled adapter maximum during construction plus correctness22220951552bytes, not serving/process/instantaneous peak. Archive int4-full-evidence.zip881110bytes/34entries SHA2563f208d652e3bd478ac7ff1e41be9127c700bc1a38f3dbe0ffabc4b7709f17e13 independently verified CRC/all33fileSHA, producer bytes againstc59949491, all12raw normalized vectors/repeated IDs/semantics,496counter and raw memory maximum plus terminal ledgers. Direct IO head-precision dump is not in this producer; successful factory assertion and archived source support the BF16-head contract. outputs/int4-full-verified.json and full report retained.

During the live native correctness run, separately read installed Linux package sources using systemPython without importing Torch/CUDA or loading a second model. Root source audit completed0, reporting vLLM0.32.0/SGLang0.5.22; optional systemd-root-session warning did not prevent execution. Producer committedb497dba27. Archive reference-int4-installed-source-evidence.zip312811bytes/67source files plus config/index/report/producer, SHA25603c3f876b65f32232f2ba3dbee0c7f05ab4022b805547e0843ed8dbff05d0793 verified CRC/all71file hashes and producer bytes.

Pinned48value heads imply merged BA96x5120. Executed the archived SGLang pure verify_marlin_supports_shape function via AST, with its actual64/128tile constants and no package imports: N48/96 reject, N128/17408 pass atK5120/G128. Installed WNA16 scheme calls a boolean-returning checker but does not inspect its result at that call. vLLM Marlin path explicitly permits tile padding, documented arithmetic maps96x5120 to128x5120. These are specific source findings and a compatibility risk for actual startup, not proof of selected kernels or full-engine support/failure. Reproducible outputs/verify-reference-int4-source.py and structured source verification retained. Actual INT4 reference model loading/quality/benchmarks still required. Next deploy hashed Python-only candidates after archive verification, then qualify head reuse and packed INT4.

## P125 - 2026-10-24: verified Python-only deployment and live head reuse probe

After original INT4 terminal success/archive/raw verification, observed39MiB adapter use/32149MiB free. Complete local package6a0268dd1 covers198tracked updates/204patch files and all28native/CMake hashes match tested library provenance. ZIP584018bytes SHA256d1172ded8387212479d118e14ccb3d784b4d6b404ac4f366bad2caf01c3b16e0. Installer verified6482old source files and original/current DLL identities, installed the source-only update with unchanged native28files; post-install verifier checked6490files/zero mismatches and removed transfer ZIP. Tested a959 DLL reused; prior INT4 source/manifests preserved in its archive.

Started original handle shared-head-reuse-qualification under command ledger/resource supervisor. Hashed root producer6d75c7c54 loads shared T4/T128 MTP3 with graph replay and explicit head-cache opt-in, then requires20reuse-off/on pairs equal retained ordinary target sequences and unchanged accepted/verification/replay work before synchronized ABBA stage traces. Still constructing at this entry; no GPU head-reuse correctness, stage duration or speedup is claimed. No other GPU model or live source synchronization occurs. Packed INT4, reference INT4 startup,4B, broader quality/cache precision/long-context/native concurrency/repetition and overall2x target remain outstanding.

## P126 - 2026-10-24: head reuse correctness passed, instrumentation failure retained

Original shared-head-reuse-qualification completed exit1: all40off/on runs (20pairs,16full128text/image and four VLM cases) first passed exact retained ordinary-target IDs and unchanged proposed/accepted/verification/replay counts. The subsequent ctypes GetModuleFileNameW call raised NotImplementedError(libffi is unavailable for ctypes) before any stage trace. Overall job is failed; do not describe it as successful profiling or invent missing durations. Decoder/session finally closed, parent resource child_exit_code1 retained.

Archive shared-head-reuse-instrumentation-failure-evidence.zip535192bytes/60entries SHA2563de02768505a6b0efe3b015e150f6b015ebe1a96e625a21fa6bd8341f14332ed independently checked CRC/all59file hashes, root producer bytes against6d75c7c54, all20ordinary target responses/request identities, paired generated IDs, semantic answers and invariant counters plus raw memory maximum. Target head computations across eight text/image requests decreased543to354/538to349, with189cache reuses each. These are work reductions, not latency/speedup results. Structured partial-result verification explicitly marks overall failed and no traces.

Committedf13b8bafe replaces foreign-call instrumentation with existing native finite_f32_bytes on a retained first-word view of last_hidden. Native code at cuda_tensor_io.cpp166..183 reads asynchronously then synchronizes CurrentExecutionStream; diagnostic barrier copies4bytes, not device-wide synchronization. Trace wall duration explicitly includes CPU/GPU/readback/sync overhead and altered scheduling; never HTTP/model throughput or isolated kernel duration. No global Python/runtime repair or native DLL rebuild.

Committedbb5f02f10 preflight ran on actual5090/embedded runtime under a separate ledger: four updates0,1,-0.25,2.5 to a5120word GPU tensor returned exact4-byte F32 first-word observations via retained1x1view; exited0 and actual DLL SHA a959 checked. Downloaded/checked raw report outputs/native-trace-barrier.json. Only after actual original terminal failure/archive and preflight success, launched new distinct shared-head-reuse-trace-qualification handle with corrected root producer and separate outputfolder; original attempt preserved. Frozen application6a0268dd1/a959 unchanged, no live source sync. New job is running; no stage timing or new HTTP measurement yet. Overall2x goal remains unmet.

## P127 - 2026-10-24: corrected stage trace passed; HTTP candidate prepared

shared-head-reuse-trace-qualification completed child/supervisor exit0 on frozen6a0268dd1/a959. Again40full-model off/on runs matched20retained ordinary-target sequences, including16full128pairs/fourVLM cases and unchanged draft/verification/commit counts. All eight warmed ABBA stage traces completed with the native4-byte stream observation. Archive shared-head-reuse-trace-evidence.zip561188bytes/64entries SHA256ef589c9ed24e1582696269cec226bdff6c9c05e6395f71daa9cfd947465ddb6b independently verified CRC/all63fileSHA, producer againstf13b8bafe,40retained raw ID/request checks, stage occurrence counts/finite durations, resource sample maximum and terminal exit0. Tracing requests assert teacher equality in producer but do not retain their separate raw IDs; that verification limit is explicit in outputs/shared-head-reuse-trace-verified.json.

Instrumented mean stage wall ms per request, two traces/mode/case: text off/on draft188.72/183.74, target_verify1454.11/1452.18, prefix_commit140.15/109.49, draft_correct116.25/111.13; image163.94/165.73,1330.44/1330.33,103.91/82.10,102.57/104.79 respectively. These include native readback/CPU/GPU/sync and omit unwrapped work; they are neither kernel timings nor HTTP speed. Target verification dominates the measured four-stage total, while head reuse only modestly reduces prefix commit. Next optimize verification with finer graph/state/head breakdown, rather than infer a2x model gain from35%head-call reduction.

Committed6b7b82234 exposes an explicit shared-MTP3-only head-cache worker profile, actual verifier property in readiness, independent launcher and explicit reuse=0 pins in every legacy launcher. Diagnostic/pilot supports a separate artifact folder and requires actual head-ready flag. All100CPU study tests passed for this integration. Follow-up041b4d978 requires positive cache exercise in all16long requests: typed integer counters, head_reuses==prefix_replays>0, head_evaluations==target_verification_calls, in addition to unchanged strict IDs/goldens/SSE. Twelve targeted gate/profile tests pass after the added counter gate. No new HTTP speed result exists yet. Source-only deployment/strict HTTP gates/pilot follow after archived trace verification and GPU release; overall2x remains unmet.

## P128 - 2026-10-24: head-reuse HTTP source deployed and original workflow live

After corrected trace termination/archive/raw verification and observed39MiB adapter use/32149MiB free, installed fully checked source74d693a2b. Package588878bytes SHA25662859d9d748a0eedd3b941d05a36b81af8d4f98bf155aedadcab4ec3cff89b66 covers200tracked updates/206patchfiles;28native/CMake hashes unchanged versus tested library. Installer verified6490old files plus current/preserved DLL identities; post-install verified6492files/zero mismatches and removed transfer ZIP. a959 native binary reused, no production checkout replacement.

Started native-mtp3-shared-headreuse-server with its own identity/source ledger and new explicitly pinned headreuse launcher. Committed root workflow80487cc9b waits on that original worker without restart, verifies actual shared T4/T128/depth3/draft4/direct/headreuse=true readiness, runs a separate20pair HTTP diagnostic, then pinned six shared-profile goldens, exercised-head counters and real SSE before a B1 matched pilot. Artifacts/results use fixed study root with distinct variant result names. Native server and workflow handles are confirmed live at this entry, waiting for readiness; no HTTP gate or head-reuse speed result is claimed yet. No live source synchronization while they run. Goal and all remaining precision/quality, tuned-reference, long-context/native-concurrency, INT4/4B and repetition requirements persist.

## P129 - 2026-10-24: head reuse HTTP negative result verified; finer verification trace live

Workflow completed0 after20exact HTTP target/MTP pairs, six pinned retained profile goldens, positive head exercise counters and real SSE gates. Pilote6ede03f9e61404da1743da8626755c4, B1/4096capacity/41text or104image prompt/128output, eight measured requests per cohort/two warmup waves, corrected seed3JPEG image-first binary-grid/IPv4. Text TTFT82.74375ms, decode67.64584977tok/s, aggregate64.74378428tok/s, sampled adapter serving maximum28489146368B. Image112.51195ms,69.55685934tok/s,63.80380649tok/s,28499632128B. Against prior shared head-off pilot, decode0.96690x/0.96759x and TTFT1.16096x/1.07077x. Against best supplied ordinary reference, decode0.98779x/1.01546x, aggregate1.01762x/1.00929x, TTFT0.54308x/0.71946x. This pilot demonstrates no speed benefit; separate single runs cannot attribute the slowdown causally to cache logic. Defaults and legacy launchers remain head-off; no half-TTFT/2x claim for this candidate.

Identity-guarded bridge/tree stopped only after terminal workflow0; server supervisor exit1 is intentional and recorded. Archive native-mtp3-shared-headreuse-evidence.zip528435bytes/188entries SHA25625f97dd8bbaa16459558f342aec9a9e82e5b6e703bb535bbd073ca63c91468a0 independently verified CRC/all187fileSHA, exact producer source versions,20pairs/sixgoldens, all16measured raw SSE metrics/fullIDs versus retained ordinary target, exercised-head counters in every timed request, both gate-stream fullIDs and raw sampled maxima. Actual cache=true readiness recorded. Request bytes/counts/IPv4 match head-off and both references; precision/quality/tuning/repetition limits remain explicit. Outputs/native-mtp3-shared-headreuse-verified.json and updated PNG/SVG/data/source negative-result chart retained and visually checked. Initial plot-copy attempt used host defaultGBK and failed to decode UTF8; corrected explicitUTF8, no data changed.

Target verification was the dominant instrumented stage in P127; do not chase head-count reduction as a model-speed proxy. Committed69b4986b7 prepares probe_shared_verify_breakdown.py: same40raw correctness runs first, then nested decode-only state-backup, chunk, whole-program, normalization and vocabulary-head wall intervals in addition to parent stages. Native graph wall includes host bindings and GPU work. Parent/child intervals overlap and must never be summed; four-byte diagnostic synchronization overhead remains explicit. Delegates forward actual native operations unchanged; trace requests now retain their full generated IDs/request/teacher hashes. Syntax check passed.

After archive verification and observed39MiB adapter use/32149MiB free, started shared-verify-breakdown-qualification under source/binary ledger and memory supervisor. Application remains frozen74d693a2b/a959, root producer69b4986b7 is hashed separately; no live source sync or competing GPU model. Original new handle is confirmed running/constructing, no finer-stage result claimed. Packed INT4 full comparison, INT4/4B matched measurements, native batching/long-context, reference MTP/tuning, broader precision/quality and repeated trials remain required; overall2x goal remains active and unmet.


### P130 — independently verified nested verification trace; profiler preflight failure retained

Verified shared-verify-breakdown-evidence.zip locally: CRC, all 63 manifest hashes, producer against 69b4986b7, 20 exact off/on correctness pairs, all eight retained trace token sequences and request/teacher hashes, interval nesting and count invariants, sampled memory peak and successful terminal ledger. Archive SHA256: 22680f317f4dc0670440099c7972deea48de64ce2e0402b52598850683ca6f89. Reproducible offline verifier and structured output are in the chat outputs as verify-verify-breakdown.py and shared-verify-breakdown-verified.json.

Head-OFF mean synchronized verification wall per request: text 1505.32 ms, of which target_program 1281.36 ms; image 1378.75 ms, of which target_program 1175.50 ms. These nested durations overlap; do not sum them. Program duration includes host bindings and GPU execution. Two traces per cohort, diagnostic synchronization overhead, no throughput or isolated kernel claim. This supports examining the complete target program before further head-cache work; it does not identify a particular slow kernel.

Original native-trt-layer-profiler-preflight terminated exit 1: dynamically setting Python os.environ did not produce the native profiler TSV. The failed report is retained; no model benchmark was run. Added explicit startup-profile mode to distinguish inherited native environment from dynamic environment propagation. Launched a separately named bounded normalization preflight using the inherited flag, leaving the original failure intact. Existing hook intentionally bypasses CUDA Graph on its sampled enqueue; any emitted layer timing is an eager diagnostic, not captured replay or HTTP performance. Remote application remains frozen at 74d693a2b with a959 DLL. No 2x target achieved.

Startup preflight also terminated exit 1 before graph construction: the admission check compared a forward-slash environment path to a Windows backslash string. Corrected the check to compare Path objects. This is a probe bug, not evidence about TensorRT profiling support. No replacement run has yet been performed.


### P131 — inherited profiling emits timing but reveals callback lifecycle defect

The original native-trt-layer-profiler-startup2-preflight handle completed exit 1. Inherited environment successfully produced layers.tsv (one RMSNorm layer, 0.314016 ms first-use eager diagnostic). Output byte comparisons completed before the graph capture assertion failed. Runtime printed TensorRT API Usage Error rejecting setProfiler(nullptr). Existing hook leaves a stack callback attached and its getProfiler()==nullptr guard prevents subsequent graph capture. Do not run the full-model profiler using the old binary. This does not affect the unprofiled measured production paths; all earlier benchmark processes had this profiling flag unset.

Prepared a native fix: context-pool slot owns the diagnostic callback through context destruction; never detach with nullptr; explicit active-sample state controls graph bypass. After sampling, enqueue emission is disabled and graph replay may resume with the retained callback. NVIDIA documentation states a retained profiler can still record events when emission is disabled: https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/classnvinfer1_1_1_i_execution_context.html . Therefore the diagnostic process must never be used for official performance benchmarks, even after this fix. Native changes are not compiled or GPU qualified yet; remote application/DLL remain frozen at 74d693a2b/a959. Full-model probe prepared with twelve retained teacher comparisons, first-use eager scope, head reuse OFF; not run. Next action is build/qualify callback lifetime and captured replay on the bounded preflight before the full probe. No 2x claim.


### P132 — source deployed with rollback binary; native build and dependent qualification active

Installed 6f943569f after verifying all 6492 prior source files and absence of study xlang3 processes. Preserved a959 DLL and prior source/manifests in build/qualified-before-profiler-lifecycle. GPU idle observation: 39 MiB used /32149 MiB free. Other CantorInstall processes were identified and left running because they are outside the study and GPU memory was sufficient. Full CPU discovery: 101 tests passed. This does not qualify the new native binary.

Build launch failure 1: artifact argument referenced an unextracted JSON. Failure 2: batch path did not resolve. Neither compiled code; corrected artifact to uploaded source ZIP and installed an explicit batch using the known remote BuildTools environment. Original live handle profiler-lifecycle-build-fixed-launcher compiled trt_builder.cpp and linked the native library, then continued post-build dependency staging. Do not restart on an observation timeout. Dependent standalone workflow profiler-lifecycle-build-then-qualification on SSH helper 95279 waits for the original build ledger's successful terminal event, then runs the bounded startup-profile preflight once with retained command/resource logs and archives on success. Full-model layer diagnostic has not run.

Downloaded profiler-preflight-failures-evidence.zip, 25066 bytes, SHA256 fcf5065e2637175f6a1bf5340d9524493dc6f90345390b0eacb084b8cf39679b. Independently verified CRC, all 17 manifest file hashes and three failed reports. Archive and verified JSON retained in chat outputs. No new model throughput or 2x result.


### P133 — diagnostic chart and guarded full-model follow-on

Revalidated original build and dependent qualification handles: both live; build remains in CMake vendor copy_if_different post-build stage. Observed child cmake PID 13928 had read 3662689024 bytes, zero writes, increasing CPU counters; no restart or second build was initiated. Do not equate this slow dependency comparison with GPU/model execution. Standalone full-model follow-on handle shared-trt-layer-diagnostic-after-qualification on helper 95279 now waits for the bounded qualification's successful terminal ledger, verifies passed report and exact current DLL hash, then invokes the twelve retained teacher comparisons and first-use eager profiler once. Failure exits before full model; no benchmark process is scheduled. Standalone workflow and archive method sources committed for review; deployed application source stays at 6f943569f.

Generated and visually inspected outputs/Garnet-Verification-Breakdown.png/.svg with reproducible plot source and JSON derived from independently verified nested traces. Head-OFF target-program shares: text 85.1%, image 85.3% of synchronized verification wall. Subtracted nested target-program from target-chunk and subtracted immediate children from verification root to form disjoint chart segments. Clear diagnostic host+GPU/synchronization/two-request scope retained. No new performance evidence and no 2x claim.


### P134 — native build and bounded GPU profiler lifecycle passed

Original build handle profiler-lifecycle-build-fixed-launcher completed exit 0; dependent profiler-lifecycle-build-then-qualification completed exit 0. New garnet.dll SHA256 9d82a3dc80e1817a50931ef130e2a6b3023248514687eddfd8f2493ee0dbb0af, native source 6f943569f. Bounded real-checkpoint RMSNorm preflight passed producer byte equality across five calls, one native TSV row, one successful graph capture and two graph replays, zero capture failures. Old a959 binary remains preserved. These checks qualify the instrumentation lifecycle only, not full model correctness or performance. Raw output bytes were not archived, so offline verification confirms source assertions and output digest, not an independent raw-output equivalence check.

Downloaded profiler-lifecycle-qualified-evidence.zip: 2836469 bytes, SHA256 579e0fcce28d808dcc29c9931998268b0559f1d5507aef9c4d82eb3a6ca6d6df. Offline verifier checks CRC/all19 manifest file hashes; archived native/probe source bytes exactly equal committed 6f943569f; DLL hash; successful build/correctness terminal ledgers; TSV/report equality; resource sampled maximum and graph counters. Reproducible verifier and verified JSON retained in chat outputs.

The original shared-trt-layer-diagnostic-after-qualification handle is now running the full shared model diagnostic after its guard succeeded. Do not synchronize app source, rebuild or start another GPU job until this original handle terminates. No full-model outcome yet. No new throughput, TTFT or 2x evidence.


### P135 — full first-use profile verified; INT4 packed qualification running

Original shared-trt-layer-diagnostic-after-qualification completed exit 0. Twelve retained ordinary-teacher comparisons passed (six text/image/VLM cases repeated twice), all raw full token lists retained. Native source remains 6f943569f, DLL 9d82a3dc80e1817a50931ef130e2a6b3023248514687eddfd8f2493ee0dbb0af. Diagnostic archive 630029 bytes, SHA256 157870f571ff28bd50b43692264fee1637acc59a6fda716133d91d741f45c1e2. Offline verifier checked CRC/all32 manifest hashes, exact committed producer, all12 teacher token lists and request/teacher hashes, all5105 TSV rows across8 engines, terminal/resource/capture-failure checks. No broad quality or benchmark admission claimed.

First-use layer timing has obvious initialization effects (e.g. one unnamed prefill plugin sample146.827 ms); generic plugin rows do not identify kernel families. Do not rank hot captured kernels from these cold eager samples. A proper warm diagnostic trigger or external profiler remains needed before causal kernel claims.

Guarded int4-packed-after-layer-diagnostic archived the completed full profile, then launched the already prepared whole64 INT4 T16/cap4096 versus retained per-layer golden on the same DLL. Its original handle is running. It checks exact full IDs, normalized vector hashes, repeats and four VLM cases before any timing. Never launch a second GPU job or synchronize application source while it is live.

Read-only CPU calibration audit completed exit0 with no GPU imports: all64 MLP gate/up pairs have bit-identical input scales AND weight_scale_2; all48 GDN QKV/Z and16 full-attention Q/K/V groups have identical input scales but differing weight scales. This supports investigating gate/up concatenation with preserved calibration, not proving a fused implementation, numerical equivalence or speedup. Raw scalar/header evidence archive preparation is in progress. Audit, archivers and guarded INT4 workflow sources retained for review. No new TTFT, decode or concurrency results; 2x remains unmet.


### P136 — independent calibration evidence verified and concrete joined-MLP plan

Downloaded shared-activation-scale-audit-evidence.zip,143531 bytes,SHA256 1d0b024eb2bb679ba798284e2255fedead12c9ae6965c3791146eefc97a92ea7. Offline verifier checked CRC/all554 manifest file hashes, committed audit producer, index/header hashes and scalar offsets, all544 raw scalar byte values and all128 group classifications, terminal audit exit0. Gate/up common-scale eligibility is therefore supported by actual retained checkpoint bytes, not previous chat claims. The implementation plan now describes explicit opt-in joined pool/lowering,336 expected mixed matrices, memory ownership, exact correctness checks and matched end-to-end validation. No joined implementation or performance gain exists yet. Original int4-packed-after-layer-diagnostic handle remains running; application source and DLL left unchanged.


### P137 — packed INT4 independently qualified; joined-MLP component implemented locally

Original int4-packed-after-layer-diagnostic completed exit0. Archived int4-packed-evidence.zip,997333 bytes,SHA256 de4d2398d22753e9eb1352241ca23e01ebdb5139d2ab3918768602a667a323d7. Offline verification: CRC/all53 manifest file hashes, producer matches6f943569f, all12 full token sequences and raw5120-word normalized vectors exactly match archived same-profile per-layer INT4 goldens; request/response hashes,496 actual native INT4 projections,2 graphs,zero capture failures,terminal0 and raw sampled memory maximum verified. Sampled construction/correctness adapter peak21792940032 bytes; not serving peak. No HTTP/performance or broad-quality claim. Full oldperlayer and newpacked raw normalized vectors retained. Remote application/DLL remain6f943569f/9d82; no GPU study worker remains live after terminal handle.

Implemented local opt-in joined NVFP4 MLP component: strict shared metadata helper admits only canonical text-layer gate_up aliases with matching128-aligned packed geometry, block16 scales and identical positive finite FP32 calibration. Pool loads concatenated gate/up bytes once into readonly INT32 storage, repacks scales at joined row offsets, retains allocation lifetime and rejects duplicate resident originals in either load order. External binder validates both pinned source tensors and joined buffer shapes; explicit qwen35_joined_mlp lowers one existing NVFP4 projection followed by ordered gate/up slices, original SiLU/product and down projection. Existing complete decoder still uses the original MLP; no full-model joined option yet.

Local MSVC compiled all3 affected native translation units successfully without GPU calls. Actual SafeTensorsIndex + shared helper CPU test passed a valid fixture,14 rejection fixtures and5 invalid names, both direct MSVC and registered CMake target build/run. Existing Python regression101 passed. Component pool-byte/lifetime and T4/T128 numerical probes prepared and syntax-checked, not run on GPU. Current-source/local-validation hashes retained in chat outputs/joined-mlp-local-validation.json. Candidate source is not synchronized or deployed remotely; its GPU byte, numerical and performance gates remain unproven. Next: build isolated candidate preserving9d82, run raw pool oracle then component exact-output gate before full integration; also obtain packed INT4 matched HTTP baselines. 2x remains unmet.


### P138 — joined pool and MLP GPU components passed and independently verified

Deployed source6012962cc after verifying6496 prior source files, no live study xlang3 worker, and exact current9d82 DLL. Preserved9d82 plus prior manifests/affected sources under build/qualified-before-joined-mlp. Native candidate build joined-mlp-component-build completed exit0. New DLL SHA256 e194bcc2cfaf923676de9dfd44ae99f8bb8fa508e00b4746d8f4639883a00209. Managed build now explicitly disables repeated vendor DLL staging; default option remains ON for ordinary distributable builds. Local generated-command tests OFF/ON both passed; SDK discovery/installation logic remains intact. This accelerates build iteration, not model inference.

Original joined-pool-then-mlp-qualification completed exit0. Pool producer verifies combined34816x5120 packed bytes and independently tiled scales against the two pinned originals, readonly rejection, no duplicate resident originals, exact accounting/reuse and retained lifetime after pool close. These GPU raw arrays are not duplicated into the archive; retain source assertions/hashes as this scope of evidence. Actual layer0 MLP on deterministic finite inputs matched the original complete F32 outputs for T4 andT128,three repeats each plus interleaved execution. Actual NVFP4 matrix count per MLP changed3→2, same physical weight/scale byte count, two profiles share one two-matrix pool,zero graph capture failures. No full decoder or real hidden-activation coverage yet.

Archived joined-components-evidence.zip,7941854 bytes,SHA256 f8246c54350998901ccfab3ed2312b46a9196f117e5e652a1d2d64811ce06921. Offline verification checks CRC/all31 file hashes, all21 patch source files exactly match6012962cc, binary identity, successful build/two correctness ledgers and sampled maxima, six full raw candidate F32 vectors equal the retained original vectors, actual graph/pool counters. Verifier and structured report retained in outputs. Source remains6012962cc/e194 on remote, complete decoder still uses original MLP. No study GPU worker remains live after its terminal handle. Next action is explicit full-program joined option with336 actual mixed matrices, including synthetic alias metadata handling before weight_info, both linear/full-attention MLP lowering and full same-profile token/state gates before HTTP benchmarking. No speedup or2x claim.


### P139 — full joined-MLP option implemented locally, GPU gate prepared

Added explicit default-false joined_mlp_projections through shared factory, packed decoder/program, Python chunk lowering and native linear/full-attention chunk MLPs. The option is admitted only for B1,cap32..4096,T4/T128,shared compressed weights, excluding embedded or BF16 joined-input modes. Every layer's synthetic alias is validated before projection allocation; alias metadata bypasses ordinary checkpoint weight_info and retains both original names. Native lowering requires explicit validated external alias bindings. MTP's own decoder retains its original MLP path. Factory validates actual128 NVFP4+208 FP8=336 matrices and exact physical byte accounting, versus192+208=400 original; health memory profiles retain the actual option.

Local compile-only native targets completed exit0. Full Python regression103 tests passed, including real pure admission rejection and factory forwarding/count/BF16/byte mismatch cleanup tests. No GPU execution of this full option yet. Prepared separate-process original-versus-joined T128 full packed output/state hash comparisons on identical saved text/image inputs, plus twelve complete-generation token/prefill-normalized comparisons against the pinned original shared-profile golden SHA634e6200658aa74467b575c8d4b5e8dde58bb7522d4de1e5ca813da7576acf4b. These remain pre-performance correctness gates, not broad quality or long-context coverage.

Remote remains6012962cc/e194 with component qualification only. No live study GPU job at this point; complete joined decoder still requires deployment/build and actual GPU gates. No new performance result;2x unmet.


### P140 — full joined candidate deployed and GPU gates started

Installed846b1e12f after verifying6508 prior source files and exact e194 component DLL, with no live study xlang3 worker. Preserved e194 and prior manifests/affected source under build/qualified-before-joined-decoder. Original joined-full-decoder-build completed exit0, retaining build ledger and source ZIP hashes; managed vendor dependency reuse remains explicitly OFF staging. Original workflow joined-full-state-then-generation on SSH helper95279 is running: original shared T128 state/output hash baseline, then joined T128 hash comparison on identical saved inputs, then joined T128/T4 generation versus the pinned original shared-profile teacher. Each step requires the preceding original process to finish successfully, records actual matrix counts and the common DLL hash, and halts before later steps on failure. No model benchmark or second GPU worker launched.

Do not synchronize source or rebuild while this original workflow is live. Actual complete-model joined output/state/count correctness remains unproven until these processes finish and archived raw/hash evidence is independently checked. Archive helper and workflow sources retained for review. Remote source frozen846b1e12f; local subsequent changes only retain methods/logs. No new performance evidence;2x unmet.


### P141 — full joined producer gates passed; HTTP/MTP admission prepared locally

Original joined-full-state-then-generation completed exit0. Original and joined T128 programs each completed text/image x3 repeats with167 packed-output hash chunks per run; producer equality comparison passed for all complete hidden/state output chunks. Full joined T128/T4 decoder then matched all12 complete generated-token sequences and prefill normalized-vector hashes against the pinned original shared-profile golden, including all4 VLM QA cases/repeats. Guarded workflow verified actual128 NVFP4+208 FP8=336 matrix counts. Archive-joined-full-evidence handle on helper48577 is preparing the raw/hash archive; independent offline verification remains pending. No performance claim yet.

Prepared local HTTP/MTP activation: strict GARNET_STUDY_JOINED_MLP only for shared MTP3 with verified-head reuse OFF; all existing MTP launchers explicitly pin0 and a distinct joined launcher pins1. Actual readiness reads decoder option plus native IO pool counts. Validator requires true boolean opt-in and typed336/128/208 counts. Joined pilot additionally requires complete same-binary state/generation reports, exact original-state chunk equality/coverage, original input files and retained complete token/normalized hashes, before the existing20 target/MTP pairs,6 goldens and SSE gates can admit timing. New workflow remains unrun/unpublished remotely; native/application source stays frozen846b1e12f during archive work.

Full local CPU regression107 tests passed, including profile exclusivity, stale/malformed readiness,16 model-gate evidence mutations and original/default behavior. New files syntax-checked. These CPU fixtures do not establish GPU correctness/performance beyond the actual producer runs described above. Next independently verify archive, sync Python-only HTTP changes without changing the qualified native DLL, and run matched B1 pilot. Native batching, tuned reference/INT4/4B benchmarks and broad/long-context requirements remain open.2x unmet.


### P142 — full joined archive independently verified; joined MTP/HTTP pilot started

Downloaded joined-full-evidence.zip,37681702 bytes,SHA256 b3dbb9267baa35a197b28a47ed5a27202782e5dc598650a5dd71fdd2c6adaa98. Offline verifier checks CRC/all81 file hashes,all15 patch source files exactly against846b1e12f,raw DLL SHA256 2d902723b85f4353db3c3feccf2088b2ab412af6deb23b529caa3bb089d7708c,all3 successful terminal/resource ledgers and sampled maxima,complete174587904-word T128 packed output coverage with six original/joined hash cases,12 retained first raw state chunks,and all12 full generated-token/raw-normalized-vector comparisons against pinned original golden. Remaining raw state arrays are producer hashes, not independently retained arrays; nonzero-state T4 byte equivalence,broad/long-context/MTP gates are not implied.

Installed Python-only HTTP source e154b6291 after verifying6515 prior source files and the exact qualified2d9027 DLL. Installer forbids native/CMake changes; source package contains20 Python/launcher/method/log updates. Native build remains846b1e12f and no rebuild occurred. Retained prior source/manifests and current DLL under build/qualified-before-joined-http. Confirmed no listener on18082 before starting.

Original server handle native-mtp3-shared-joined-server is running on helper95279. Original driver native-mtp3-shared-joined-http-workflow is running on helper48577, waiting for actual ready profile before20 self-target/MTP pairs,six pinned goldens,same-binary full-model gates,real SSE and one matched B1 pilot. Do not synchronize source,rebuild,start another GPU job or restart these handles solely on an observation timeout. Success-only cleanup watcher joined-pilot-success-cleanup on helper48577 waits for terminal successful pilot/manifest,then identity-guards the exact study bridge process tree before shutdown. A failed pilot is retained for inspection and does not trigger this success-only cleanup. No benchmark result yet;2x remains unmet.


### P143 — joined MLP matched B1 pilot verified: small observed difference,2x unmet

Original native-mtp3-shared-joined-http-workflow completed exit0;20 exact target/MTP pairs,6 original profile goldens,same-binary full-model gates and2 real SSE gates passed. Run7471e2a5a057487d9389a8139f0b80f1,application e154b6291,native846b1e12f,DLL2d902723b85f4353db3c3feccf2088b2ab412af6deb23b529caa3bb089d7708c. Actual worker joined=true,head reuse=false,336 mixed matrices,prefill128/decode4,MTP3/draft4,direct vision. Per cohort8 measured requests,128 outputs,text41/image104 prompt tokens,IPv4. This is short-context with4096 capacity,not a4096-token prompt.

[
  {
    "case": "text",
    "concurrency": 1,
    "ttft_median_ms": 64.71939999755705,
    "decode_median_tokens_per_second": 72.25899562353953,
    "output_tokens_per_second": 69.18361242908233,
    "sampled_peak_adapter_used_bytes": 28357025792
  },
  {
    "case": "image",
    "concurrency": 1,
    "ttft_median_ms": 103.71515000224463,
    "decode_median_tokens_per_second": 74.1574261696415,
    "output_tokens_per_second": 67.90184904425222,
    "sampled_peak_adapter_used_bytes": 28363317248
  }
]

Offline archive verification completed: native-mtp3-shared-joined-evidence.zip1072905 bytes,SHA256 19bafb0f2cf8f518ad199677ad509c66e903abbecd94eb122b498048a437081f. CRC/all193 manifest file hashes,committed producer sources,full-model archive linkage,20 diagnostic pairs,all16 measured full SSE token lists versus retained ordinary target,head reuse OFF/evaluation counters,2 stream gates,raw memory maxima,terminal workflow0 and identity-guarded successful shutdown checked. Server ledger exit1 is intentional taskkill after terminal pilot,not a benchmark failure. No study server/GPU worker remains live. Native/source remain2d9027/e154 on remote.

Relative to prior separate original-MLP shared pilot: decode ratios1.03283795(text),1.03158304(image);aggregate1.03097673/1.02653677;TTFT ratios0.90806526/0.98704796. Relative to faster supplied ordinary vLLM/SGLang baseline: decode1.05515503/1.08262861;aggregate1.08740609/1.07412061;TTFT0.42477626/0.66320968. Request bytes,transport and counts match. These are separate single pilots on different source revisions;no controlled causal attribution,repeated-confidence or fastest tuned-reference claim. Text TTFT is below half the ordinary reference in this trial;image is not.2x decode/concurrency remains unmet.

Generated and visually checked outputs/Garnet-Joined-MLP-vs-References-B1.png/.svg,with reproducible plot/data and clear2x/half-TTFT lines and scope. Keep joined MLP opt-in. Next quantify same-binary A/B repeat variability and investigate actual cuBLASLt algorithm/workspace/kernel efficiency,with cold-weight component and full-model gates. Consult the installed-version constraints before trying operand/layout changes: NVIDIA cuBLAS13.2.1 documentation https://docs.nvidia.com/cuda/archive/13.2.1/cublas/index.html ; FP4 block-scaled matmul has transpose/scaling/alignment restrictions,so arbitrary layout support or speed must not be assumed. No alternate layout or algorithm benchmark exists yet. INT4 and4B serving/reference comparisons,native batching correctness,full context,broad quality and tuned-reference/MTP baselines remain open.

### P144 — standalone NVFP4 operand-order experiment prepared

Added an isolated CUDA test target; production Garnet matmul and deployment remain unchanged. The proposed TN operand swap pads activation rows to 16 and explicitly transposes the BF16 result back. A structured packed E2M1/E4M3 scale oracle checks every output against an independent analytical sum at batches 1, 4 and 128 before optional projection-shape timing. Timing includes padding computation and transpose, uses captured CUDA event nodes and ABBA order with an eviction kernel outside the timed interval. These are synthetic inputs, not actual checkpoint weights, model tokens, or evidence of a 2x gain.

Local CUDA 13.2/MSVC compile initially rejected obsolete graph-edge API signatures. Corrected the calls to include edge-data arguments; the standalone target then compiled and linked successfully. No local GPU execution. Build log retained in out/qwen38-native/operand-swap-build.log and copied to the chat outputs. Remote read-only check reported RTX 5090, 39 MiB used, 32149 MiB free, 0% utilization. Remote correctness and timing remain pending; the validated 2d902723 runtime has not been replaced.
### P145 — RTX 5090 operand swap: oracle passes, limited/negative timing result

Deployed source 14bae1df7 after verifying all 6521 prior source files and retaining qualified DLL/source manifests. Standalone build, small structured oracle, and projection-dimension oracle/timing jobs all exited 0. Production DLL SHA256 remained 2d902723b85f4353db3c3feccf2088b2ab412af6deb23b529caa3bb089d7708c before build, after build and after GPU execution. No serving runtime changed.

Retained operand-swap-attempt1-evidence.zip (217279 bytes, 55 entries), SHA256 04b99acc4ee7b22ef11b43748e41e8c46c7131ea0664a398dcb9e322426d69e8. Downloaded archive CRC and all 54 content hashes passed. Independent CPU BF16 rounding/sum oracle checked every retained original/swapped output at batches 1/4/128 for N256/K256 and all three actual projection dimensions; all exact. This structured test uses positive E2M1 ones and varying exact E4M3 block scales, not checkpoint weights or general numerical coverage. Raw arrays, timing samples, source package, resource observations and terminal ledgers retained. The first local verifier attempt rejected a manually transcribed checksum containing an extra character; rereading remote and local archive SHA256 resolved the transcription error, and the full verifier then exited 0.

ABBA median original/swapped ratios for batches 1/4/128: N17408 K5120 = 1.0421/1.0000/0.6814; joined N34816 K5120 = 1.0467/0.9684/0.6391; down N5120 K17408 = 0.9973/1.0003/0.9077. Single component trial, captured event intervals, 256 MiB eviction intent, transpose/padding included. No checkpoint/model/HTTP speed claim. No runtime promotion: small B1 gate/up gains, no down gain, and T128 regressions do not justify broad integration. Keep production path and target status unchanged. Remaining work includes matched INT4 and 4B serving/reference measurements, stronger tuned MTP baselines, native batching correctness/concurrency, repeated controlled trials and further backend optimization; 2x remains unproven.
### P146 — INT4 serving candidate and strict gate prepared

Added a separate research-only packed INT4 G128 worker (B1/T16/cap4096, max128 outputs), explicit bridge model route and fixed-folder Windows launcher. Uses the same synchronized native prediction/token pipe and direct BF16 vision as the qualified whole INT4 probe; no MTP or generic ServeModel admission. Readiness requires actual496 INT4 projections. Added a gate against SHA-pinned whole-model INT4 golden (2607a4183044309698c2f5852578b6d6e2825187730e577670ec2efefca6f775): six exact prompt/full-output/text cases twice and two real SSE checks before seed3 JPEG/image-first binary-grid B1 eight-request/128-output measurements. CPU bridge/worker-I/O/position tests:9 passed. GPU worker/HTTP gates and timings pending. Mixed NVFP4 versus INT4 is cross-precision, not a matched-engine speedup claim.
### P147 — INT4 HTTP process started; reference candidates prepared

Installed5194b73b7 after verifying6525 prior source records; retained source and qualified2d902723 DLL. Started original remote handles native-int4-server (helper95279) and native-int4-http-workflow (helper48577), with raw ledgers and fixed result folders native-int4-worker/native-int4-http-pilot. Repeated handle observations confirm both running during construction; no readiness, correctness or performance success inferred from liveness or allocations. Observation timeouts are not terminal and must not cause relaunch. Source is frozen on remote while this worker lives.

Prepared separate pinned INT4 vLLM/SGLang candidate launchers preserving prior4K context, seq8 and memory/cache/dtype policies. Both pass Git Bash syntax checks locally. Actual quantization dispatch, startup (including small GDN split Marlin compatibility), quality and performance are unverified. No concurrent GPU reference launch while native worker lives. No model re-download or precision/checkpoint substitution. Identity-guarded stop and archive scripts prepared in fixed study root; stop requires terminal successful workflow and completed benchmark manifest.
### P148 — INT4 HTTP gates pass; warmed pilot live

Original helper48577 native-int4-http-workflow emitted INT4_HTTP_EXACT_GOLDENS_AND_SSE_PASS and remains running. All12 HTTP golden checks and2 real SSE gates completed before benchmark invocation. Preserve original server95279 handle, currently live; do not sync/rebuild while it runs. No partial pilot summary is admitted. Prepared independent archive verifier checking CRC/all hashes, pinned golden identity,12 exact HTTP IDs/prompts/text,2 SSE goldens, actual496-projection readiness, committed producer sources, raw SSE arithmetic, measured HTTP-vs-worker full IDs and phase-window NVML peaks. Execute against terminal successful archived evidence before reporting metrics.

Prepared fixed-folder native4B worker launcher and4B vLLM/SGLang candidates using existing local checkpoint and prior4K/seq8 reference policies. Both reference scripts pass shell syntax checks. These are preparation only, not GPU serving qualification or results; no launch/source sync during current INT4 job. The existing4B worker still requires actual HTTP golden/semantic/stream gates and reference comparison. Target remains unchanged/unmet.
### P149 — saved MTP acceptance audit while INT4 pilot runs

Audited all16 complete128-token joined MTP3 diagnostic responses against their retained archive hashes and iteration counters. Text/image proposed-draft acceptance fractions0.6297348485/0.6477382098; weighted target positions advanced per verification2.8700564972/2.9111747851. Median verification calls44/42.5, partial replays23.5/22. The replay source explicitly reuses retained projections and does not rerun target GEMMs, so these calls cannot be treated as equivalent full-model forwards or used for a counterfactual speedup. No new timing, causal bottleneck or model quality claim. First exploratory audit used sum(processed-1) for accepted count; this fails when the terminal accepted token is sampled but not committed. Correct invariant is final sampled_so_far minus iteration count. Corrected audit passes all16 records, with total committed positions127 each. Retained source and JSON for reproducibility.

INT4 workflow remains live after its exact HTTP/SSE gates; benchmark manifest status running, partial result files retained. No interim aggregate reported. The original process must finish or fail authoritatively before stop/archive/reference launch; do not restart on observation timeout.
### P150 — original-layout NVFP4 heuristic/workspace sweep prepared

Extended standalone operand probe with explicit --algorithm-timing (old modes unchanged). For original TN layout, request up to32 cuBLASLt heuristic candidates at4/32/64MiB workspace. Retain actual returned count, candidate index/algorithm ID/status/required bytes. Preserve unsupported outcomes. Every launch must match the previously independent structured BF16 oracle before paired ABBA4 timing against the existing first-supported4MiB path. Include GPU event nodes and cache-eviction intent, keep descriptors alive during graph replay, and preserve actual mismatching output before fatal numeric failure. CUDA asynchronous/numerical failures are not silently skipped. Local CUDA13.2/MSVC compile/link passes; no local GPU execution and no production matmul edits. Raw local log copied to outputs/nvfp4-algorithm-sweep-local-build.log. Actual Blackwell candidates, oracle and timings remain pending until native INT4 worker ownership is released. This is not arbitrary-layout support, actual-checkpoint coverage or a model performance claim.

Original INT4 server and HTTP workflow are still live by helper handle observations; no source sync/rebuild/reference GPU launch occurred. Keep original process and raw partial pilot, with no interim benchmark claim.
### P151 — explicit model routing in reference quality capture

Added --model choices for pinned NVFP4, INT4 and4B to reference readiness/VLM/full-greedy capture, retaining NVFP4 default. Readiness checks actual served model ID; all outgoing quality requests carry selected model and provenance hashes the corresponding existing checkpoint config. Same-engine speculative comparison rejects an ordinary oracle from a different model (legacy missing field maps only to the historical NVFP4 default). No automatic timing. Six CPU reference tests pass, including subprocess/actual loopback HTTP fixtures verifying all four image requests and saved oracle use each selected INT4/4B name; fixtures are synthetic and not GPU evidence. Remote study source remains frozen5194b73b7 during the INT4 pilot. Original handles still running; text cohort completed, image cohort in progress, no full pilot metrics admitted yet.

### P152 — verified INT4 B1 HTTP pilot; vLLM INT4 startup underway

INT4 run837ba0310dbb482087ff0db06b5ba299 source5194b73b7/qualified2d902723 DLL completed exit0 after12 exact retained HTTP goldens and2 actual SSE gates. B1/T16/cap4096,41/104-token prompts,128 outputs,8 measured requests/cohort,2 warmup waves, seed3 JPEG/image-first/binary-grid, literal127.0.0.1. Each measured HTTP final full ID sequence matches actual worker output; all16 raw SSE counts/timestamps/text/metrics replayed exactly. Independently checked ZIP CRC/all124 content hashes, committed producers, pinned full-model golden, actual496 INT4 projection readiness and raw NVML phase-window maxima. Archive 545125 bytes/125 entries SHA256 97f710d6fd1b4f36bb08a2f3ad4feba0ff1c3283535a08074ac5b68a14803b09.

text: TTFT 697.010450000 ms; decode 4.455554473421 tok/s; aggregate 4.381545864518 tok/s; sampled adapter peak 21786648576 bytes.
image: TTFT 1641.609499999 ms; decode 4.449489909450 tok/s; aggregate 4.240589486730 tok/s; sampled adapter peak 21792940032 bytes.

Native INT4 server deliberately identity-guard stopped after successful terminal workflow; server wrapper exit1 records intentional shutdown, not benchmark failure. GPU subsequently39MiB used/32149MiB free/0%util. This is a single target-only INT4 compatibility-path pilot, not tuned INT4 or a precision-only causal comparison against mixed NVFP4 MTP3. NVML adapter samples are not instantaneous/process-attributed peaks; text maximum sampling gap2.248144s, so true peak may be missed. Both precisions use BF16 vision. Saved and visually inspected scientific chart outputs/Garnet-NVFP4-vs-INT4-B1.png/.svg, plot and underlying hashed data; footnotes state differing speculation, single pilots and no2x/reference claim.

After all native handles terminal and archive verified, installede307bd000 after6530 prior source hashes passed; production DLL retained. Started original helper95279 vllm-int4-server under raw run_logged ledger, pinned checkpoint/config and launcher hashes, same4K/seq8/cache/dtype reference policy. WSL systemd-root warning observed, process remains running with startup log; no readiness/quantization support, quality or performance result yet. Do not restart on observation timeout. No competing model/probe or source mutation while reference lives. Standalone algorithm sweep, SGLang INT4 and4B serving/reference/concurrency/quality/repetition requirements remain open. Target2x remains unmet.


### P153 — vLLM INT4 load, quality and eight-cohort pilot verified; image cache contamination retained

Pinned vLLM0.32.0 INT4 ordinary sourcee307bd000 loaded actual compressed-tensors WNA16 with MarlinLinearKernel. Startup reports some shape-padding overhead, KV37774tokens (9.22x4096 capacity), graph capture0.03GiB. Four VLM semantics/64image-token cases and16 full128-token greedy captures passed before timing. Original quality and pilot handles exit0; identity-guarded SIGTERM of actual Linux parent270 after completed pilot, then original server terminal0 and GPU39MiB used/32149free/0%util. No engine overlap.

Archive 913282 bytes/455 entries SHA256 a1e858e3abdef4594ee5754b0682da736859dcf036d930c36576309cc90eb8bb. Downloaded CRC/all454 content hashes, source commits, all64 raw measured SSE arithmetic, all8 phase-window NVML maxima,16 C1 stream texts against full-ID greedy captures,4 VLM responses and terminal ledgers verified. Native/vLLM INT4 B1 request bytes, transport and counts match; all16 prompt-ID pairs exact, only5/16 full generated-ID pairs exact. This does not prove cross-engine/broad quality equivalence.

text C1: TTFT(raw) 80.405099998ms; decode 77.366757288804tok/s; aggregate 74.113348144051tok/s; sampled peak 24687009792bytes.
text C2: TTFT(raw) 161.834999999ms; decode 77.359322100429tok/s; aggregate 142.035325433863tok/s; sampled peak 24687009792bytes.
text C4: TTFT(raw) 159.034800003ms; decode 75.379746581091tok/s; aggregate 277.373748422414tok/s; sampled peak 24687009792bytes.
text C8: TTFT(raw) 153.144700002ms; decode 72.435113946641tok/s; aggregate 535.718956813002tok/s; sampled peak 24687009792bytes.
image C1: TTFT(raw) 95.132499999ms; decode 77.333857501788tok/s; aggregate 73.699900998873tok/s; sampled peak 24687009792bytes.
image C2: TTFT(raw) 160.573900001ms; decode 77.336258878415tok/s; aggregate 140.756938581647tok/s; sampled peak 24695398400bytes.
image C4: TTFT(raw) 198.556799998ms; decode 75.370454688572tok/s; aggregate 271.317047934086tok/s; sampled peak 24697495552bytes.
image C8: TTFT(raw) 279.939650001ms; decode 72.317244030630tok/s; aggregate 501.220722090833tok/s; sampled peak 24827518976bytes.

Important limitation discovered during independent log review: nonzero multimodal cache hit rates after quality captures reuse the measured C1 image fixtures. Raw image TTFT remains archived but is NOT admitted as matched uncached image TTFT. Independent verifier now flags cache hits and suppresses image TTFT ratios in its matched comparison. Native INT4 path remains much slower in observed decode; no2x gain or tuned-fastest claim. Keep this first pilot, do not overwrite it or silently relabel its cache behavior.

Read-only installed-primary-source audit retained17 matching files with SHA256/context under results/vllm-int4-mm-cache-source-audit. API has separate POST /reset_mm_cache (processor/transport caches) and /reset_encoder_cache (scheduler and GPU encoder outputs); core source explicitly clears both components only through their respective methods. These debug endpoints warn that API return alone is not success verification. Next check router availability, clear both only while idle, retain responses/source and a fresh repeat with unique fixture evidence/cache counters before matching image TTFT. Benchmark already validates encoded AND decoded pixel uniqueness across its own warmup/measured cohorts; prior quality-capture overlap is the missing boundary. Initial read-only rg attempt found Linux rg unavailable (pipeline exit0); Python primary-source audit then completed successfully. No GPU/runtime change.

Original helper48577 sglang-int4-server started after vLLM terminal/idle, same sourcee307 and pinned INT4 checkpoint/4K/seq8 policy. Latest log shows shard loading; no readiness/quality/quantization compatibility/performance claim yet. Keep this original process, no restart on observation timeout and no source sync while live. NVFP4 algorithm sweep and4B/native concurrency/quality/repetition/full-context requirements remain pending;2x target unmet.


### P154 — stock installed SGLang INT4 load failure verified; standalone sweep resumed

Original helper48577 sglang-int4-server terminal exit9. Actual stack: gptq_marlin_repack raises size_n=96 not divisible by tile_n_size=64, then child SIGQUIT shutdown. This confirms the prior installed-source risk for the merged GDN96-row layer in SGLang0.5.22 compressed-tensors INT4. No quality or timing was admitted. No universal SGLang/backend unsupported claim and no checkpoint/precision substitution. Archive 338940 bytes/13 entries SHA256 792d3ae9e67cd8e6ca534fb65aaa4720f33a64ba7af20c3ae908e31716a22a18, downloaded CRC/all12 hashes and exact producer/source/terminal error verified. Nested prior installed-source archive matches SHA256 03c3f876b65f32232f2ba3dbee0c7f05ab4022b805547e0843ed8dbff05d0793.

After authoritative terminal state, GPU returned39MiB used/32149MiB free/0%util. Started original helper48577 operand-sweep-attempt2 using remote-only standalone target build, frozen sourcee307bd000, old production2d902723 DLL hash guard. New fixed result subfolder results/operand-sweep-attempt2 retains build/oracle/algorithm/resource ledgers. This is the prepared32-candidate4/32/64MiB heuristic sweep, independent structured arithmetic before timing; actual GPU results pending. No concurrent model or production runtime change.

Trusted coordination note requested no local compiler/RTX4080 work until10:20UTC; observed10:15:39UTC and kept compilation remote. Follow-up released the window early at10:16:02.809UTC. No local build/GPU execution during the requested interval; local lightweight evidence/file analysis only. This is method provenance, not a cross-host performance claim.

For vLLM INT4 cache correction, installed source confirms reset_mm_cache clears processor/transport caches separately from reset_encoder_cache clearing scheduler/GPU encoder outputs. Router availability and actual clearing/repeat evidence still needed. The first cached image-TTFT trial remains excluded; no new reference run or broad/cross-engine quality equivalence claimed. Keep remainingNVFP4/INT4/4B/reference/concurrency/repetition/long-context scope and2x target intact.


### P155 — original-layout NVFP4 candidate screening verified, not model promotion

Original remote helper48577 operand-sweep-attempt2 terminal0: standalone build, structured small oracle and nine projection-shape/workspace screens. Sourcee307bd000, qualified production2d902723 DLL identical before/after build and GPU execution. Archive 249818bytes/55entries SHA256 222f5592152db752e2c736ef061470e954fdfb849800df9df757c56b722fae07; downloaded CRC/all54 hashes, exact producer/CUDA sources against Git, three terminal ledgers, DLL identity and every retained canonical original/swapped BF16 output checked by independent CPU scale/sum/rounding oracle. Candidate numerical equality is an archived GPU producer assertion against that oracle, not separately retained full raw output for every successful candidate. No actual checkpoint weights or activations in this screen.

Up to32 heuristic candidates requested per4/32/64MiB workspace and shape;161 supported candidate/workspace trial groups across9 shapes (duplicates across workspaces, not161 distinct algorithms), each16 ABBA samples with explicit outside-event cache eviction. All returned supported IDs70, but heuristic variants differ; numeric algorithm ID alone cannot identify an implementation variant. Best-of-many selection is biased and not a confidence interval.

n17408-k5120-b1: best screening ratio 1.028232266, baseline 0.051279999ms/candidate 0.049872000ms, workspace budget 67108864bytes, heuristic index 1, algorithm ID70.
n17408-k5120-b4: best screening ratio 1.022566536, baseline 0.048575999ms/candidate 0.047503999ms, workspace budget 4194304bytes, heuristic index 0, algorithm ID70.
n17408-k5120-b128: best screening ratio 1.015586570, baseline 0.039616000ms/candidate 0.039007999ms, workspace budget 67108864bytes, heuristic index 2, algorithm ID70.
n34816-k5120-b1: best screening ratio 1.000169723, baseline 0.094288003ms/candidate 0.094272003ms, workspace budget 33554432bytes, heuristic index 1, algorithm ID70.
n34816-k5120-b4: best screening ratio 1.006676522, baseline 0.086847998ms/candidate 0.086272001ms, workspace budget 4194304bytes, heuristic index 0, algorithm ID70.
n34816-k5120-b128: best screening ratio 1.013686536, baseline 0.073472001ms/candidate 0.072480001ms, workspace budget 4194304bytes, heuristic index 4, algorithm ID70.
n5120-k17408-b1: best screening ratio 1.235372316, baseline 0.059455998ms/candidate 0.048128000ms, workspace budget 33554432bytes, heuristic index 3, algorithm ID70.
n5120-k17408-b4: best screening ratio 1.261016925, baseline 0.059519999ms/candidate 0.047200000ms, workspace budget 33554432bytes, heuristic index 4, algorithm ID70.
n5120-k17408-b128: best screening ratio 1.220383012, baseline 0.060160000ms/candidate 0.049295999ms, workspace budget 67108864bytes, heuristic index 1, algorithm ID70.

Joined gate/up B1/B4 ratios1.00017/1.00668 show no meaningful gain in this screen. Down5120x17408 candidate at32MiB gives1.23537 atB1 and1.26102 atB4; T128 best64MiB gives1.22038. Do not multiply these into model tok/s. No production runtime change/default promotion or2x result. Next retain full variant configuration, repeat selected candidates with fresh/actual checkpoint data and signed activations, then full model state/token/VLM gates before any HTTP timing; warm full-model profiling should determine whether the down projection is a worthwhile bottleneck. Broader remaining objective unchanged.

New trusted local quiet window10:18–10:26UTC was received; no local compiler/RTX4080 execution started. It was released early at10:19:25.599UTC. All builds in this interval remained on remote5090. A read-only installed-vLLM cache-router source audit is still live on original helper48577 audit-vllm-mm-cache-router; inspect its handle rather than restarting. No model/GPU worker is live after the standalone sweep. Cache correction/repeated reference,4B, native concurrency, quality/full-context and target2x remain open.

### P156 — vLLM cache-reset route registration inspected

Original helper48577 audit-vllm-mm-cache-router completed0;26 matching installed-primary-source files retained with hashes/context. Read copied entrypoints/launchers/api_server/routers.py: register_vllm_dev_api_routers(app) is conditional on VLLM_SERVER_DEV_MODE. The separate reset_mm_cache/reset_encoder_cache methods and core implementation were inspected in P153. Next preserved loopback repeat must explicitly expose the reset routes, verify idle state, clear both processor/transport AND scheduler/GPU encoder caches after quality captures and before the distinct warmup/measured workload, retain HTTP/reset/log/counter/source evidence and confirm image-cache behavior before admitting matched image TTFT. Do not assume either route exists in the stock launcher or that processor reset clears encoder outputs. No model/GPU process remains live now; no repeat or cache-clearing success has yet been claimed. Source audits remain in fixed study results folders; raw cached first pilot and its exclusion flag are unchanged. Standalone workspace screen remains a candidate, not a model speedup or target completion.
### P157 — explicit idle cache-reset repeat prepared

Added loopback-only vLLM cache-control helper: verify actual served model ID, require finite running/waiting/process/cache metrics, observe idle before any reset, acknowledge BOTH installed reset_mm_cache and reset_encoder_cache routes, retain raw metric snapshots and responses, preserve existing evidence paths, and require unchanged process identity plus positive query/zero new hit deltas after timing. Failed/missing metrics or busy state reject reset, and nonzero post-run cache hits reject pilot completion. This is bounded source/API/counter evidence, not a hardware cache-residency claim. Five synthetic CPU HTTP/control tests passed, including busy refusal, both endpoints, invalid/missing/ambiguous metrics, URL restriction, prior-output preservation and post-reset hit rejection. No model results in fixtures.

Added fixed-folder cache-reset wrapper enabling VLLM_SERVER_DEV_MODE only on the existing literal-loopback launcher, with occupied-port and existing-log refusal. Base checkpoint/quantization/context/seq/cache/dtype/graph policies stay inherited. Extended reference pilot with explicit output/quality/log/launcher selection and opt-in reset; historical defaults/first evidence retained. Shell syntax and source parsing pass. No local compiler/GPU execution during the newly requested quiet window through10:32UTC; lightweight CPU analysis/tests only. Remote deployment/GPU repeat pending; metrics schema/reset routes must pass actual runtime validation before any new measurement admission. Model2x target remains unmet.
### P158 — cache-reset source installed; original repeat handles live

Installede6d3aea5b after6536 prior source hashes verified, preserving qualified2d902723 DLL and compiled-native846b1e12f provenance. GPU immediately before launch39MiB used/32149MiB free/0%util. Original helper95279 vllm-int4-cache-reset-server and helper48577 vllm-int4-cache-reset-workflow started with raw source/artifact ledgers and fixed result names. Root workflow runs fresh4 VLM/16 full128 captures, requires all16 exact previous same-engine sequences, then explicit dual-cache reset and8-cohort pilot with zero-new-hit/process/counter gate. Do not admit metrics before actual schema/API/gates and terminal independent verification. No source sync/rebuild while worker lives; no other model launched.

Owned Linux parent171 was confirmed by checkpoint/served name/port command. Initial read-only process observation showed ~657MiB RSS with free WSL memory46GiB/no swap used; later identity-checked proc/status observation at10:36:58UTC records ~2.65GiB RSS/67 threads and poll wait, no owned TCP/UDP socket rows. This is startup activity, not a deadlock/network diagnosis or benchmark. Original readiness workflow is still waiting; no restart for quiet output/observation timeout. Raw startup proc/fd/stack observations and exact method retained in fixed study paths. No process signals or global environment repairs performed.

Cache-reset stop method requires successful terminal quality/http stages, a completed benchmark and unique checkpoint/model/port identity; records only selected VLLM_SERVER_DEV_MODE/GARNET_VLLM_LOG environment keys before SIGTERM. Archive method also requires complete workflow, all source/metric/raw files, both reset acknowledgements and positive-query/zero-hit counters. This makes incomplete or cache-affected attempts fail admission while retaining their evidence. These new root methods are reviewable here but remote app source remains frozene6d3aea5b. The prior cached first pilot remains excluded for matched image TTFT. Native optimization/actual-checkpoint candidate/full-model profiling,4B, native batching/concurrency, repetitions/full-context/broader quality and target2x remain open.

The local10:32UTC quiet-window request was released early at10:26:52.788UTC; a later bounded request through10:50UTC was received and is being respected. No local compiler/RTX4080 execution; remote5090 work and light CPU evidence/tests only.
### P159 — cache-reset exploratory pilot verified; cross-process mismatch retained

The original strict workflow failed before timing: previous-process ordinary captures matched15/16 full128 output sequences (all16 prompts/semantic request hashes identical). Only s3-text-c1-r2 differed, first index60; no harmlessness or cause established. Its failed report/ledger remains in the evidence archive. On the same live ordinary vLLM process, a further repeat matched all16 full sequences exactly. A separately labelled self-qualified exploratory pilot then completed; it is explicitly not2x-eligible and does not erase the failed cross-process gate.

While idle, both /reset_mm_cache and /reset_encoder_cache acknowledged HTTP200. Raw process identity stayed unchanged. Processor counters went20queries/8hits to82queries/8hits:62 new queries, zero new hits. The62 correspond to image warmup plus measured fixtures. Cumulative earlier hits remain in logs; absolute cache-rate logs are not a new-hit admission criterion. This is processor-counter/reset evidence, not proof of hardware cache residency.

Independent downloaded archive verification passed CRC, all559 file SHA256s, source producers at e6d3aea5b, four actual VLM semantics, both16 full128 raw captures, the15/16 prior-process mismatch,64 measured raw SSE streams, eight cohort arithmetic and sampled NVML maxima, current C1 stream text matching all16 quality captures, successful terminal ledgers and raw after-run Prometheus counters. Text C1 TTFT86.3078ms/decode77.45048tok/s; image C1 TTFT124.8349ms/decode77.42279tok/s. C8 aggregate text531.64265/image490.36042tok/s. These are INT4 vLLM exploratory results, not Garnet gains or broad quality. Text C1 includes batched token arrivals; replay retains actual token accounting. Same short41/104-token prompts,128 outputs,capacity4096,8 requests/cohort; not full4K prompts. Peaks adapter-wide sampled: C1 24687009792bytes; imageC8 24825421824bytes.

Archive vllm-int4-cache-self-qualified-evidence.zip1339595bytes,560entries,SHA256114eb8f1361fda431fd39ae7b4c942d7ecabd861f7ecc8c7b363b3062ef8ac7a. New root workflow/diagnostic/stop/archive methods retained verbatim under this study directory with underscore filenames. Identity-guarded SIGTERM only after completed pilot; original server handle observed terminal0, GPU39MiB afterward. Production DLL2d902723 unchanged. No source synced or rebuilt while server lived. A first local verifier invocation passed a decoded string to a bytes API, failed before admission, and was corrected; complete rerun passed. Goal remains open:2x decode/concurrency not demonstrated, native batching/4B/repeats/full-context/broader quality and actual-checkpoint kernel optimization remain required.
### P160 — repeated NVFP4 screen identifies complete candidate configuration

Standalone probe now queries all nine public CUDA13.2 cuBLASLt configuration attributes with documented storage types (ID/splitK int32; tile/reduction/swizzle/custom/stages uint32; inner/cluster uint16). Every query records API status and bytes written; successful unexpected sizes fail the probe. Estimated waves, workspace requirement/budget and heuristic index remain retained. Installed source1f6ae24813d9cb8b77ad82b07f52cb96e9c5a9cf after6549 existing source hashes verified. Original vLLM server terminal0 and GPU39MiB/0% were observed before source update. Fixed study folder reused; serving DLL2d902723 unchanged before/after standalone build and both runs. Native compiled source remains846b1e12f.

Actual5090 build, small independent structured oracle and full shape/algorithm repeat all exited0. Archive operand-variant-attempt1-evidence.zip247267bytes/55entries/SHA2560ec97ef866e2bcf3a26fbe0c8508cef7a8fadf6737567186295d5812152acb1e. Independent verifier checked CRC/all54 file hashes, source producers, every retained original/swapped BF16 value against CPU analytic oracle, all successful candidate trial structures and all9 attribute query sizes/statuses. Candidate output equality is GPU-producer checked against this oracle; full successful candidate arrays are not independently retained. This remains positive structured synthetic data, not actual checkpoint/sign/cancellation/general model correctness.

Down-projection N5120/K17408 best screening ratios repeated at1.2612(B1),1.2559(B4),1.2190(B128), selected from many candidates. Each best configuration: algorithm70,tile20,stages37,cluster5,inner0,swizzle0,custom0,splitK-2,reduction2;64MiB allowance. Previous screening also showed~1.24–1.26 atB1/B4 but did not retain full configuration. SplitK-2 is recorded verbatim from typed API; no undocumented interpretation assumed. Reduction changes require signed/actual checkpoint numerical gates before runtime promotion. Joined gate/up gains remain~noise-level. Best-of-many selection is biased; no repeated confidence/full-model tok/s or2x claim. Production unchanged; final GPU39MiB/0%.

Next optimization gate: test selected complete configuration against current production first-supported path using actual pinned down-projection weights/scales and signed activations, retain all raw outputs and an independent bounded decoded oracle. If correctness passes, benchmark the fixed candidate across repetitions, then apply shape-specific selection and rerun full-model target/MTP/HTTP gates. Keep budget/variant identity, fallback and prefill behavior explicit. In parallel scope remains native batching correctness/concurrency,4B HTTP/references,full-context and broad quality. No local GPU work or model duplicate created.
### P161 — actual full down-projection/signed-input candidate correctness verified

Source9b50fe9c2a5ed98f0d2e04ce69b8dadfa5923898 installed after6558 existing source hashes verified. Actual5090 fixture preparation, standalone build, canonical small oracle and full checkpoint candidate gate all terminal0. No source sync while GPU worker lived. Serving DLL2d902723 unchanged; compiled native provenance remains846b1e12f. Final GPU39MiB/0%. Local quiet-window request through11:20UTC was respected (no local compiler/GPU work) and released11:02:38.214UTC. Local work only lightweight source and independent CPU evidence checks.

Fixture reads the entire actual pinned layer0 down-projection N5120/K17408 packed weight, straight E4M3 scales and calibrated F32 input/weight scalars. Index JSON independently equals the committed pinned NVFP4 header-evidence index. Reuses existing checkpoint; one bounded fixture under results/nvfp4-down-candidate-attempt1/fixture, no model clone. Signed arbitrary packed E2M1 inputs and scales1/2/4 generated with PCG64 seed20261024,128 rows; not captured model activations. Global alpha F32 product2.8225300411577336e-05. CPU float64 decoded oracle spans32 selected full-K channels and128 rows; fixed prior tolerance max(1e-5,abs(expected)*0.0078125). Independent scalar compensated sums additionally checked four cancellation cases.

AtB1/B4/B128 all5120/20480/655360 retained BF16 outputs equal baseline exactly (680960 total), both paths finite and both meet every selected-channel CPU tolerance. No timing performed. The tested policy chooses the first matching public configuration: algorithm70,tile20,stages37,cluster5,inner0,swizzle0,custom0,splitK-2,reduction2 under64MiB allowance. Required workspace15729024bytes. Importantly, THREE heuristic entries share all nine public attributes at each batch. Tested first indices3/4/0; prior best-of-many timing screen chose4/5/1 under64MiB. Do not equate these choices or transfer the1.26x component ratio without further identity/timing evidence. CUDA13.2 cublasLt.h explicitly permits serialization of the64-byte algorithm descriptor for reuse only with the same cuBLAS version. Next retain opaque descriptors and compare duplicates, then freeze an exact selection policy before runtime promotion.

Raw archive nvfp4-down-candidate-attempt1-evidence.zip54206386bytes/50entries/SHA25644a374aa2b96d3316715f70aad5e44b1eeec1dbc6e71409439cf4f80c2cdb2ca. Independent verifier passed CRC/all49 file hashes, pinned index, fixture/source tensor hashes, full output bytes/counters/configuration, all4096 decoded CPU oracle outputs and compensated scalar spot checks, source producers and four terminal ledgers. Full fixture/raw outputs retained for paper reproduction; source9b50 and execution/install/archive methods preserved. CPU signed decode and explicit scale-layout boundary checks passed before execution. No per-layer/full-model token parity, actual activation coverage, model speedup or2x conclusion follows from this one-layer gate.

Implementation task updated to supersede stale live-process assertions and record current native/vLLM/stock-SGLang INT4 results, reset qualification limits, fixed candidate gates and serialization/workspace safety. Future opt-in runtime policy must preserve legacy engine deserialization and persist candidate workspace requirements; environment changes must not enlarge scratch needs for an existing cached engine. Next after descriptor identity and fixed-candidate timing: serialization/clone/workspace checks, full-model retained state/token/MTP/HTTP gates, then matched performance. Native batching/concurrency,4B serving/references,repeats/full4K prompts/broader quality and requested2x remain open.
### P162 — fixed-policy signed-weight timing; opaque identity failure retained

Installed407a57206e63e5507b5137cee13e2d803aa4f152 after6562 prior source hashes verified. Reused the complete P161 fixture in place; no new model/fixture copy. Actual5090 standalone build, canonical small oracle and checkpoint correctness/timing workflow all terminal0. AllB1/B4/B128 preflight cases passed before any timing; every subsequently timed instance also independently checked full BF16 equality and selected-channel CPU oracle immediately before capturing that same selected instance. Full gate and timed-phase raw outputs retained. Serving DLL2d902723 unchanged; finalGPU39MiB/0%. No local compiler/GPU work; new25-minute local4080 quiet request respected.

Within each heuristic query, all three entries sharing nine public settings have IDENTICAL64-byte opaque descriptors. However the chosen opaque bytes differ between separately created preflight and timed contexts at each batch. An initial independent verifier requiring cross-phase byte identity failed and is retained verbatim as verify-nvfp4-fixed-timing-strict-descriptor.py plus nvfp4-fixed-timing-descriptor-gate-failure.json; no cause or harmlessness established. A separately scoped verifier admits only the public-config-first-match POLICY and each timed instance's own full-output/CPU gate; it explicitly records all three cross-phase identity failures and cannot claim identical opaque algorithms across contexts. Descriptor serialization reuse remains limited to the exact cuBLAS version. Do not erase the failed stronger claim or substitute byte-identity assumptions.

Independent scoped verification passed archive CRC/all45 file hashes, exact linkage to P161 fixture archive44a374aa..., independently rechecked all4096 decoded CPU values plus compensated sums, every new gated/timed raw BF16 output, public configuration/within-query descriptors,32 ABBA event samples per batch, source407a57206 and three terminal ledgers. Component medians: B1 baseline0.058848001062870026ms/candidate0.047519998624920845ms ratio1.238383896585562; B4 0.058687999844551086/0.047839999198913574 ratio1.2267558701356305; B128 0.06014399975538254/0.047839999198913574 ratio1.257190651390069. Eight ABBA repetitions within ONE process, signed synthetic activations/one actual layer0 projection, eviction intent256MiB outside event timing. No independent-run confidence, actual activation coverage, full-model/token quality, end-model gain or2x result.

Archive nvfp4-down-fixed-timing-attempt1-evidence.zip4511930bytes/46entries/SHA256b8255331b3372dcdf4d0923a89dba3e95bda68afb791da83347d866777d0c7666. It links the previous full fixture archive by verified SHA rather than duplicating the54MB fixture archive. Actual reports expose CUDA runtime13040 and cuBLASLt130400, including the prior P161 report: authoritative loaded runtime/library here are13.4, whereas earlier general setup summaries cited installed CUDA13.2 tools. Do not conflate installed toolkit/driver with loaded library version. Public attribute types were inspected in local13.2 headers and successfully compiled/queried remotely; all9 returned documented byte widths. Runtime candidate compatibility must use actual loaded library information, not the older setup summary.

Next implement an explicitly opt-in, shape/batch-specific candidate POLICY (not opaque-byte equality across contexts), persist selection/workspace requirements in plugin serialization while preserving legacy16-byte engine state, and verify clone/reload/workspace/missing-candidate behavior. Then run retained full-model target/state/MTP/HTTP gates before end-to-end timing. All native batching/concurrency,4B references/serving,repeats/full4K prompts/broader quality and2x requirements remain open. No runtime promotion yet.
### P163 — default-off native candidate integrated; component gates and native build verified

Added GARNET_NVFP4_DOWN_CANDIDATE=1 at lowering time only, scoped to N5120/K17408. Selection is first match of all9 public attributes among32 heuristics, cuBLASLt130400 only, B1/B4/B128 and exactly64MiB workspace; unsupported policy/library/batch/workspace or absent/unlaunchable candidate fails explicitly. Default path retains8-heuristic first-supported selection and4MiB workspace. No assumption of opaque identity across separately created contexts; P162 stronger descriptor failure remains retained. Other batches of an opt-in plugin use its unchanged legacy algorithm/workspace policy.

Plugin v1 retains its16-byte payload and4MiB contract. Opt-in plugin v2 appends policy tag1 to20bytes, retains flag through clone/reload, and asks64MiB only at qualified batches. Separate creators reject cross-version payloads; malformed/truncated/unknown tags and wrong candidate geometry are rejected. Environment is sampled only by lowering; existing cached engine scratch requirements cannot change when environment changes. Source inspection confirms CUDACompiledGraph builds into a new per-instance temp engine path, then serializes/deserializes that engine. Full-model probes therefore exercise rebuilt engines with persisted plugin policy. The fixed workspace is independent of opaque descriptor bytes.

Attempt1 source7dbfcf74b installed after6565 previous hashes verified; original build terminal1 at modelopt_plugin_test.cpp errorC2248: std::unique_ptr's default deleter cannot invoke protected IPluginV2DynamicExt destructor. No GPU tests or serving DLL rebuild occurred. Preserved complete failed source/ledger/log. Corrected test ownership to destroy() via custom deleter in5d675933663536f27f36bcb8e9566ced3dbc7656; installed after6568 source hashes verified, retaining qualified2d902723 DLL. A diagnostic log-print attempt also encountered consoleGBK encoding on replacement characters; bounded ensure_ascii JSON filtering then captured the actual C2248 error. This was observation encoding, not a model failure.

Attempt2 actual5090: standalone build0; serialization/clone/workspace/legacy version contracts0; all4 legacy native plugin serialized-engine/CUDA-graph cases0; canonical small oracle0; actual signed-weight checkpoint baseline/manual candidate/production policy three-path gate0; native Garnet rebuild0. All680960 BF16 words equal across all three paths; bounded CPU oracle passes and explicit4MiB andB2 requests reject. New serving DLL SHA256e82279af698dd7798be3f158ace999eef29be4ed609e8cd50cf1ae36c90cd750, effective native source5d6759336. Old2d902723 DLL and prior source remain in qualified-before-nvfp4-down-runtime backup; no primary deployment assets removed. No source synchronized during live GPU stages.

Component archive nvfp4-down-runtime-evidence.zip3459607bytes/63entries/SHA2569778245cdc4f336627513063402b73a79bc79ecc3999adde7fb29d7e400b532e. Independent verifier passed CRC/all62 content hashes, source5d6759336 producers, linked P161 fixture archive and decoded CPU oracle, every baseline/manual/production raw output,6 successful terminal ledgers, original failed compile ledger/error and component success markers. Contract and legacy plugin GPU assertions are archived producer checks, not independently retained intermediate engine tensors. Build identity is recorded by raw build/identity artifacts; full-model validation remains separate.

After original attempt2 handle terminal0, GPU85MiB/0% was observed. Original helper48577 job nvfp4-down-full-qualification is now LIVE, source frozen5d6759336/librarye82279af. It sequentially runs: flag0 full12 target/token/normalized goldens on rebuilt DLL; flag1 full T128 packed-state hashes versus retained original teacher; flag1 full12 target/token/normalized/VLM goldens. Methods and exact flag-setting batches are retained. At latest read baseline statusconstructing_decoder,0cases,GPU21519MiB/0%; this is model construction, not a success/timing result. Original handle remains authoritative; do not restart on quiet output/observation timeout. MTP/HTTP gates precede future benchmarks. No new end-model speed claim;2x decode/concurrency,4B serving/references, native batching,repeats/full4K prompts and broader quality remain open. Local quiet window was released11:17:34.7499223UTC; no local compiler/GPU use during it.
### P164 — all-batch candidate fails image state; same-binary off controls verified

Original helper48577 nvfp4-down-full-qualification is terminal1, not a timeout. Rebuilt-library flag0 baseline passed all12 retained full128/text/image/VLM token and normalized-vector goldens. Flag1 fullT128 packed-state run repeated deterministically three times per input but failed equality to retained original on image; workflow stopped before candidate generation/MTP/HTTP or benchmarking. Source remained frozen5d6759336, DLL e82279af698dd7798be3f158ace999eef29be4ed609e8cd50cf1ae36c90cd750. Candidate remains default-off and is NOT qualified for VLM serving.

CPU diagnosis and independent downloaded raw verification: text matches all167 chunks in all3 repeats; image differs in chunk0 and146..165 (21/167) in all3 repeats. Each case hashes174587904 F32 words; remaining full arrays are producer hashes, not independently retained. Raw first1048576-word chunk independently shows image307084 changed hidden words, zero changed prefix state words, max absolute15.375 and RMS0.2067020162799245; all finite. Changed hidden rows44..103 are60 ACTIVE image-prompt rows of104, with zero changed padding rows. Text prefix is byte-identical. Same input hashes and repeat stability exclude different inputs/within-run random drift, not internal numerical causes or broader quality issues. Layout-only interval mapping via the retained decoder_state_layout function places changed state chunks across layers56..63; this is not proof of the first changed operation.

Failure archive nvfp4-down-state-failure-evidence.zip18383085bytes/45entries/SHA256d872a146138e9d3bdc6f14d9d140ba9bfd1f9a1c0dcc023c5ea694b839306a87. Independent verification passed CRC/all44 file hashes; exact links to component9778245... and original full-modelb3dbb926... archives; all12 baseline golden fields/raw normalized digests; original copied teacher; full167-chunk producer comparisons/repeat stability; independently computed raw-prefix difference counts/magnitudes; source5d6759336 and baseline0/state1 terminal ledgers. No candidate generation result or new benchmark admitted. CPU per-row and documented-layout analyses saved separately in outputs with source/config hashes.

To separate the option from the rebuild, original helper48577 nvfp4-down-state-controls ran two NEW sequential flag0 T128 controls on exactly the samee822 DLL: nonjoined/original192NVFP4+208FP8, then joined128NVFP4+208FP8. Both terminal0. All inputs identical; original control matches old teacher all six full-state cases, joined control matches original, while active candidate differs. This isolates the regression to candidate configuration under these retained inputs. Split-K/reduction/greater workspace may affect execution; exact internal cause remains unproven and no harmlessness assumption made. GPU85MiB/0% observed after both actual processes closed; no source sync during live workers. New local4080 topology quiet request was respected with no local compiler/GPU work and later released.

Control archive nvfp4-down-state-controls-evidence.zip7288674bytes/SHA256a4de542248261b474c704675295f43e8096868140a6fb20dd4abb4a44c7ecdbf. It has33 logical files,23 stored blobs and10 SHA-verified aliases for byte-identical snapshots; original raw files remain in the fixed remote study folder. Independent verifier resolved/hashed every logical path, checked archive CRC/failure linkage,6-case/167-chunk coverage, raw first snapshots, native projection counts/source/binary identity and both terminal ledgers. Default-off new binary's retained state behavior therefore matches qualified teacher; option-on does not. Source/binary backups and all failed artifacts remain intact.

Next implementation hypothesis: a distinct serialized decode-only policy, preserving baselineT128 prefill, may retain the B4 decode-component gain without this prefill regression. It is unimplemented/unproven here. Preserve v1 legacy16-byte engines and historical v2/policy1 behavior for reproducibility; give the new policy its own version/tag, test clone/reload/workspace routing, then rerun complete full-model state/target/MTP/SSE gates before any timings. Do not waive P164 or present component1.24x as model gain. Full VLM/concurrency and2x objective remain intact, alongside4B comparisons,repetitions/full-context and broader quality requirements.
### P165 — separate V3 decode-only policy built; component proof and live full-model gates

Implemented V3/tag2 decode-only policy: qualified input rows1/4 may request the fixed candidate; T128 and all other rows use the original4MiB/8-heuristic path. V1/16-byte legacy behavior stays unchanged. V2/tag1/20-byte historical all-row policy retains its original128-row64MiB behavior for failure reproduction; creator rejects V3 payloads and vice versa. V3/tag2 uses its own creator/version3, serializes workspace routing, and preserves it through clone/reload; unknown/truncated tags rejected. The same explicit lowering opt-in now constructsV3 on new builds. This changes an implementation candidate, not the full VLM/concurrency/2x objective. P164 is not waived and historicalV2 is not admitted for serving.

Sourcecf1e43a34e09286952f783e3136fb01b5d16ac63 installed after6568 prior hashes verified, backing upe82279af DLL/source. Decode-only attempt1 build exited1 at test line46/C2660: creator result has IPluginV2 base interface, whose getWorkspaceSize overload is not the dynamic-descriptor overload. No GPU tests or native rebuild occurred. Fixed with checked dynamic_cast<IPluginV2DynamicExt*> in754cfa42473dd106a584c5e9584a54e8d994169b; installed after6585 prior source hashes verified. Original failed source/build log retained, rather than overwriting attempt1.

Attempt2 source754cfa424 actual5090: test build0; V3 decode-only plus historicalV1/V2 serialization/clone/workspace/creator-admission contracts0; four legacy plugin serialized-engine/CUDA-graph GPU cases0; small canonical oracle0; pinned signed-weight manual/bare-production selector comparison0; native Garnet build0. The bare helper deliberately still supports historical128-row selection for research and its680960 equal words do NOT proveV3 prefill routing; that routing is independently exercised by the versioned plugin contracts and pending full-model state gates. New DLL ba67dc9d4f8b9948615e522e3b6ad0aa10ea6f742505465084449fe57ead0b8b. Qualified olde822/2d902 backups and all model assets preserved.

Component archive nvfp4-decode-only-evidence.zip3479964bytes/63entries/SHA256a2b1160b6f4d7dece5b06a562cafb6a910359a5478b3ff0f6835d5beb1cc8fad. Independent verifier passed CRC/all62 file hashes, source754cfa424 producers, prior full fixture linkage/decoded CPU oracle, all raw baseline/manual/bare selector outputs, actual V1/V2/V3 contract success markers,6 successful terminal ledgers and originalC2660 failure. No new model-speed or broad quality admission follows from components. GPU85MiB/0% observed after component job terminal0.

Original helper48577 job nvfp4-decode-only-full-qualification is LIVE, libraryba67dc9d/source754cfa424 frozen. Four sequential stages: same-binary flag0 original T128 full state versus retained teacher; flag1 joined/V3 T128 full state versus that new original; flag0 full12 target/token/vector goldens; flag1/V3 full12 target/token/vector goldens. Existing validate_joined_model_gates runs only after all stages pass. Latest authoritative handle still running; stage state-original passed and matched the old full-state teacher, state-candidate running. Do not restart for quiet output; no new benchmark yet. If full gates pass, MTP target pairs/SSE and then matched serving timing remain required. New local50-minute compiler/RTX4080/SDK-header freeze is respected: remote5090 and light local CPU verification only, no shared SDK edits.

Light CPU analysis also replays all16 measured C1 SSEs in the previously qualified19bafb0f... joined-MTP archive (not this candidate). Text totals354 full target forwards/1024 output tokens,665 accepted/1056 proposed draft tokens; image349 forwards/1024 output,673/1039 accepted/proposed. Post-initial tokens per full target forward2.8701/2.9112; these are counter ratios, NOT speedup. MTP prefill calls total96/216 over8 text/image requests (12/27 per request). Source confirms prefix replay commits retained authoritative projections, not another full target forward; do not double-count it as a full-body pass. Counts alone do not establish stage durations or causal speed bottlenecks; hot profiling remains needed. Exact method/JSON retained for paper planning, without timing estimates or2x claims.

P165 live-handle follow-up: both state-original and state-candidate are now terminal0/pass in the SAME original helper48577 workflow. V3 T128 text/image packed-state cases reproduce the new original/retained teacher; this is the producer full-hash gate, not yet a separately downloaded full-model archive. Baseline generation stage is live; candidate generation/MTP/SSE/performance remain pending. Source754cfa424/libraryba67dc9d remain frozen; no source update or restart.

### P166 — V3 full-model proof verified; strict HTTP workflow started

Original nvfp4-decode-only-full-qualification completed0. All four stages passed: same-binary flag0 original T128 state, flag1 joined/V3 state, flag0 full12 generation goldens, flag1 full12 generation goldens. Independent downloaded verification checks all73 logical files (44 stored,29 digest aliases), raw first state chunks, normalized vectors, retained token IDs, source754cfa424 and DLLba67dc9d. Archive nvfp4-decode-only-full-evidence.zip7537999bytes/SHA256cebf38e7833e1d0eb15f32c655f91c555e1e5bf1c7c2704e1dd72e2909e0012b. Remaining full-state chunks are producer hash assertions, not retained full arrays. This does not establish broad precision, MTP correctness or performance.

After independent proof and matching local/remote root-method SHA256, started original helper95279 native-mtp3-decode-only-server and helper48577 native-mtp3-decode-only-workflow. Both remain live at this entry. Exact app source754cfa424/libraryba67dc9d remain frozen; no rebuild/source install while worker live. Whole worker-lifetime adapter VRAM sampler wraps bridge and is excluded from eventual owned bridge-tree shutdown. Existing resource phase text says construction-and-correctness; here it spans the entire lifetime, including any later pilot, and cannot imply process-attributed instantaneous peak or uninstrumented performance.

Timing admission requires20 target/MTP full-token pairs including four semantic VLM cases, six retained goldens, exact20 prior-qualified target identities/request payloads, two actual SSE gates and eight simultaneous HTTP-client output-isolation controls. Then offered-load C1/2/4/8,8requests per cohort,2warmups,128outputs,seed3 fixtures. Bridge uses one serial native B1 research slot; concurrent callers queue. This explicitly DOES NOT qualify true CUDA batching or replace the original concurrency-throughput objective. One pilot cannot establish causal improvement/repeat confidence. Correctness-first failure is retained and excludes timing claims.

Root methods committed9cca61372 and pushed on qwen38-5090. New raw archive method preserves either failure or success plus terminal ledgers; independent CPU verifier prepared (syntax only, not yet run on an archive) to replay64 measured SSE streams, compare worker IDs, recompute summaries/sample maxima and link full-model/prior-qualified archives. No new throughput/TTFT result exists at this entry. Actual latest sample22246264832 adapter bytes while worker loading; no diagnostic response yet. This is construction observation, not serving peak.2x decode and true concurrency remain unmet; native batching,4B HTTP/reference comparisons, broad/long-context quality, repeats, tuned references and hot-stage profiling remain open.

P166 terminal follow-up: original HTTP workflow exited1 at prior20-output identity admission, BEFORE SSE/queued controls or any benchmark. All20 current target/MTP comparisons pass, including four semantic VLM cases; independent replay also confirms six retained goldens. Across old qualified20 requests, identical request payloads and prompt IDs,15 full outputs equal and5 differ. First changed zero-based generated index: text-r5=35,text-r6=26,image-r1=65,image-r4=103,image-r6=102. Current target and MTP agree for each failed old-identity case. This is NOT sufficient serving admission and no harmlessness/cause assertion is made.

Before shutdown, original native-v3-live-failure-repeat ran each of five failing requests in both target/MTP modes against the SAME live worker, all10 complete sequences repeat exactly; terminal0 and raw artifacts retained. Identity-guarded bridge-tree stop excluded resource wrapper; original server terminal1 reflects intentional shutdown. GPU85MiB/0% observed afterward. No timing or concurrency throughput result exists forV3. Prior chart remains unchanged.

Raw archive native-mtp3-decode-only-http-evidence.zip424792bytes/SHA256a8a1628572e4081bd7daff926f93b341833285269a9f0021679611f812c22f040. Independent local failure verifier PASSED CRC/all140 file hashes, three linked archive digests, all20 current pairs/sixgoldens, fullold20 request/token comparison, raw10 repeat pairs, successful diagnostic/repeat and failed primary terminal ledgers, absence of benchmark/SSE files, guarded stop and whole-lifetime12904 NVML sample maximum. Evidence verification does not make the optimization pass. Prepared same-DLL flag0 HTTP20 control to distinguish option effect from rebuilt/process differences; not yet a result. Full original2x/nativeGPUconcurrency/4B/longcontext/repeated quality scope stays open.

### P167 — same rebuilt-DLL option-off HTTP20 control exact; profiler capability smoke live

Original helper48577 native-mtp3-decode-only-off-workflow completed0. All20 same-profile target/MTP full-token pairs pass, then every target AND MTP output (40mode captures) equals the old qualified19bafb0f archive. Identical semantic request payloads/prompt IDs are independently checked. Same rebuilt source754cfa424/DLLba67dc9d as failed option-on experiment. Off versus on differs in exactly the five P166 requests; off matches old in all20. This makes the explicit option the observed differentiator under these retained controls; it does not identify the internal numerical cause or prove universal behavior. Option remains default-off and rejected for serving/timing. No speed result added to chart.

After original control workflow terminal0, identity-guarded bridge-tree stop leaves monitor alive; original server terminal1 is intentional shutdown. GPU85MiB/0% observed. Control archive native-mtp3-decode-only-off-http-evidence.zip333486bytes/SHA2565654bc6c86a4ced45da2f804fe77c187addb7624d4a1a012d8001bba8770ef948. Independent local verifier PASSED CRC/all115 file hashes, linked full/failure/old archives, common DLL and explicitflag0 launcher, all20 self-target/MTP comparisons/sixgoldens, actual old/off/on fullIDs and request matching, terminal ledgers, guarded stop and whole-lifetime sample maximum. This is untimed correctness evidence, not GPU batching, broad quality, full context or2x qualification.

Prepared and started original helper95279 nsight-cuda-smoke-attempt1 only after control original server terminal and GPUidle. Nsight executable is locally verified2024.6.2.225-246235244400v0; installed presence does not prove compatibility with loadedCUDA13040/SM12.0. Smoke wraps an existing bounded real normalization graph with --trace=cuda and node-level graph tracing; stores binary/version identity and raw profile/stats ledgers. No timing or profiler-usability claim until actual CUDA kernel/API records are verified. Source/library remain frozen; no local compiler/GPU/sharedSDK work during other-thread quiet window, extended through13:25UTC. No trace attached to serving pilots. Original2x/concurrency/4B/long-context/repeatedquality and tuned-reference scope stays open.

P167 profiler follow-up: original nsight-cuda-smoke-attempt1 completed0; native bounded normalization correctness report passed, graph captures1/replays2/failures0; profile and stats ledgers both terminal0. Downloaded archive nsight-cuda-smoke-attempt1-evidence.zip1057910bytes/SHA25695313a47ccff9f017115786872307df2bee488a91ed5898ca55705d4aff6b5ed independently passes CRC/all13 file hashes, off-control archive linkage, tool/library identity, terminal ledgers and direct read-only SQLite inspection. Actual CUDA kernel rows5, runtime/API rows24688; GPU kernel durations sum19904ns for this tiny smoke only. This proves bounded real CUDA trace coverage on this setup, NOT model timing. Raw output arrays were not retained, so normalization equivalence remains producer digest/assertions.

Prepared sourceef7ec2b4a root study methods for a separate warm joined-MTP trace, then started original helper95279 warm-mtp-cuda-trace-attempt1 under an outer correctness ledger. Application source754cfa424/DLLba67dc9d stay frozen and NVFP4 candidate flag0. Native probe warms 12 exact cases (text/image full128 and four semantic VLM cases, repeated), signals warm-ready, waits for controller nsys start, executes two exact text/image cases under collection, waits for nsys stop/release, then closes naturally. Controller keeps original launch future authoritative, records launch/start/stop/stats separately and abort/release signals on failures; no restart on observation timeout. Nsight node tracing has overhead, so eventual data is a hot-kernel/API diagnostic, not a new HTTP pilot, causal gain or benchmark. Latest original job is started/live, no full trace result yet. True CUDA batching,4B matched serving, full-context/broad quality, repeated/tuned references and2x remain open. Local compiler/GPU quiet window still respected.

### P168 — interactive warm trace access-violation failure retained; startup-marker alternative

Original helper95279 warm-mtp-cuda-trace-attempt1 completed1, not a timeout. All12 warm cases reproduce the current option-off full IDs, then interactive nsys start returned0 and capture-go was written. Original launched process returns3221225477/0xC0000005 before capture-done or either capture case. Controller marks abort/release and preserves original future/outer terminal failure. No captured-case correctness or bottleneck ranking is admitted. Read-only Windows Application/1000 query filtered to the study path returned no matching records; faulting module/internal cause is therefore unproven. Stats tool itself returned0 but explicitly SKIPPED CUDA reports: exported SQLite has no CUDA kernel or API table. Tool exit0 is not trace success.

GPU85MiB/0% observed after actual terminal. Raw partial report, SQLite, full12 warm outputs, launch/start/controller/failure-stats ledgers, source and tool linkage archived to warm-mtp-cuda-trace-attempt1-evidence.zip462464bytes/SHA25607a4ddad4889e3817677724c46c54c9bc6690feba8ab60cdd565a95d8d75d6f2. Independent downloaded verifier PASSED CRC/all31 hashes, linked archives,12 raw fullID comparisons, exact launch/outer exit status and absence of CUDA data/capture completion. No application arithmetic/source/library changed; candidate flag0 remains selected.

Prepared an alternative collection-from-process-launch method, following the successful bounded smoke mode. Native graph warm-up still requires12 exact cases and two later full128 text/image cases, but no interactive start on already-created graphs. Unique bounded HtoD copies (8973/8977 F32 elements,35892/35908bytes) bracket the warm section. Independent SQLite verification MUST prove unique markers and ordering before filtering kernels/API records; aggregate startup/build report is never a hot bottleneck ranking. Full profiler overhead remains diagnostic, not HTTP/model speed. New methods syntax-checked; no result yet. Preserve original failed attempt rather than overwriting. Original2x and full native concurrency/4B/longcontext/broad-quality/reference/repeat scope remain unchanged.

P168 live follow-up: original helper95279 warm-mtp-cuda-trace-startup-attempt1 started under a distinct controller correctness ledger and was authoritatively polled running. Previous original failure was terminal and archived, GPU85MiB/0% observed before launch; this is a changed collection experiment, never a restart due to timeout/quiet output. New source4cebf736f root methods retain the same app754/libraryba67 and explicit candidate0. Warm-interval verifier/terminal-only archive prepared and syntax-checked, not yet run on a startup trace. It requires unique ordered HtoD markers, complete14 exact outputs and terminal ledgers, then excludes cold/build rows before ranking. No new benchmark or hot-profile result yet; keep full goal active.

### P169 — startup trace failure independently verified; identical unprofiled control live;4B HTTP gates prepared

Original warm-mtp-cuda-trace-startup-attempt1 terminal1, underlying profile launch3221225477/0xC0000005. Native report retains12 exact warm-up cases then statuscapturing; zero completed capture cases. Failure-stats original handle completed0 and exported actual CUDA rows, but aggregate startup report includes cold/build/first-use work and cannot rank warm bottlenecks. After original terminal, GPU85MiB/0% observed and terminal-only archive finalized. No app source/DLL install occurred.

Archive warm-mtp-cuda-trace-startup-attempt1-evidence.zip9973616bytes/SHA2569d14bd5b66125b4ab25b2c5577168d6855a2d992edb0d17c6a2f7831ef6813888. Downloaded independent failure verifier PASSED CRC/all24 file hashes, linked archives,12 actual full-ID comparisons, terminal statuses, SQLite79438 kernel/145689 API rows. BOTH proposed35892/35908byte HtoD marker counts are0, so no marked warm window exists. Original strict-marker assertion (expected start1/end0) FAILED and its exact verifier/log are preserved separately; scoped failure verifier reports marker_gate_passed=false and admits no hot ranking. Source CreateGPU immediately issues a HostToDevice copy and synchronizes host-data lifetime; absence from a crash-truncated trace does not identify a marker or CUDA cause. No harmlessness/profiler-specific cause assumed.

Before another profiler experiment, started original helper95279 warm-mtp-unprofiled-control under its outer correctness ledger, with same root14-case probe, same DLLba67/source754, same launcher environment/markers but NO Nsight. It is currently live constructing. It will distinguish whether this observed probe failure reproduces without instrumentation; no result yet. NVIDIA official Get Started page https://developer.nvidia.com/nsight-systems/get-started currently lists2026.5.1 and explicitCUDA13.4 support; current2024.6.2 is preserved, no new tool installed or driver changed. Consider a separately preserved tool update only after control evidence.

Prepared native4B HTTP B1 server/driver using existing worker and pinned851bf6e... checkpoint, configSHA ddc63e1c..., retained beb80c6e... grid16 smoke archive. Gates: old text/image complete IDs twice each, four image semantics,16 full128 same-live repeat pairs, two actual SSE comparisons, THEN8 measured requests/cohort with matchedseed3/JPEG/grid16 and4096 capacity. No4B GPU HTTP run or speed result yet. Binary/source frozen while27B control lives; these are local/root study methods, not application-source synchronization. Reuse fixed study/ports after prior worker terminal only.

Also corrected future retained V3 driver --concurrency argument to four separate integer arguments, matching actual benchmark argparse nargs contract. Original archived failed method remains unchanged and never reached timing, so no published result changes. Existing real HTTP/stream-accounting CPU tests passed10/10; new4B methods syntax passed (GPU gates still required). Full2x faster-reference decode/trueGPUconcurrency objective, nativebatching,4B references, repeats/fullcontext/broadquality/tunedreference scope stay open.

P169 control follow-up: original warm-mtp-unprofiled-control completed0. The byte-identical native probe and launcher differing ONLY output directory completes all12 warm plus2 follow-up full128 text/image cases without Nsight. GPU85MiB/0% observed after natural closure. Archive warm-mtp-trace-unprofiled-control-evidence.zip175173bytes/SHA2569b55712ca377e6499ca06463c03ad30fbfb46f3661b695fe83b9bf1f312d39ae independently passes CRC/all23 file hashes, source/launcher equality to failed instrumented producer, linked archives,14 full-ID comparisons, common DLLba67, both terminal0 ledgers and raw lifetime sample maximum. Failure did not reproduce in this single unprofiled control; this distinguishes conditions but does not identify an internal cause or prove universal stability. No benchmark or GPU-marker-window admission.

Started original helper48577 nsight-2026-side-by-side-setup under setup ledger. Method downloads official NVIDIA2026.5.1.161 MSI into the existing fixed tools folder, validates MSI header/SHA256 and Valid NVIDIA Corporation Authenticode BEFORE execution, then uses administrative extraction /a /qn /norestart into tools/nsight-systems-2026.5.1. It does not globally install/replace the old profiler, change driver/CUDA runtime/application source, or restart. Free disk692720558080bytes observed before setup. CLI path/hash/version still pending; download/extraction is not trace usability evidence. Tool source/support reference is NVIDIA official Get Started page, explicitCUDA13.4 support. Setup has no GPU job;4B methods remain prepared but unexecuted. Keep full goal and2x requirements active.

P169 setup follow-up: original nsight-2026-side-by-side-setup terminal1. Official MSI downloaded663678976bytes/SHA256379c0a15a9cf7b8028081073fdd1d9798b9aaa618fb46c6a01785b69ed01d5f2. Authenticode VALID, signer NVIDIA Corporation/thumbprint7B7B0B6697AFB438CF6F65A155F00E86676FB186. Administrative /a extraction exited1603 and rolled back; no extracted files found. Raw UTF16 MSI log identifies Error1304 writing Linux support __init__.cpython-312.pyc under a260-character destination path. Path length is a plausible explanation, not an assumed fault-module diagnosis. No global installer fallback/reboot/driver change.

Started original helper48577 nsight-2026-short-path-extraction with SAME already-downloaded signed/hash-verified MSI and shorter fixed study tools/n26 destination (failing sample path becomes240characters), preserving original failure log. No duplicate download/model/clone; still administrative extraction only. Independent control verified14 cases already; no GPU worker currently scheduled, no tool usability or timing claim. CLI identity and actual tracing remain to be validated after terminal extraction.

P169 extraction guard correction: original nsight-2026-short-path-extraction exited1 at the package guard BEFORE msiexec. The manually duplicated digest literal had65 characters; authoritative download receipt/current package SHA256 agree and Valid NVIDIA signature is unchanged. Original failed method/ledger retained; paper-log fingerprint transcription corrected directly from downloaded JSON receipt. New extract-nsight-2026-short-fix.py compares actual bytes to that same signed-package receipt without a hand-copied duplicate, uses a new fixed results ledger and SAME unused tools/n26 extraction target. No duplicate download or global install. Do not interpret this guard failure as package tampering or another MSI extraction attempt.

### P170 — native4B HTTP correctness live; signed Nsight2026 Windows payload extracted

Original helper48577 native-4b-http-server and helper95279 native-4b-http-quality are LIVE, current app754cfa424/DLLba67dc9d frozen. Root worker uses existing native_4b_stream_worker.py, BF16 target-only B1/T16/cap4096/grid16. Driver now has explicit --quality-only; four old text/image repeat captures, four VLM semantic cases,16 full128 same-live repeat pairs and two real SSE gates precede timing. No4B timing yet. Source/launcher/old archive guards verified and methods committed; source is not synchronized while worker lives. Latest read had GPU10630516736bytes and no ready event, a construction observation only.

Profiler MSI extraction with shorter TARGETDIR still failed1603/Error1304 at a deep Windows Python recipe file. Read-only Windows Installer table inspection found public INSTALLDIR, but a separate /a attempt with that property also failed: actual MSI log resets INSTALLDIR to the nested company/product path. Both raw ledgers/logs preserved; no global install/reboot/driver change. Rather than repeat the same mechanism, mapped signed MSI File/Component/Directory/Media tables read-only and selected4039 Windows host/target/shared files,1046133602bytes. Used existing7Zip26 to extract the four authenticated embedded CAB streams and selected file keys into a fixed staging folder, then guarded NONRECURSIVE per-file moves into tools/n26 with exact MSI FileSize and SHA256, retaining complete host/target relative layout. No payload path may escape fixed tools target, and no existing files are overwritten. Original MSI/source hash/authentication stays authoritative, no duplicate package download.

Original helper48577 nsight-2026-windows-payload-extraction completed0. All4039 planned files placed/hashed; target-windows-x64/nsys.exe Authenticode VALID/NVIDIA Corporation and actual --version reports NVIDIA Nsight Systems2026.5.1.161-265138896106v0. Setup receipts contain actual executable/package/plan hashes (do not transcribe from wrapped PTY). This is tool payload/version qualification only: no new CUDA trace or model speed result yet. Old2024 profiler remains intact. New4B timing wrapper additionally refuses to run until actual tool-extraction and quality ledgers are terminal, preserving host timing isolation; then requires unchanged qualified artifacts/ready profile. Prepared benchmark wrapper remains unused here.

Local4080 quiet window was released by owner, then a new independent component window reserved through14:00UTC Oct24; this task continues remote5090/light CPU only, with no local compiler/nativeGPU/sharedSDK changes. Entire2x decode/trueGPUconcurrency plus4B references, repeated/fullcontext/broad quality/tuned reference scope remains open.

### P171 — native4B matched-content B1 pilot independently verified; new-profiler full trace started

Original native-4b-http-quality completed0:4 old text/image repeat captures,4 image semantic checks,16 full128 same-live repeat pairs and2 actual SSE gates. Only after CPU/IO tool extraction original handle terminal0 did native-4b-http-benchmark start; it completed0. Both cohorts8requests/128outputs,seed3,JPEG/binary-grid,image-first,IPv4 loopback. Batch1/T16 BF16 target-only,4096capacity; actual prompts41/104, not4K-input benchmarks. Source754cfa424/DLLba67dc9d4f8b9948615e522e3b6ad0aa10ea6f742505465084449fe57ead0b8b.

text: TTFTmedian=63.23539999721106ms,decodemedian=85.83239937140999tok/s,aggregate=82.61979428429088tok/s,sampledAdapterPeak=11335159808bytes,maxSampleGap=0.04804629999853205s.

image: TTFTmedian=119.13459999777842ms,decodemedian=85.7469673973019tok/s,aggregate=79.62750501985836tok/s,sampledAdapterPeak=11337256960bytes,maxSampleGap=0.03535879999981262s.

After actual original quality/benchmark handles terminal0, guarded exact4B bridge-tree stop; original server terminal1 intentional, resource wrapper finished, GPU85MiB/0% observed. Archive native-4b-http-evidence.zip1027121bytes/SHA2562058e289c726fc680b925c8cd7f4f1fb0fc712551b3449c51fd7fbf429f3a856. Independent verifier PASSED CRC/all183 file hashes, retained smoke/config/image linkage,4 semantic cases,16 repeated full sequences,2 SSE gates,16 measured rawSSE/client-accounting/fullworkerIDs and current greedy IDs, actual image hashes, both cohort sampled maxima and lifetime monitor. All16 normalized request payloads (only model alias differs) AND complete prompt IDs match the prior27B pilot; generated output quality across different models is not equated. No4B vLLM/SGLang reference or concurrency/full-context/broad-quality qualification yet.

New descriptive chart and exact plotting/verification scripts retained. Comparison is separate pilots/different architectures, precision, speculation, chunk profiles and binaries:27B mixedNVFP4/FP8 joinedMTP3 T128/T4 versus4B BF16 greedy T16/T16. Do not claim a causal model-size/precision gain, cross-model quality parity, repeated confidence or fulfillment of27B2x. Original27B2x decode/trueGPUconcurrency target remains unmet.

New Nsight2026 bounded CUDA smoke original handle completed0; actual raw statistics contain5 normalization kernels and CUDA/graph API records, native norm graph capture1/replay2/failures0. Preserved tool setup failure logs, authenticated payload map/signature/hash/version receipts and raw smoke trace in nsight-cuda-smoke-2026-attempt1-evidence.zip; independent offline verification pending at this log entry. Tool binary payload remains in fixed remote tools/n26.

Started original helper48577 warm-mtp-cuda-trace-2026-attempt1 under distinct controller ledger. Same native probe bytes as successful14-case unprofiled control, same app754/DLLba67/candidate0, only profiler2026.5.1 and fresh fixed results/launcher differ. Native4B worker is already closed; one GPU study job only. Source frozen. Twelve exact warm-up cases and two exact capture cases plus unique ordered HtoD markers are required before any hot ranking. Profiler overhead remains diagnostic, never a new HTTP speed point.

P171 new-tool smoke independent follow-up: downloaded nsight-cuda-smoke-2026-attempt1-evidence.zip4132190bytes/SHA256cc37ade5bd19ba07aa35e93172b88ddcfd2d20e3b38f5ff02cccb1da4a615b9f. Verifier PASSED CRC/all49 file hashes,4B archive linkage, signed payload/CLI receipt consistency and4039-entry producer hash manifest, native report/library identity, terminal profile/stats ledgers and direct read-only SQLite validation:5 actual CUDA kernel records, nonzero CUDA API records, native graph capture1/replays2/failures0. Kernel-duration sum also matches the raw CSV report, but is ONLY this cold/build instrumentation smoke, not a model latency claim. Tool binary bytes remain remote; offline signature/payload entries are producer receipts. Original helper48577 full2026 trace remains live; its result is not inferred from the bounded smoke. Local-only plotting/JSON verification uses no compiler/GPU; new local4080 quiet window through14:10UTC respected and subsequently owner released14:00 window as recorded.

### P172 — new-profiler warm trace independently verified; resident controls exactness experiment

Original helper48577 warm-mtp-cuda-trace-2026-attempt1 completed0. Archived all24 raw artifacts in warm-mtp-cuda-trace-2026-attempt1-evidence.zip,53573617bytes/SHA2566d8662718341ffb9e5580ff41a263458aae0e7780e9577a596ce211d87f81dec. Separate verifier passed remotely and offline locally: CRC/all24 SHA256, linked prior archives, frozen source754/DLLba67,12 warm cases plus2 complete128-token capture cases exact against retained target captures, terminal profile/stats/controller ledgers, unique ordered35892/35908byte HtoD markers and no kernel crossing. Read-only SQLite selected195280 warm kernel records (192387 graph nodes),138963 API records,15255 copy records from627331 full-trace kernels. Warm marker interval219365865751..223477384467ns; kernel duration sum3069060475ns and union3046528333ns. These are instrumented diagnostics, NOT HTTP/model speed or2x evidence; window includes warm text/image prefill and generation, not decode alone.

Collected kernel ranking (sum durations overlap and profiler overhead applies):
- cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128x128x256_1x1x1_0_tnn_align32_o_vs16_bias_bf16_relu: 11654 launches, 1019.158589ms, 33.208% of collected kernel-duration sum.
- sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize32x64x64_stage5_warpsize2x2x1_tensor16x8x32_execute_kernel__5x_cublas: 16512 launches, 399.004285ms, 13.001% of collected kernel-duration sum.
- void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params): 10218 launches, 339.934444ms, 11.076% of collected kernel-duration sum.
- Garnet::<unnamed>::recurrence(Garnet::Qwen35DeltaConfig, int, const float *, const float *, const float *, const float *, const float *, const float *, const float *, const float *, const float *, float *, float *): 6384 launches, 278.255740ms, 9.066% of collected kernel-duration sum.
- Garnet::<unnamed>::QuantizeActivation(const __nv_bfloat16 *, unsigned char *, unsigned char *, float, int, int, int, unsigned long long): 11654 launches, 216.547550ms, 7.056% of collected kernel-duration sum.
- <unnamed>::logitsTop1BatchBF16Kernel(const __nv_bfloat16 *, long long *, float *, int): 390 launches, 121.084661ms, 3.945% of collected kernel-duration sum.

Host API diagnostic: cudaFree_v3020 count2816 sum2179.369825ms; cudaMemcpyAsync_v3020 count3610 sum719.555148ms; cudaGraphLaunch_v10000 count566 sum352.198114ms; cudaStreamSynchronize_v3020 count2534 sum105.117984ms. These API durations overlap GPU execution and cannot be added or interpreted as recoverable speedup. Collection coverage has not yet been proved operation-by-operation.

Source inspection finds existing default-off resident_controls paths in cuda_speculative.py/cuda_verify.py/cuda_hybrid.py. Legacy embedding(ids) creates temporary GPU outputs; resident path instead uses embedding_chunk/update_request with inline CUDA argument payloads. This is a candidate for reducing allocation/copy/synchronization, not proven attribution of every profiler record. Prepare and launch warm-mtp-resident-controls-attempt1 with unchanged754/ba67 app, candidateNVFP4 selectorOFF, no profiler. First runs existing transport oracle across B1/2/4/8,T4/16/64/128,3trials (48total) with bitwise controls/embedding/padding/CPU argument lifetime and invalid-input no-mutation. Only transport pass admits same14-case model producer with sole decoder.resident_controls=True change, including text/image full128 and4 image semantic checks. Separate fixed results/controller/resource ledgers, source frozen. This is correctness only; no HTTP/performance/default promotion.

2x matched27B decode and trueGPU concurrency target remains unmet;4B references, full-context/broad-quality/repeated/tuned references and native batching still open. Local4080 owner released quiet window early; no local native/compiler/GPU work was needed here.

P172 supplementary operation-count check PASSED: pinned source-derived geometry and exact two capture-case counters predict11654 NVFP4 matmul,11654 quantizer,10218 small-tile BF16 GEMM and6384 GDN recurrence launches; collected counts match all four. Inputs41/104 each require one T128 target prefill; target calls88 including prefill, MTP body295, shared head390 including prefill and45 prefix replays. Partial-family coverage check only, not proof of every CUDA record or timing fidelity. Original resident-controls job reports transport16profiles/48trials completed and remains running in model phase; no performance admission.

### P173 — resident-control transport passes; HTTP opt-in wiring prepared behind gates

Original helper48577 warm-mtp-resident-controls-attempt1 remains live (authoritative handle polled, not inferred from files). Transport stage terminal0,16 batch/time profiles with3trials=48producer assertion trials; full-model report still constructing when inspected. Live resource monitor shows~26.7GB adapter use and actual xlang worker CPU progress. No restart, rebuild or remote source sync; old frozen754/ba67 remains authoritative for this experiment. Extra unrelated older xlang processes were observed but not terminated.

Prepared immutable success/failure archiver and offline verifier for candidate evidence, linking old unprofiled control/off20-golds/new2026 trace. Verifier requires exact probe diff limited to resident_controls=True plus reporting,14 full outputs/request+teacher hashes, known DLL/source, unchanged shared source bytes, transport16profile/48trial producer assertions and both original stage resource/terminal ledgers. Actual transport arrays are not retained; document assertion scope, do not call them independently replayed tensors.

Locally added research-worker environment flag GARNET_STUDY_RESIDENT_CONTROLS (defaultOFF), restricted before GPU initialization to shared joined MTP3/head-reuseOFF. It sets decoder.resident_controls and reports resident_request_controls in ready health. Prepared separate root launcher/server/workflow/candidate methods; they are NOT launched or synced into active app. Server requires independent14-case proof plus same ba67 library and GPU/port idle. HTTP quality requires20 target/MTP pairs,20 old-qualified full identities, retained6profile goldens and2 actual SSE gates before B1 timing. No queued HTTP-load test is substituted for trueCUDA batching. Same-binary repeated A/B still required for an optimization claim.

Local profile/HTTP bridge/MTP event CPU suite13tests passed after setting documented source import path. Initial aggregate invocation without that path failed one module import (native_http_bridge missing); no implementation failure was hidden. New methods compile under CPython; Xlang/GPU execution remains pending full correctness completion. No new timing or2x claim.

P173 terminal follow-up: original helper48577 warm-mtp-resident-controls-attempt1 completed0,14cases exact and all48transport assertion trials passed. Guarded archiver required actual terminal outer ledger and GPU idle. Downloaded warm-mtp-resident-controls-attempt1-evidence.zip187547bytes/33files/SHA256802c8ab7dec50edb7984c19f0a4cd9d1ebf9af276e7fd00f43d4adaf9e565ba4. Offline verifier PASSED all33file hashes/CRC/linked archives, exact producer changes, common frozen754/ba67 source/runtime, all14 complete outputs/request+teacher identities/semantic controls, unchanged common source files,16profile transport producer assertion records and two terminal resource monitors. Retained verifiedJSON and original logs. No timing inferred.

Guarded source-only HTTP install completed0 after original GPU worker closed. Prechecked both original worker/profile file SHA256 against saved local899265bcc versions before overwrite; copied originals into results/resident-controls-http-source-install/backup. Receipt records package/new/old hashes, library remainsba67. Only two Python research-worker files plus root study methods/proof were installed, no native binary,driver,model,compiler or defaults altered. Remote Git source remains754 plus explicitly archived two-file patch; do not describe it as clean754. Installer/ZIP/receipt/backups retained in fixed study root/results.

Started original helper95279 native-resident-controls-http-server and helper48577 native-resident-controls-http-workflow. Workflow first waits for actual ready health resident_request_controls=True and joined336entry mixed pool, then20full target/MTP pairs; candidate separately checks20retained qualified identities and6profile goldens, then2SSE gates before B1 timing. Source frozen while live. B1 timing is a pilot, not same-binary repeated A/B, trueCUDA batching, broad-quality/fullcontext or2x proof. Original target remains unmet.

### P174 — resident controls B1 HTTP pilot independently verified; planned four-leg A/B/B/A

Original helper48577 native-resident-controls-http-workflow completed0.40 captured responses establish20currenttarget/MTP pairs, all20old-qualified identities, retained6same-profile goldens/4image semantic cases and2actual SSE gates before16 measured requests (8percohort;128actualoutputs;41/104promptIDs;capacity4096). No profiling during timing. Original owned bridge stopped only after workflow observed terminal; actual server95279 completed1 intentionally and whole-lifetime monitor saved. No app source or library changed while live.

Archive native-resident-controls-http-evidence.zip638016bytes/199files/SHA2566d8c4e7049a9bafe7df3c4d91963f96a02915860ac5d2b0554e046666589f708. Offline verifier PASSED CRC/all199file hashes, linked oldarchives, common ba67DLL/pinnedXlang identity, exact reviewed worker/harness bytes, installation package/oldbackup/newhash receipts,20full retained pairs/6goldens,16actual measured SSE streams and prompt/output IDs/modes/counters,2stream gates and NVML cohort/lifetime maxima, controller/server terminal states and guarded stop. Independently reaccounted pinned ordinary vLLM/SGLang archive hashes and raw SSE/counts. No new quality equivalence with references is inferred.

text: TTFTmedian73.81760000134818ms,decodemedian73.92196190968554tok/s,aggregate71.11760701310129,sampledpeak28892516352B,maxsamplegap0.18967029999475926s.
image: TTFTmedian104.93154999858234ms,decodemedian76.12551135754535tok/s,aggregate67.43298281002573,sampledpeak28896710656B,maxsamplegap0.29792409999936353s.

Provisional B1 decode ratios to faster ordinary reference: text1.0794383379541284,image1.1113608050916315; TTFT ratios0.4844909611242147/0.6709879837132877. These separate single pilots/binaries cannot establish causal gain, tuned-reference superiority or2x. New chart inspected with raw values and explicit unmet2x line. New27B sampled peak~26.9GiB is adapter-wide; do not attribute difference from prior pilot to the control option without controlled evidence. Source remains copied754 plus reviewed2-file Python patch; same nativeba67 binary. DefaultsOFF.

Prepared fresh controlled A/B/B/A sequence a1(false),b1(true),b2(true),a2(false). Prior exploratory pilot excluded. Prespecified8requests/cohort/2warmups, exactsame corpus/precision/MTP3/T128/T4/cap4096 and same native binary/source hashes. Each new worker must pass20self-target pairs+20retained identities+2SSE gates before its own B1 pilot. Original server handle/terminal workflows govern owned shutdown, no timeout restart; archive each leg immediately including failures, stop sequence on any workflow failure. GPU telemetry before/after legs retained without collecting it inside timed requests. Source frozen for entire sequence. Finite fixed study results, no clone/weight duplication. Generated24methods+plan+manifest are reviewable from committed builder: all26raw package members independently reproduced byteexact from source templates. ZIP timestamps alone are not claimed reproducible. Root controller/methods ZIP retained under fixed remote study root.

Started original helper48577 native-controls-abba-attempt1 under native-controls-abba-controller-ledger. Full4-leg evidence must be independently replayed and linked before gain confidence; source/quality/temperature/variation limitations remain. Local tree-state study requested quiet window through14:30UTC; only local CPU plotting/archive/source work performed, no local compiler/GPU/sharedSDK mutation. Original2x decoded/trueGPUconcurrency,4B references, nativebatching, broader/fullcontext quality and tuned/repeated references remain open.

### P175 — first two ABBA legs verified; actual layer0 stage diagnostic prepared

Original48577 native-controls-abba-attempt1 remains live: a1OFF and b1ON workflows completed0 and were archived after identity-guarded bridge shutdown/server terminal1; b2 is running. Downloaded both immutable archives. Independent verification PASSED all199file hashes/CRC/source/runtime/20target-MTP+20oldID+6golden+2SSE+16measuredSSE/VRAM/terminal gates per leg. Raw metrics:
[
  {
    "leg": "a1",
    "sha256": "da0dbf3d2307d6f1e4ccefccbc35b208ea0d58f40688dad38d3301c460133f65",
    "cohorts": [
      {
        "case": "text",
        "ttft_median_ms": 80.69479999903706,
        "decode_median_tokens_per_second": 67.40036169801411,
        "output_tokens_per_second": 65.23517669807451,
        "sampled_peak_adapter_used_bytes": 28405977088,
        "observed_sample_gap_max_seconds": 0.35044960000232095
      },
      {
        "case": "image",
        "ttft_median_ms": 117.83205000028829,
        "decode_median_tokens_per_second": 70.43828545536203,
        "output_tokens_per_second": 30.59310539734861,
        "sampled_peak_adapter_used_bytes": 28412268544,
        "observed_sample_gap_max_seconds": 3.5748804999966524
      }
    ]
  },
  {
    "leg": "b1",
    "sha256": "36c42f4c5bc4255c1a30ad7d947e08894671d1f239f4379b358bab0bb36a4870",
    "cohorts": [
      {
        "case": "text",
        "ttft_median_ms": 87.50764999786043,
        "decode_median_tokens_per_second": 68.95736189441553,
        "output_tokens_per_second": 36.82144208653321,
        "sampled_peak_adapter_used_bytes": 28892516352,
        "observed_sample_gap_max_seconds": 8.306386099997326
      },
      {
        "case": "image",
        "ttft_median_ms": 110.61675000019022,
        "decode_median_tokens_per_second": 77.07152733636434,
        "output_tokens_per_second": 69.89740480763848,
        "sampled_peak_adapter_used_bytes": 28896710656,
        "observed_sample_gap_max_seconds": 0.0948156000013114
      }
    ]
  }
]

Pauses retained without exclusions: a1 image r5 elapsed16.25505400000111s (decode12.055322899999737s,tail4.070352400005504s),r7 elapsed5.530732300001546s. b1 text max sampler gap8.306386099997326s; its aggregate36.82144208653321 versus median68.95736189441553 shows why a median alone is insufficient. Reproducible raw-gap observer added. Causes unproven; both modes show pauses. Every request remains in metrics; planned4leg order continues unchanged. No causal/control-gain or2x claim. If pauses persist, separate worker/bridge/client/logging/driver boundaries before selecting another performance change. Current client/bridge flush event logs; no evidence yet that logging causes pauses. Any future measurement protocol change needs matching reference reruns.

Initial a1 verifier failed because shared installation-receipt root B0 method files are inside authenticated install ZIP, not separately copied into each leg archive. Strict verifier and failure log retained. Corrected replay verifies every installed payload hash inside ZIP, requires actual application files separately present/equal, verifies executed leg methods from metadata and preserves original source backups. No quality/SSE/output gate relaxed. Corrected a1,b1 pass and original B0 pilot re-verifies unchanged. Parent ABBA archiver/verifier prepared to require original terminal0/GPU idle, exact order/linked4leg hashes/26-member planned methods/source/runtime/workload and perleg independent reaccounting, ranges/means/mirrored ratios and before/after telemetry. Parent not executed to claim unfinished legs.

Prepared default-off bounded diagnostic_linear_stages native keyword plus real-activation witness probe, detailed in qwen38-5090-layer0-stage-protocol.md. Actual layer0 graph stage tensors are appended only under explicit mixed27B embedded B1/2 T16 guard; normal output contract unchanged. Probe saves full ordinary/diagnostic primary and11stages from real first16 prompt embedding IDs, with complete state,2repeats and duplicate rows before attribution. Four invalid-profile rejection checks and3NVFP4/3FP8/zeroINT4/output-size/graph-count checks prepared, UNRUN. Historical170MB prefix archive pinned32586aea53c0fd77b1d96935f700abe0061bd24f2cc1915400454929aac10564 supplies actual hidden/state witnesses; current primary must equal ordinary before stages are interpreted. Does not repair/admit full64-layer batching or establish speed.

After local quiet-window early release, CPU-only MSVC17.14 /Zs syntax check of changed actual trt_builder.cpp against CUDA13.2/TensorRT10.16.1.11 headers completed0. No codegen/link/GPU/remote source change. Existing object SHA9761c041b96c00b7922ef32411bbe29aaa3b1c491e3102c947b2c1b1b0d5edb0 unchanged. Initial return-code recorder used a relative slash path and failed before compiler; corrected absolute-path recorder/log retained. Probe/setup Python syntax passes. CPU evidence ZIP79907bytes/12files/SHA c5434213bdbd9c506e26e38a1a85be432122dc16aafc18517becbcd3fde586c6 verified CRC/allSHA. Native build/deployment and all GPU diagnostic gates remain pending; remote754+2Python/ba67 frozen throughout ABBA. A new local quiet window through15:00UTC was requested later; no compiler/GPU remains here.

Source hypothesis only: inline RequestPayload2560B/EmbeddingPayload4096B by-value dynamic access may merit deployed-kernel local-memory inspection. NVIDIA documents const __grid_constant__ preventing per-thread parameter copies (https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html). This does NOT prove spilling here or explain extra~0.5GB. After ABBA, inspect actual deployed DLL resource/SASS/attributes before changing kernels. No core IO source/policy changed from this hypothesis.

Original2x matched decode/trueGPUconcurrency is unmet. Remaining scope includes all4leg verification, pause diagnosis, actual batching mismatch/stage witnesses,4B references, fullcontext/broader quality and tuned/repeated references.

P175 b2 follow-up: original controller reports b2 workflow0, owned server1 and immutable leg archive; a2 is now live. Downloaded b2 archive640665bytes/SHA256e521b14bd6eb9d3033e0ab6c3ac4656066adc523e6099962c461a1e3ddabd2d3 and independently PASSED all199 hashes/CRC/quality/oldIDs/measuredSSE/VRAM/source/runtime/terminal gates, with raw-gap audit.
text: TTFT65.16929999997956ms,decode75.68652265915196tok/s,aggregate72.4774817630914,peak28892516352B,maxgap0.03386740000132704s.
image: TTFT112.24325000148383ms,decode77.69713612612179tok/s,aggregate71.17121619166068,peak28896710656B,maxgap0.035903800002415664s.
Third leg verified does not complete the four-leg comparison. First/second ON variation and earlier pauses remain in all results; no data excluded, no gain/default promotion or2x claim. Native diagnostic remains unbuilt/undeployed; local quiet window now through15:00UTC respected after completed earlier syntax-only check.

### P176 — four-leg comparison verified; resource hypothesis rejected; native diagnostic built

Original48577 native-controls-abba-attempt1 completed0. All4leg workflows0, owned shutdowns/server1, independent full parent/leg replay PASSED. Parent archive41399bytes/21files/SHA256a09e3f0587b7cf252b2f60bf26df6d3e52642504f09f1496dbf629877b6d7245 links all4immutable199-file leg archives; parent verifies planned OFF/ON/ON/OFF order, original package/plan, common archived application hashes (not every external OS file), same runtime, payloads/IDs, perleg20self-target+20oldID+6goldens+2SSE+16measuredSSE and NVML. Prior pilot excluded, no requests excluded. New control chart inspected, with medians AND cohort throughput and 2x-reference target lines. Temporary plotting-source CP936 read caused mojibake; repaired from original UTF8 and final render inspected, no measured data changed.

text-c1:
decode_median_tokens_per_second: OFFmean=67.75931756657505,ONmean=72.32194227678374,ON/OFF=1.0673357535770016,mirrored=[1.0231007691528056, 1.1111045368937993].
ttft_median_ms: OFFmean=73.42092500039143,ONmean=76.33847499891999,ON/OFF=1.0397373092005175,mirrored=[1.0844273732496352, 0.9852185395759834].
output_tokens_per_second: OFFmean=65.29020709471214,ONmean=54.64946192481231,ON/OFF=0.837023871673986,mirrored=[0.5644415168361158, 1.109147116845138].

image-c1:
decode_median_tokens_per_second: OFFmean=70.093937667842,ONmean=77.38433173124307,ON/OFF=1.104008910127841,mirrored=[1.0941709730457012, 1.1139439853257405].
ttft_median_ms: OFFmean=120.11632500070846,ONmean=111.43000000083703,ON/OFF=0.9276840595995574,mirrored=[0.9387662355014581, 0.9170155211694131].
output_tokens_per_second: OFFmean=47.207448104868625,ONmean=70.53431049964958,ON/OFF=1.4941352123707614,mirrored=[2.284743699595015, 1.1151554239660322].

Observed decode means+6.7%text/+10.4%image, but text throughput worse on average and pauses affect both modes. Only2runs/mode, raw ranges/pairs retained; no broad confidence/default promotion/causal2x claim. Before/after telemetry: driver617.14,powerlimit600W,SM2977MHz/memory14001MHz after initial idle,40..52C; not continuous timed-clock proof. TrueGPUconcurrency/tuned refs/broad/fullcontext quality remain open.

Offline actual deployed ba67DLL CUDA metadata collected with locally NVIDIA-signed CUDA13.2 cuobjdump (toolSHA7405a7a6f0c64bc7d962bb5cfc30ce24d122c8c6ac6d89c0636f12f60bf3c6fe). Resource tool terminal0, initial parser controller1 because actual format is Function NAME: rather than Function: NAME. Raw failure retained, separate corrected parser0 reuses exact raw tool output, no rerun. Independent CRC/all14fileSHA/toolbytes/terminal/raw line-state replay PASSED archive6064929bytes/SHA1aac0ae7f6c44d64c69bd5fdcd0046cb644d0cc1ff075a03fadb5c2b4b44bf21. Actual sm120 GatherChunk and WriteRequest both LOCAL0/STACK0,constant0 sizes5024/3488. CMake cache actual architecture120. This does not support proposed per-thread parameter spill explanation; no __grid_constant__ code change. Extra~0.5GB and pauses remain unexplained, static resources not total-VRAM attribution. DLL bytes remain remote, artifact hashes link original input; tool bytes present in archive, remote Authenticode a retained receipt.

After ABBA/runtime workers terminal and GPUidle, guarded source install0 preverified original80409e5bb file hashes, preserved native header/builder and ba67DLL (build/bin/garnet-before-layer0-stages.dll), installed only2native files+2study probes from source ZIP. Old unrelated Xlang processes are D:/CantorInstall/WorldSense, left untouched. No study Xlang worker or competing GPU allocation before install. Original48577 layer0-stage-native-build-attempt1 completed0: MSVC17.14.42, one trt_builder.cpp object/device link/final DLL; vendor stagingOFF, fixed build slot reused. New unqualifiedDLL34cb221458064a4b7264586412be5b3c7acca888cd9e74681ac2a41d259413da retained additionally as garnet-layer0-stage-attempt1.dll. Source-install receiptSHA51319b65a3d839f52c4bc0751d7b247e9bf650ee19fa50311d9753cf72031ca9. Build is not numerical/serving/performance admission. Remote source now copied754+earlier2Python+2native/study patch; do not call clean754 or benchmark newDLL without default full-model gates.

### P177 — first stage probe fails test setup; corrected actual inputs and completed warmups

Original48577 layer0-real-stages-attempt1 completed1, monitor saved and workerclosed. Historical six real witnesses extracted from pinned32586a.. archive. Four invalid profiles(flag2,-1,B4,T4) correctly rejected; B1 engine has3NVFP4/3FP8/zeroINT4 and correct7051264 output bytes. First test failed strict graph gate:2contexts,1capture,0replays. Source TRTContextPool::Acquire shows an unfinished asynchronous warmup can open a second context; ordinary run_device reuses owned output storage, so output reference retention alone is NOT established as cause. Raw26B1 arrays saved; no B2/stage attribution admitted.

Independent source inspection also found my setup error: CUDAPackedDecoder.chunk_hidden copies output into decoder.hidden, so the test cloned output as next input. RPC byte check confirms both primary-vs-ordinary comparisons fail. This was not a backend correctness finding. Original source/build/26raw arrays/ledgers and witness linkage preserved in layer0-real-stage-attempt1-evidence.zip18110031bytes/61files/SHA65cb85f337c943a1025fe67f6201bb5a150087fea15a68b06d4723d153cccfc1 (downloaded; offline failure replay pending at this entry). No original records/gates overwritten or waived.

Guarded study-probe fix install0 only AFTER original worker terminal/GPUidle and immutable failure archive. Original probe backed up with old/new hashes; native DLL stays34cb. New probe constructs fresh independent input from original prompt embedding token rows and saves prompt_input; requires exported masked_input exactly matches it. Three explicit stage warmups each copy/read a bounded word to host before next enqueue, recording boundary hashes; two retained stage repeats and all primary/negative/row/projection/output/graph gates remain. Results/temporary handles released after copies. No native rebuild, no timing claim. Original six witnesses reused with hashes. Attempt2 controller checks native source against build, changed study probe against separate fix receipt, actual library, old failure archive, input witness hashes.

Started original48577 layer0-real-stages-attempt2 under layer0-real-stage-attempt2-controller-ledger. Last live poll shows ordinary graph startup for both profiles; no completed result claimed. Frozen source/runtime while this GPU worker lives. Current native CPP/header unchanged since successful build, only corrected study probe differed under receipt. Next: observe terminal originalhandle, archive before any further mutation, independently verify actual prompts/primary/repeats/all arrays before finding first cross-batch stage difference. Default full27B numerical gates remain required for rebuiltDLL, and full64layer B2/true concurrency admission remains unresolved.

Local coordination windows released then renewed through14:55UTC and released early; no local native compiler/GPU jobs during those windows. All work here after completed syntax check was local CPU analysis/plot/archive and remote5090. User2x fullobjective remains unmet/unblocked/active;4Brefs, fullcontext/broader quality,tuned/repeated refs and actual batching remain.


### P178 — actual layer0 stages independently replayed; real FP8 projection fixture

Original48577 layer0-real-stages-attempt2 completed0, worker closed. Immutable archive52716641bytes/106files/SHA256d05aa555feac3183d54dd7c1975741fc3bb525739eccff26a98166b5db761a99 downloaded and independently replayed locally with NumPy. Separate remote CPython replay additionally read the actual BF16 embedding rows directly from pinned checkpoint shard: input F32 SHA256f28e82d6bd01f4867adaaedc8626bfd72ddbc4f0b4f6afdf0046967b571da5ff. All81 raw arrays finite/size/hash verified; entire diagnostic primary equals ordinary hidden+conv+recurrent state for both profiles/all rows/two repeats. Ordinary and exported outputs match historical witnesses exactly. Negative-profile/count/output/graph/repeat/duplicate-row/resource/terminal gates pass. No gate waived. Failed first-attempt source remains linked under immutable archive65cb85f3... and fix receipt.

Actual cross-profile masked_input,input_norm,qkv,z,beta,a,gdn_gated_output are bitwise identical. First observable difference is attention_output:9/81920F32words,maxabs3.0517578125e-5,indices4396,20682,27150,29993,56154,61987,62873,66615,77442. Residual and output_hidden retain these9changed words; complete ordinary packed909312words differ at those9hidden positions only, state exact. This localizes the layer0 FP8 out-projection path (cast/quantize/matmul/restore/select), not a unique cuBLAS algorithm cause. Default full27B rebuiltDLL qualification and full64-layer B2 gates remain open.

CPU-only actual-layer0-fp8-fixture-attempt1 completed0. Fixed results/actual-layer0-fp8-fixture contains real common gated F32 input (M16/M32), archived expected BF16 outputs, actual layer0 out_proj checkpoint E4M3 weight [5120,6144] plus scalar input/weight scales. Source archive, independent verification, safetensors index/header/offsets, exact bytes and hashes retained in manifest. Bounded31.46MB weight extraction, no new checkpoint copy. Scales0.0322265625 and0.003016880713403225. Standalone untimed native control prepared using unchanged RunModelOptFP8Matmul and QuantizeModelOptFP8Activation, F32-to-BF16 RNE cast,4MiB workspace. Must exactly reproduce all archived M16/M32 output words twice before candidate algorithms or timing. Records selected algorithm opaque bytes; preserves cast,quantized,and output arrays on failure. No native application math/source/default change. Full2x/trueGPUconcurrency objective still unmet and active.


P178 projection follow-up: original48577 actual-layer0-fp8-control-attempt1 completed0 (build0/native0), then workerclosed/GPUidle. Archive3209591bytes/31files/SHA2561cfa7c21bb9e65fc02c0262079d540b472dd9117f34f405c36b9f8efca9eb308 includes native executable, all8 source snapshots, cast/quantized/expected/actual outputs, both ledgers and fixture/independent stage manifests. Actual31.46MB checkpoint weight stays in fixed remote fixture with hash/header/offset provenance, explicitly outside archive. Independent local CRC/all31SHA/source/exe/ledger/stage-linkage replay passed. Native unchanged FP8 helper exactly reproduces archived M16 and M32 outputs, both repeats. Independent NumPy BF16 RNE and finite E4M3 nearest-even quantization reproduce native saved arrays; both profiles' per-row cast and packed FP8 activations are exactly identical. Native output differs at same9indices. cuBLASLt130400 selects ID35/M16 and ID67/M32 with4MiB workspace (same per repeat, raw opaque bytes retained). This reproduces the actual failing projection outside TensorRT and excludes a cast/quantization input difference for this witness. An alternate-algorithm intervention is still required before unique-cause/candidate claims. Fullmodel batch and2x remain unqualified.

Prepared fresh-output default full27B14-case regression for new34cbDLL using byte-identical old unprofiled producer, candidate0/profileOFF and original archive/source guards. It must preserve all old14outputs; broader20oldIDs/HTTP/SSE/fullB2 gates still needed. No app source sync/rebuild while any GPU worker is alive. Research methods through independent stage replay pushed as f72fc8585 on qwen38-5090.


### P179 — checked shared algorithm removes real projection discrepancy; native candidate isolated

Previous goal turn made progress: independent106-file stage/real checkpoint-input replay,31-file native real-projection replay and pushed reviewable methods (aefb7aaf3). Original48577 layer0-build-full-control-attempt1 now completed0; workerclosed/GPU85MiB. Default34cbDLL14-case producer byte-identical to prior successful unprofiled control. Archive165412bytes/23files/SHA2564175e4a45c34f1e57cb3cc8b0932b9301a2dca470889a553cae51fc1f71c785a independently replayed CRC/allSHA/source/requests/teacherhash/all output IDs/resource/terminal. Six128-output text/image cases and eight2-output semantic cases match old14exactly. Initial local verifier wrongly assumed all14cases128outputs; next assumed capture cases contain repeat. Both local failing verifier versions/receipts preserved, corrected assertions enforce actual old/new schema/key-set equality, case lengths and12warmup+2capture order; no producer/runtime/array change or gate waiver. This is untimed regression, not broad20ID/HTTP/SSE/fullB2/performance admission.

Original48577 actual-layer0-fp8-shared-algorithm-attempt1 completed0, build/control/shared all0. Byte/array evidence archive3226211bytes/33files/SHA2562d1061813fd425d6a01c3c59836d06e01d09b9e487723c6ffb03f3678294b364 independently replayed locally. Same executable first reproduces both archived M16/M32 outputs exactly twice with IDs35/67. Intervention saves actual successful M16 algorithm opaque state and applies it to actual M32 only after cuBLASLt AlgoCheck success/workspace<=4MiB. Selected opaque algorithm equals canonical M16 on all4calls. Both M32 duplicate rows match canonical M16 output bitwise twice; F32/BF16/FP8 inputs unchanged. Thus an algorithm intervention removes the9word discrepancy for this actual layer0 projection. Not a proof for other matrices, full64-layer batches, numerical quality or speed.

Prepared native defaultOFF research option GARNET_FP8_GDN_M32_CANONICAL=1: only physicalM32,N5120,K6144 FP8 GDN out projections. Native helper reconstructs M16 heuristic, requires ID35 and actualM32 AlgoCheck, SM12.0/cuBLASLt130400/4MiB workspace; no silent fallback. Other profiles/default use original path. FP8 plugin preserves default16-byte wire; optional20-byte policy1 survives clone/deserialization; unknown/truncated policy or wrong geometry/profile rejected. Lowering accepts only0/1 policy values. Full application source/DLL remain unchanged remotely. New code isolated in fixed results/native-canonical-fp8-source, package19457bytes/SHA2568b34a3f2c992a8ecf526fcb1963cb9753085852c520eb1fd31e0540ff0ed6b7d. Original48577 native-canonical-fp8-attempt1 started: standalone native helper and plugin-policy compilation/gates before any app installation. No candidate correctness/integration claim until terminal and independent raw replay. No native app sync/rebuild while GPU workers live.

Local compiler/GPU quiet window extended through16:30UTC by peer coordination; respected. All native compilation here remains remote5090, local CPU analysis/ZIP/NumPy only. No message reply sent to peer. Original2x decode/trueGPUconcurrency,4B references,tuned/repeated refs/fullcontext/broadquality requirements remain unmet/active.


P179 isolated-build failure retained: original48577 native-canonical-fp8-attempt1 completed2. Actual helper executable compiled, but plugin-policy test clone used generic std::unique_ptr<IPluginV2DynamicExt>; TensorRT interface destructor is protected (MSVC C2248). This is my test lifecycle error, not native numerical failure. No policy/matrix gates ran. Archive215793bytes/20files/SHA256a8acff814e2a920102114be2d500347f106a6631e92885d6e3db94f5220086ba preserves raw build log, original source package/source snapshots/workflow/partial executable before correction; downloaded. Test now uses custom destroy() deleter; candidate math/plugin source unchanged. Retry reuses same fixed source slot only after terminal failure+immutable archive+old source hashes verified. New package19936bytes/SHA256475e426b6929828562295abb41f97d3c2fbc499004fa86d41f58c061006a5dcf and explicit old/new source-fix receipt; original package/failure archive remain intact. Original48577 native-canonical-fp8-attempt2 started. Still no app installation or speed claim.


P179 native candidate retry original48577 native-canonical-fp8-attempt2 completed0: build0/policy0/control0/candidate0, GPU workerclosed. Plugin default16-byte/candidate20-byte serialization, clone/deserialization/policy/profile rejection tests pass. Same native executable default path reproduces archived M16/M32 outputs; optin native helper (no wrapper override) selects canonical ID35 and makes bothM32 rows exactly match M16 twice,18word deviation per completeM32 output from old algorithm67 is retained explicitly. Cast/quantized arrays unchanged. Actual application DLL stays34cb. Archive and independent replay pending this entry; lowering source is packaged but not compiled in this isolated component build. Next after replay: guarded5file source install/native build, actual TensorRT stage/default-path/full64B2 gates before timing or promotion.


P179 isolated candidate evidence closed: archive3391505bytes/44files/SHA25649b0f6508fc5e8719cba4f38a1defb9cbe2270f494d4a4c9a9e82d1beb2e6a64 downloaded and independently PASSED all CRC/file/package/source/exe/terminal/failure-link/receipt/policy and raw-array gates. Retried source differs from failed original only in lifecycle test and retry-controller bookkeeping; native math/policy hashes unchanged. Both default profiles reproduce original control arrays; native canonical mode (without wrapper intervention) selects same opaque35algorithm asM16 for all4calls, casts/packed arrays unchanged, bothM32 rows exactly matchM16canonical twice. ApplicationDLL remains34cb; isolated result does not compile lowering nor establish actual TensorRT/full64batch success. Prepared guarded5file app-source install requiring verified archive and exact original aefb source hashes; originals/34cb retained. Existing fixed native build slot reused, no new clone/model copies.


P179 integration follow-up: guarded native-canonical-fp8-source-install-attempt1 completed0 after all original/new source hashes verified. Five originals saved under results/native-canonical-fp8-source-install/original; previous34cbDLL retained in existing garnet-layer0-stage-attempt1.dll (no duplicate old binary). ReceiptSHA256fc172c3eb3e74ed1069f904337ba8e77d5f91102b0c57db7c8a608d754c6d376. Original48577 native-canonical-fp8-integrated-build-attempt1 completed0: existing fixed CMake/build slot, MSVC17.14.42, sixCXXobjects/device link/newDLL. New unqualified DLL8568b9192a06067352636bfd94c97439d5524c3c0a0e7af744b19963cc0abe51 additionally retained as garnet-native-canonical-fp8-attempt1.dll. Lowering now compiled as part of app build, not a numerical admission. Source base is copied754 plus prior Python/stage patches and five native candidate files; do not call clean754. Candidate remains defaultOFF. Actual TensorRT stage controller prepared: unchanged earlier probe, control81files exact vs old per-profile outputs before candidate81files exact vs canonicalB1 across all rows/repeats. No timing or full64batch claim. Native candidate/methods pushed2b1da83ad; stage launcher/controller not yet pushed at this entry.


### P180 — actual TensorRT layer0 candidate closes; qualify deeper GDN family next

Previous goal turn made concrete progress: isolated native helper/policy and raw-array candidate proof, guarded source install, integrated8568DLL build and pushed fae5448ad. Original48577 native-canonical-fp8-actual-stages-attempt1 now completed0; control and candidate monitors0, no mismatched files. Default control preserves all81 old per-profile arrays. Candidate all81 arrays match corresponding canonicalB1 observation for every B1/B2 row/repeat, including entire primary packed hidden+conv+recurrent state and all11intermediates. Original prompt first16tokens/zero state only; no deeper/full64batch admission.

Closed archive losslessly deduplicates repeated raw bytes:9735754bytes/202 original file observations/57uniqueobjects/SHA256bf0788e7c4fdd95f50c16a525aeb9acafd9ed5f6b6e0043cb5b2af8be10f97a7. Every original path remains mapped to its content object;162raw F32observations retained, no trials/files excluded. Includes build/install/old-source backups/source/runtime/worker logs; old archives linked by hash. Separate remote CPython/NumPy independent replay completed0: full CRC/allobjectSHA/allfileobservations/links/input/config/all162raw arrays/primary/repeat/row/projection/negative-profile/graph/resource/terminal gates pass. Archive download/local duplicate replay pending at this entry. Not a model timing or2x result.

Architecture/source inspection confirms full-attention o_proj has the same [5120,6144] dimensions as GDN out_proj (24*256=48*128=6144). Initial option was geometry-only and could also cover that currently unqualified family, though this actual layer0 test contains GDN only. Prepared narrowing at lowering call site to .linear_attn.out_proj.weight names, retaining original M32/N5120/K6144/SM12.0/cuBLASLt130400/workspace/AlgoCheck guards. Full objective unchanged; full-attention and full64batch still require their own correctness evidence. Local name-guard source prepared while worker ran; no remote app-source mutation/rebuild during worker. After closure/archive/independent proof, guarded one-file scope install and existing-slot rebuild are next, then unchanged historical real-prefix producer across deeper layers and all41prompt tokens/chunks. Historical prefix producer already exists/tracked and is byte-identical to old internal archive (SHAa5892d0342f333e4108210caa22de2063d9c1542ef3696ef5e8379eeba72873e); attempted restore refused existing file, no overwrite occurred. Local compiler/GPU quiet window through16:30UTC respected, all native builds on remote5090.


P180 local replay follow-up: downloaded9735754-byte object archive independently PASSED locally too, same202observations/57objects/162arrays as remote proof. One-file family-scope install original48577 native-canonical-fp8-family-scope-install-attempt1 completed0 after archived/verified closed stage worker; original lowering saved, native helper/plugin/default math unchanged. Scope fileSHA1c0dc3fc32af61c4545d18ae33c47ce58103ce6a774a7ea02058356762acbae2. Original48577 native-canonical-fp8-integrated-build-attempt2 live in same build slot. Prepared unchanged historical prefix producer (a589...) through prefixes1/2/3, bothB1/B2/two resets, all41tokens/3chunks,162internal state/hidden arrays plus12final hidden files. Explicit gates: preserve old B1 prefixes1/2 internals and prefixes1/2/3 final hidden; every B2 row/chunk/state must match current B1. This remains deeper GDN correctness only, not full64-layer token-ID/HTTP/concurrency or speed admission. Original2xdecode/trueconcurrency and4B/tuned/broad/fullcontext scope remains active/unmet.


P180 scope rebuild original48577 native-canonical-fp8-integrated-build-attempt2 completed0: one lowering CXXobject/device link/DLL; new unqualified1ed48e36d31c1cd7cca8fe6825049331fc0b584f54215fef0d940d4aa624d767 retained as garnet-native-canonical-fp8-attempt2.dll, previous8568retained. Scope-install receiptSHAbe6c847cc58a39ddefe518875f55508eb6d2690ae39eae66dff7daf51ad562eea. Names now restrict option to GDNfamily; only matrix-selection callsite changed, helper and plugin hashes unchanged. Prefix123 controller/launcher uploaded; numerical gates not yet claimed on this new DLL. Default full27B and full64B2/HTTP performance admission still required.


### P181 — deeper GDN prefixes complete; restore original full64B2 gate next

Previous goal turn progressed via independently verified actual TensorRT layer0 arrays, guarded GDN name scope, rebuilt1edDLL and pushed3bc520342. Original48577 native-canonical-fp8-prefix123-attempt1 now completed0 (outer controller0/monitor0/probe0/arraygate0). All three prefixes1/2/3 report B1/B2 last_hidden exact, maxabs0, relativeL2=0. Producer captures all41text tokens over starts0/16/32, both profiles/two resets/duplicateB2 rows:162full hidden/GDN-state observations plus12final hidden files; all27chunk/state cross-profile comparisons and retained oldB1 witnesses pass controller. Numerical admission remains pending separate archive replay at this entry; full-attention/full64 IDs/HTTP/MTP/throughput are not yet qualified.

Optional outside-process loaded-module query was attempted too late: first PowerShell query had a foreach/pipeline parse error, corrected query then found no study Garnet module because authoritative original worker had exited. No module-observation file was written, no mapping/hash claimed. This is an observation miss, not a worker/runtime failure or reason to restart. Next full-model run can record loaded module during construction; existing binary source/launcher/build identity retained separately. No app source sync/rebuild while any GPU worker lived. Local compiler/GPU quiet window remains through16:30UTC; peer continues local tree regressions, remote5090 unaffected, no peer reply sent.

Prepared unchanged original probe_native_batch.py full64 B2 ragged text/image/red/green ID gate against old B1 outputs (probe SHA17c028175de510fe0945a7164bac6eb187e6caa037e0f71123a1215f901e03ad), byte-equal to preserved initial failure archive0d79e863108f3efa9ffb1af90b777ec1f1b71d37c6ecf3dad654bf1eb89abca4. Fresh output slot, candidate1/defaultNVFP4selector0, current1edDLL/source guards, independent deeper-prefix proof required before launch. No relaxed token-ID assertions or invented benchmark. Real fullmodel/serving/2x and4B/tuned/fullcontext/broadquality requirements remain active.


P181 closed deeper-prefix evidence: lossless object archive29261705bytes/210original file observations/59uniqueobjects/SHA256a515db12bad70689fc8590166640f1086abe27c4f321fc5a8916b0c24dc345ec. All162internal+12final hidden raw observations retained with no repeat exclusion. Remote independent verifier started under original48577 native-canonical-fp8-prefix123-independent-replay-attempt1; full29MB download in flight. Default app/source remain frozen at1ed/GDN-name-restricted version. Full64B2 controller/launcher ready but blocked on this specific independent proof, not a user blocker; no speed/admission claim yet.


P181 independent remote replay completed0: all210observations/59objects,162internal+12final hidden arrays,27crossbatch chunk/state comparisons,36retained oldB1 array witnesses and actual3/6/9 NVFP4+FP8 body/count/graph/repeat/row/resource/terminal gates passed. No gate relaxed. Full64B2 original gate may now launch; local duplicate replay awaits complete29MB download. Current1edDLL candidateGDNfamily only, no additional native edits.


P181 fullmodel follow-up: original48577 native-canonical-fp8-full64-b2-attempt1 started through append-only outer correctness ledger after successful independent prefix proof. Same original17c.. fullbatch probe and frozen1edDLL. Live external process-module observation succeeded while worker14080 was constructing: path D:/CantorAI/5090-study-20261023/build/bin/garnet.dll, diskSHA1ed48e36d31c1cd7cca8fe6825049331fc0b584f54215fef0d940d4aa624d767, UTC2026-10-24T15:56:46.9417359Z. Receipt saved under fullB2 output, observes mapped module path plus on-disk filehash, not relocated memory-image bytes. Earlier missed prefix query remains honestly missing. Current fullmodel report starting/cases0, no completed fullID result claimed.

Downloaded29MB deeper-prefix archive independently PASSED locally too:210observations/59objects,162internal+12final hidden arrays,27crossprofile comparisons,36oldB1witness arrays. No numerical/source mutation or rerun. Prepared fullmodel archival/replay methods that explicitly preserve any failure as failure and compare actual IDs to raw captured teacher responses; no threshold relaxation or timing claim. Existing MTP7 negative run is documented P079 (text/image decode0.95747/0.89335 times MTP3); do not assume draft-depth increase provides speed or repeat this unchanged experiment. Full-model runtime is still live; no app source/rebuild/competingGPU job started. Original2x and broader reference/context/quality scope remains active.


### P182 — investigate original live fullmodel wait without restart

Previous goal turn progressed through independently verified deeper-prefix arrays/oldB1 preservation and original full64B2 launch (pushed4c6f61710). Original48577 native-canonical-fp8-full64-b2-attempt1 remains authoritatively running, no output cases yet. Worker14080 main-thread CPU near100s and all queried threads waiting; GPU4393MiB/utilization0 at observations. Available RAM~49.4millionKiB of100.3millionKiB, so available-memory reading does not support a current RAM-exhaustion claim. Original historical fullB2 failure353.438s is retained for context, not an ETA or reason to restart. OneDrive process lists parentPID14080 but its creation predates current Xlang process; parentPID alone is reused/stale evidence and does not establish it as a child/cause.

Read-only source audit confirms TensorHelper DeviceAllocation destructor calls synchronous cudaFree and Embedding allocates two device tensors; current resident-controls path already bypasses repeated embedding allocations. Existing ABBA measured+6.7/+10.4%decode is still the only scoped result, no additional speed or causal gain inferred from source/API sums. Consulted related Garnet-QWen3.5 thread read-only: native tree consumer and selected-path commit work remains in progress in separate qwen35-cuda worktree. No code copied, runtime mixed, speed claim or peer message sent; preserve its local compiler/GPU quiet window now through16:50UTC. Related tree work is a potential future backend method, not completed27B optimization evidence.

Captured external normal minidump of original still-live process14080 (birth1792857388.6424217 matches observed process instance), no termination/restart/source/DLL mutation/full address-space dump. Dump flags6176 excludes FullMemory;870329bytes. Capture original48577 native-full-b2-wait-dump-attempt1 completed0 through setup ledger. Archive2292859bytes/SHAf4ed3a31669248a9bec37410976a98a24bc697f0deb745c6f5edce4afa36dd5a includes dump,receipt,helper source and exact1edDLL; no PDB available in build/bin. CapturedUTC2026-10-24T16:12:50.387955Z. CPU increased to104.9s around this observation; not a terminal state or cause proof. Offline native stack inspection next using installed local Microsoft CDB10.0.28000.2526 (CPU-only, no local compiler/GPU). Still no fullmodel ID gate, HTTP/concurrency or2x claim.


P182 dump replay: archive CRC/allcontained SHA verified locally, installed CDB exit0 offline. Main thread context is ntdll!NtWriteFile via KERNELBASE!WriteFile and Garnet C++ ostream frames, not an executing GPU/CUDA wait at the top. Export-nearest labels such as BuildSingleImagePromptIds+largeoffset are NOT trusted function identities without private symbols. Context source file handle0x580, NtWriteFile length0xfffffffe (4GiB-2), deeper ostream size argument0x3f2c31bbc; source TRTBuilder.BuildCapturedGraph writes serialized network to model.engine.tmp in one ostream.write(size), then atomically renames without checking write/close status. CUDACompiledGraph creates unique temp directories and CloseUnlocked releases execution but contains no removal of disk cache directory. This points to large engine-file I/O as the next investigation, not yet a proven disk-full or backend math cause. Full volume/free-space command plus broad temporary-file enumeration timed out; do not treat that observation timeout as worker exit or restart. New read-only handle/volume/private-tmp inventory helper prepared to identify the actual file and available space; no file deletion performed at this entry. Original fullmodel handle stays running and code/DLL frozen.


P182 original full64B2 run authoritatively completed1, not restarted: two128-token rows saved then old-ID assertion failed. GPU85MiB after closure. File-handle inspection then returned WinError87 because originalPID was already gone; its ledger/failure is retained, no file handle/path observation claimed. Full failure archive261328bytes/39files/SHA25601d1341e2c6d7f5f58880255ff0dabdb451a7c9b58b6e9730ec8276a04c20b82 includes all4 raw teacher request/responses, generated IDs, loaded-module receipt, source/build/runtime/monitor and failed/successful observation ledgers, links original initial failure, deeper-prefix proof and small dump. Remote independent CRC/allSHA/teacherIDs/counts/precision/graph/resource/terminal/module replay PASSED as FAILED MODEL GATE: text firstdiff14 (was0 initially), image firstdiff8 (unchanged),128outputs each. No fullbatch/HTTP/performance admission and no request exclusion.

Post-exit direct disk query records Cfree753672888320bytes/Dfree646974857216bytes; not evidence of earlier disk-full. Actual physical mapping: C:disk1 CT1000P510SSD8 SSD/NVMe, D:disk0 TOSHIBA HDWD130 HDD/SATA. Thus large cached embedded engine writes on studyD temp plausibly explain cold-load delay; stack supports file I/O but exact historical delay cause is not fully proven. New experiments will keep source/checkpoints/results on fixedD study and use one fixed study-owned cache under userLocalAppData on C:NVMe, with owner receipt/reserve check. Only new private engine files from a closed test leg may be removed after absolute-path/reparse/name/birth guards and hash/size recording; no oldDcache, source/model/raw/deployment asset deletion.

Prepared controlled prefix4 diagnostic using byte-identical old probe: first current1ed GDN-name-only binary, then preserved8568 geometry-wide binary (same helper/plugin math, earlier lowering also selects same-shaped full-attention o_proj). Both B1 outputs must preserve old witnesses, repeats/duplicate rows required; record B1/B2 hidden equality as diagnostic, not speed or model admission. Exact binary hashes/original manifests distinguish older binary from current source checkout, restore1ed after closed queue even on error. No native source/rebuild required. Fixed SSD scratch only changes engine placement, not model/checkpoint/math. It is not a warm TTFT/decode speed claim. Original2xand trueGPUconcurrency, broader quality/fullcontext/tuned/4B references remain unfulfilled and active.

P183 SSD setup original attempt1 completed1 before folder creation: Python rejected encoding alias utf8-sig. Corrected only to utf-8-sig; retry attempt2 completed0, physical C SSD/NVMe owner receipt created 2026-10-24T16:37:23.973127Z. Failure and retry source/command ledgers retained. Earlier P182 full-B2 archive hash transcription corrected against independently verified bytes (data unchanged). Original prefix4 two-binary queue native-canonical-fp8-prefix4-scope-comparison-attempt1 started on preserved handle; no application-source updates/rebuilds during worker execution. All performance/2x admissions remain unchanged.

P184 Controlled prefix4 queue completed0 for both existing binary variants and restored current1ed DLL. Local and remote independent replay PASSED: 128812-byte archive SHA256260dccac79495484ec69e24d7d2c4e19c4bd6d3a9fb34dc140f880dd299e7a45,46 file observations/39 SHA objects,8 finite raw last-hidden arrays. Both B1/B2 rows/repeats byte-exact and preserve historical B1; no measured advantage from expanding canonical policy to full-attention o_proj in this prefix. All private SSD engine files created by each closed leg were hash/size recorded and retired with nonrecursive owner/path/reparse guards; no source/models/results removed. This excludes first4layers for identical41-token text prefill only; ragged full64 text/image generation remains failed.
Prepared next full64 duplicate-text prefill diagnostic, same probe logic with explicit deeper-prefix selector and separate file (historical probe untouched). No head/vision/generation/HTTP timing: discriminate depth accumulation from ragged/decode failure. Current1ed source/native frozen once worker starts; same fixed SSD owner,96GiB reserve, closed-run private engine cleanup. Not a substitute for full B2/VLM admission or2x target.

P185 Full-depth original handle native-canonical-fp8-prefix64-attempt1 revalidated running, PID30240 with increasing CPU (34.11875 to66.09375seconds) and working set (13.06GB to18.13GB); no terminal result and no restart. Prepared closed-run archive and independent raw-array replay methods locally; not deployed into a live application checkout. Source review: prefill_sequence per-row masks use absolute token lengths, CopyLast kernel preserves prior hidden when an entire row is inactive. No proven mask/index defect from review, so no speculative math change. Existing older native-batch2-prefill-controls run (source3ef07624c) already passed ten captures/eight companion-isolation checks; this historical result is not automatically a current1ed result and is not full B1 equivalence. Next current full-depth witness will distinguish duplicate-text prefill divergence before choosing further ragged/generation diagnostics. No performance admission.

P186 Original full-depth handle remains running. Read-only PID30240 sample nowCPU163.8125seconds; native report contains completed B1 case: repeats exact,192NVFP4/208FP8,1capture/4replays, zero capture failures; B2 still pending. Fixed C cache contains completed16957510948-byte model.engine. These are progress observations, not final cross-profile/model or timing qualification. No worker restart/source/runtime deployment.
Source inspection confirms external shared FP8 family-selection gap: generic compressed_packed_<index> symbolic input names do not preserve checkpoint suffix used by current optional lowerer. BindExternalModelOptWeight already receives validated name; future fix must retain logical name in NativeFP8Weight metadata for both embedded/external paths and qualify both. No such source patch or binary deployed yet. Updated concrete implementation task to current evidence/gates and full remaining objective.

P187 Original full64 duplicate-text prefill completed0; terminal controller recorded/retired only4 new owned SSD engine files (B1/B2 bodies16957510948/16957774780bytes plus norm plans) after closed worker. Lossless archive162757bytes/34observations/32objects/SHA256af7900b621c50874757e65b332b36b439170cd72dfc6926632733e6a776c87c0 independently replayed locally and remotely:4finite raw arrays, exact within-profile repeats/duplicateB2 rows, actual192NVFP4/208FP8 and graphreplay pass; cross-profile all5120hidden values differ,maxabs33.5263671875,relativeL2 .18555398043329663. Sampled construction/correctness adapter peak25593696256bytes, not serving peak. This locates divergence in full-depth duplicate-text prefill, before ragged VLM/generation/head; no fullmodel/performance admission. Next prefix8/16 same1ed localization guarded on this independent negative proof; historical prefix probe remains untouched.
Local-only pending native fix carries logical checkpoint name in NativeFP8Weight for both embedded/external registrations and uses that identity for optional GDN selection. Three source files changed, no remote deployment/build, no qualification claim. Separate shared-input prefix witness prepared with only storage binding mode changed. Preserve these edits until current frozen1ed localization closes and prior evidence is archived.

P187 localization launcher original attempt1 completed1 at proof-status guard before output directory/GPU worker creation: retained four-layer status string instead of full64 negative-proof status. Corrected status+negative exactness guards, source receipt preserved in failed outer ledger. Fresh supervised handle native-canonical-fp8-prefix8-16-attempt2 started using same fixed output slot (no previous experiment outputs created), unchanged1ed runtime/probe. No original GPU process was restarted. Native logical-name patch remains local-only/unbuilt and separate from these results.

P188 Original prefix8/16 localization attempt2 completed0. Independent local+remote replay PASSED as negative cross-profile evidence: archive153887bytes/38observations/34objects/SHA2567e6a1a482611ae7c5d7735a053b132f5467571b2dc165460c42ad939373c0b53,8finite raw arrays, resets/duplicate rows exact, graph counts/replays/resources/terminal ledgers reaccounted. Prefix8 changed5115/5120values,maxabs.0810546875,relativeL2.013935913373372138; prefix16 changed5120,maxabs.2073974609375,relativeL2.047504427598729526. Sampled whole-test peak9181384704bytes is not serving peak. Generalized archive/replay methods accept explicit64 or8-16; default64 replay reproduces all prior numeric/hash/resource results exactly.
Earlier prefix4 witness compared only final prompt hidden, not every token. New separate hidden-chunk probe captures every output token of all3chunks at prefix4, B1/B2/two resets, avoiding large full-attention KV dumps and retaining original probe untouched. Same model/kernel/control logic; current1ed binary/model source stays frozen. Supervised native-canonical-fp8-prefix4-hidden-attempt1 launched only after negative prefix8 proof. This investigates first-attention-layer earlier-token differences that could seed later GDN states; not a numerical cause or performance claim yet. Three-file logical-name native fix plus external-binding probe remain local-only/unbuilt, not part of recorded runtime. Original2x/full scope remains active.

P189 Prefix4 every-token witness closed0 and independently replayed on both machines: archive1315728bytes/52observations/38objects/SHA256a208ea9daf57e9d239d8c5ef60b46041a2956008c47ca59193ebc8c1436f8a06,22finite raw arrays, historical B1 final hidden preserved, all within-profile repeats/duplicates exact. Final prompt hidden is B1/B2 exact, but earlier-token full-chunk outputs differ: starts0/16/32 changed4799/4810/4962 words,maxabs.00390625/.002593994140625/.00390625; changed valid tokens [1,2,4,6,9,11],[23],[33,36,38,39]. Thus last-token equality alone hid first-full-attention-layer earlier-token differences capable of seeding later states. No unique operation/algorithm cause yet.
Generalized archive/replay methods now support hidden-chunk experiment and reverified both prior64 and8/16 raw/numeric/resource results unchanged. Next geometry-wide8568-only controlled intervention uses exactly the same hidden-chunk producer, validated negative current1ed proof, saved build1 provenance, historical B1 witness, idle-only DLL swap and finally restore1ed. Current source remains GDN-only (not represented as geometry-binary source). Original supervised handle native-canonical-fp8-prefix4-hidden-geometry-attempt1 started; do not mutate source/runtime while live. Native metadata fix remains local-only/unbuilt. Original2x/fullscope unchanged.

P190 Geometry-only hidden-token intervention closed0 and restored1ed. Local+remote independent replay passes archive1327591bytes/57observations/43objects/SHA256e19f4588db72e1463e4011f8897571eddc913db35703a11cd50f140424af7054,22raw arrays; all B1 hidden chunks preserve prior GDN-only bytes, but B2 changes only9/1/3words versus GDN-only across starts0/16/32. Remaining cross-profile changes4793/4810/4959words are localized to valid tokens4/23/33 respectively; all repeats/duplicates stable. Full-attention output FP8 canonicalization removes some minor differences but does not explain/fix dominant differences. No fullmodel/batch/performance admission. Initial archive helper failed on guessed old-lowerer backup path; read-only inventory found original-lowering.cpp, corrected path and retry2 succeeded without rerunning GPU test. Actual old lowerer bytes verified against build1 manifest, current source against build2, both identities retained.
Prepared default-off diagnostic_full_stages at embedded mixed27B layer3 B1/2,T16,H5120,Q24/KV4,D256,rotary64,mrope11/11/10,capacity4096. Exports ten bounded per-row stages while preserving complete normal hidden/KV primary as prefix. Fused attention and no joined/external projections required; invalid flag/profile rejected. Diagnostic primary must exactly match ordinary/retained witness before stage attribution. Additional exports may alter TRT fusion, so no numerical conclusion from unqualified debug outputs. Local source edit first refused a CRLF-specific search before writing; corrected scoped byte edits preserve existing EOL.
Three-file logical checkpoint identity fix is included with two-file diagnostic support in guarded five-source build candidate. Actual remote originals snapshotted and all five normalize to dedicated branch baseline; payload hashes match local bytes. Originals and existing1ed DLL retained; no live GPU worker at install. Original supervised native-attention-stage-build-attempt1 started using fixed build slot; source/runtime frozen until terminal. Build not yet qualified, no default serving admission. Original2x and remaining full objective unchanged.

P190 build original handle completed0: existing MSVC/CUDA/TensorRT fixed-slot build compiled13C++/link steps plus CMake successfully, new library saved garnet-native-attention-stage-attempt1.dll; manifest/source receipt and original5sources retained. New native code is build-checked only; no GPU primary/stage/negative-profile or fulldefault-model gates have run. Do not use compile success as arithmetic or serving admission. Next prepare first16 canonical prefix3 raw hidden witness from independently verified prefix123 archive; execute ordinary layer3 and diagnostic layer3 under identical controls and compare complete primary hidden/KV plus retained prefix4 hidden before attributing any of ten exports. Then qualify shared FP8 logical-name binding behavior separately. New library and source provenance must remain frozen for GPU probes.

P191 Prepared first-attention witness from independent prefix123 and prefix4-hidden archives. CPU setup original prepare-first-attention-stage-witness-attempt1 completed0: all6 prefix3 first-chunk input row/repeat observations identical; retained per-profile prefix4 outputs/repeats checked, config/prompt IDs agree and checkpoint config hash matches. Three raw fixture files and linked archive/receipt hashes retained under results/first-attention-stage-witness; no fresh model/weight copy.
New actual layer3 probe supplies those input floats directly, zeroKV, exact first16 positions/slots and fused attention. Six negative profiles (flags2/-1,B4,T8,capacity32,wronglayer2) must reject. Ordinary and diagnostic graphs each3 fenced warmups+2captured resets, both B1/B2. Saves complete primary hidden+KV in <=1M-element row parts, all ten stage arrays and input bytes; requires full ordinary/diagnostic primary exact, retained prefix4 hidden exact, input exact, finite values, repeats/duplicate rows,3NVFP4+4FP8 and graphcapture/replay/count/output-size contracts. Additional exports may alter fusion, so primary failure disqualifies attribution. Methods parse; original supervised first-attention-real-stages-attempt1 now live on frozen fec5 native build (full hash in build manifest). Fixed C owned cache and postclosure guarded cleanup reused. No serving/quality/performance or2x admission.

P192 First-attention ordinary/diagnostic witness original completed0 and independently replayed locally+remotely: archive4920259bytes/218observations/70objects/SHA25609960fad5618072da5eb93eb61c12368d03b0203f2f06a6764026aabe88a5143,171finite raw arrays. Complete primary hidden/KV, historical per-profile hidden, fresh canonical input, all row/repeat bytes, zero unused KV positions and projectedV-to-value-cache layout exact; six unsupported profiles rejected;3NVFP4/4FP8, graph replay/output sizes/resources/terminal/source hashes verified. First changed stage q_gate:16words across10tokens,maxabs.03125, while maskedinput/inputnorm/projectedK/projectedV exact. Attention core1823words,maxabs.0003576185554265976; outputhidden4799words,maxabs.00390625. This identifies first observable operation, not yet unique algorithm cause. Sampled diagnostic adapter peak5414899712bytes is not serving peak.
Actual layer3 query/gate FP8 fixture extraction setup completed0: real common inputnorm M16/M32,K5120; raw E4M3 checkpoint q_proj [12288,5120]62914560bytes and exact calibration scalars/header offsets; expectedBF16 outputs extracted from verified stage arrays with low16bits zero. Fixed bounded fixture folder, no checkpoint clone. Untimed native-helper control uses same helper/quantizer with observed actual algorithm opaque bytes and optional checked shared-M16 intervention, no assumed algorithm ID; default reproduction must pass both archived profiles before candidate admission. Original actual-first-attention-fp8-control-attempt1 now live for standalone compile/default control, appsource/DLL unchanged fec5. Source creation guards refused incomplete local removal of old unused mode before writing; final control exposes only default/shared modes. Full-model/batch/performance/2x gates remain unfulfilled.

P193 Actual query/gate control build/default execution original completed0, then independently replayed locally and remotely: archive4320642bytes/35files/SHA256be6ddc004ed5bf817412aaace45ca33268f480ced518a17cc57e9b5e02abcd1a. Both actual M16/M32 BF16 outputs reproduce archived stage words twice, CPU BF16RNE cast/E4M3 quantization match every saved input byte, duplicate/cross-profile quantized rows identical.16BF16 output words differ per row. Both algorithm IDs are67, but opaque configurations differ and are retained in raw logs/proof; do not call this different-ID behavior. Unique cause is not established by default observation.
Only after independent control proof, supervised actual-first-attention-fp8-shared-algorithm-attempt1 launched for same-executable fresh default control then checked common-M16-configuration intervention. AlgoCheck/workspace acceptance required, all output arrays retained on failure, no appsource/DLL/math/default change and no timing admission. Fixed build/fixture/results roots reused. All original2x and broader model/quality/concurrency/reference/context requirements remain active.

P194 Same-executable query/gate control/common-configuration intervention original closed0, independently replayed on both machines: archive4871649bytes/35files/SHA256322bdc34b58d9a90f864c0b5dd12f840eb84260a3b01773bcfd0a35d21a40ea1. Default control reproduces both archived profiles and exact previous opaque configurations; checked commonM16 configuration makes bothM32 rows exactly canonicalB1 twice; all casts/quantized bytes unchanged. AlgoCheck succeeds/workspace0 under4MiB bound. This is a successful intervention on one actual projection, not fullmodel/performance proof. ID67 alone is insufficient; full opaque config retained.
Prepared local native candidate: research canonicalM32 permits only N5120/K6144 or N12288/K5120, exact4MiB,SM12.0,cuBLASLt130400, derives actual M16 configuration with expected ID35/67 by geometry and requires AlgoCheck for physicalM32; no fallback. Existing GDN flag remains; separate defaultOFF GARNET_FP8_FULL_ATTN_M32_CANONICAL flag matches validated logical self_attn q_proj/o_proj names and exact geometry. Default16-byte wire and optional20-byte policy1 preserved; internal bool renamed generically, Q geometry allowed only for candidate policy. No app install yet.
New isolated source slot (eleven small source files, no checkpoint copy) and existing build directory compile query control, GDN regression and plugin tests. Policy checks cover wire/clone/deserialization/rejects/M32 for both geometries; existing six synthetic embedded/external serialized graph CPU-oracle regressions remain, plus real query and GDN default/candidate matrices. Original native-attention-fp8-policy-attempt1 launched only after independent common-config proof. Active app remains fec5/source unchanged; candidate source isolated. Local compiler/GPU quiet window remains respected (all builds/tests remote5090). Full objective/2x target remain unfulfilled.

P195 Isolated policy original completed0 without termination/restart. Both geometry wire/clone/profile checks, six native serialized/CUDA-graph CPU-oracle regressions and real default/GDN candidates passed. A long quiet plugin process PID30860 was observed CPU8.5seconds/GPU629MiB/util0; external small-dump setup later failed WinError87 because instance had already exited; no dump captured, failed diagnostic ledger/source retained. No deadlock cause claimed.
Inspection found query test bug: broad condition replacement also enabled shared-config observer override in native-candidate mode. Thus query canonical output does not independently qualify helper selection. Archive8255916bytes/73files/SHA256a7eed7d07e2dfdaf9434b4dad7b3511e83a1c060cd8a336ace36f24d15fa5300 independently replayed locally/remotely as INCOMPLETE, native_qualification_passed=false; GDN/default/wire checks remain valid, no app install. All raw data/source/failed observation retained.
Observer-only repair changes exactly the first condition back to sameAlgorithm-only and requires no nativeCanonical reference inside ObserveMatmul; native math/helper/plugin/lowering sources unchanged. After all prior files/archive/fixture hashes and application DLL checked, source slot reused with original bad query source retained. Rebuild only query executable; fresh same-executable default/native query checks require no SHARED_ALGORITHM_CHECK in candidate log and canonical rows. Original native-attention-fp8-query-repair-attempt1 started after incomplete proof. Existing plugin/GDN evidence reused to avoid unnecessary repetition. App still fec5; full2x/model/concurrency/reference/quality/context requirements open.

P196 Observer-only query repair completed0 and independently replayed locally+remotely: archive4870779bytes/33files/SHA256bffdb16c125b6903eac79ea5a2cbc32d08e146f1e3e8496dd4cbcf7c1ad196a4. Exactly one test condition changed, native source unchanged; observer body contains no nativeCanonical dependency, no SHARED_ALGORITHM_CHECK in either fresh log. Same executable default outputs preserve baseline, native candidate bothM32 rows exactly canonicalB1 twice, casts/quantization unchanged. Combined isolated_native_attention_fp8_policy_verified uses immutable prior six plugin CPU-oracle/graph and GDN/wire checks with all other source hashes identical; original first proof stays INCOMPLETE/false. No application/model/performance admission from component proof.
Guarded five-source app installer/build prepared and started only after combined independent proof. Expected old source hashes/bytes come from verified first-attention stage archive, current DLL must match retained fec5. New helper/plugin/lowering files must exactly match qualified isolated source hashes; originals saved before writes. Fixed CMake build slot reused and old DLL retained/restored on build failure. Original native-full-attention-fp8-build-attempt1 now live, no source/runtime mutation until closure. Both GDN and new full-attention flags defaultOFF; actual same-DLL control/candidate primary+allstage and full64raggedB2-ID/HTTP gates still precede timing. Full original2x and broader objective active.

P197 App native integration build completed0; new DLL SHA2563c3207b84b0e048f7e69736f3ae927ffb629c40ffe086994765e9a70ac5f58dd. Five installed source hashes match qualified isolated repair proof; retained fec5 unchanged. Build success is not model correctness/performance admission. Prepared same-DLL flagOFF first-attention control with original producer and historical per-profile hidden gates, explicit FULL_ATTN flag0, old GDN flag1. Fixed results folder full-attention-policy-control-stages-attempt1; source/runtime frozen during supervised worker. Candidate canonical references must be distinct from historical B2 references; not yet run. Original 2x objective remains unmet.

P198 Same-DLL default-off stage control completed0; lossless archive9488378bytes/212observations/64objects SHA2566cb6e67427e3e4e25646e4bb224293279427afb3a5064a253ea47af2432af137 replayed independently on both machines. All171raw arrays exactly preserve retained fec5 results; primary hidden/KV, historical hidden, six negatives, repeats, valuecachelayout and graph/source/resource gates pass. Diagnostic peak5014343680bytes is not serving performance. Quiet live worker27932 showed increasing CPU/memory; prepared small diagnostic capture but worker closed before invocation, so no dump captured/no diagnostic attempted. Candidate launcher now requires verified same-DLL control; canonical B1 hidden plus all ten stages and complete primary cross-batch bytes, no tolerance relaxation. Original full-attention-policy-candidate-stages-attempt1 launched after both independent replays; no appsource/runtime mutations while live. Full64, model quality/performance and original2x target remain unfulfilled.

P199 Candidate stages original completed0 and independently replayed remotely+locally: archive7810795bytes/213observations/59objects SHA256ef0f263ce5fab14ff44f6e5acfb85a9050afb5d9dc4683081cf7dca50b23b5fd. All171arrays exactly match retained canonicalB1 including full ordinary/diagnostic hidden+KV and alltenstages; repeated/duplicate/crossbatch bytes, historicalB1hidden, sixnegativeprofiles, valuecachelayout, source/runtime/graph/resource gates pass. Firstattention q_gate16word mismatch and downstream4799word hidden mismatch now absent in this bounded layer3 first16/zeroKV witness. Peak5398122496bytes diagnostic only. No fullmodel/serving/performance admission. Full64 original immutable B2 raggedtext/image/earlyEOS probe launched only after both independent stage replays; guarded fixed SSDscratch, raw failures retained and no appsource/DLL mutation while supervised full-attention-policy-full-b2-attempt1 live. Fullmodel/2x/broadcontext/quality/concurrency/reference requirements remain active.

P200 Original full64 B2 raggedtext/image/earlyEOS gate completed0 and independently replayed on remote+local. Archive96527bytes/32files SHA25692d1ce96638dc6f4f2a07a7400b95e1249245193df57d854361fd17dffabaecc. Four group/repeat captures yield eight exact per-request comparisons: text/image128IDs each across two resets; red/green2IDs each across two resets. Original probe unchanged against initialfailurearchive; all expected promptlengths/limits and saved request/response hashes,192NVFP4/208FP8 graph/runtime/source/resource/terminal gates verified. This clears the previously failed full64 B2 pilot ID gate on native3c3207 with bothGDN/fullattentioncanonical flags. No loaded-module path observation was collected for this run; disk/build/source provenance only, explicitly reported. Fixed private SSDengine17GB plusvision/head files hashed/cleaned after closure only. Fullmodel pilot is not HTTP/MTP/broadquality/context/throughput/2x admission. Next shared-external logical-weight binding qualification and native batched-serving measurements remain required, alongside defaultB1 regression/tunedreference/4B/4K broader scope.

P201 Shared external full64 B2 qualification launched after independently verified embedded fullmodel pass, same3c3207DLL and bothcanonical flags, same16/16profiles. New probe differs from immutable original only in decoder construction plus strict B2/profile,400poolentries192NVFP4/208FP8 and externalbinding byte/count assertions; generation/imageprep/reset/ID checks unchanged. No production factory expansion or application/native source mutation. Original shared-full-b2-attempt1 remains live; sampledconstructionmemory21663633408bytes not serving/performance proof. Prepared lossless archive and independent source-construction-delta, pool/graph binding, original request/response and repeated-full-ID replay. Original prototype prefix script untouched/unqualified; full objective remains open.

P202 Shared full64 B2 original completed0; archive91657bytes/34files SHA2566be0bc2285837f7a5f5b471711ea1f47c65767e62834315594d55ca9aa2c2b2e independently replayed bothmachines. All8fullID comparisons exact, text/image128IDs and earlyEOS2IDs eachtworepeats. Original generation/input/reset/ID source outside decoder-construction block identical afterEOLnormalization;400compressedpool192NVFP4/208FP8, externalgraphbinding count/bytes and zeroBF16external verified. Newnative logicalweight fix now passes actual fullsharedB2 generation. Decoderengine72738500bytes versus embedded16957774908bytes; serializedstorage observation only, not modelmemory/throughput. FixedSSDscratch artifacts hashed/retired afterclosure. NoHTTP/MTP/broadquality/4K/2x admission. Nativebatchbridge existing CPUprotocol fixture isn'tGPUworker; current decoder prefill rejects zero lengths, so partialgroup padding needs valid duplicate prompt with dummy row excluded from streamedoutput and generationactivity. Next implement/qualify research nativeB2 streamworker beforeHTTPtiming; app/native defaultoff retained.

P203 Implemented research native B2 shared streamworker: same400pool/16profiles, verifiedsharedproof and exactDLLguards, live synchronized per-row token events, independent completion/active masks; partialgroup borrows duplicate valid prompt for physicaldummyrow, dummyinactivegeneration/nooutput. ExistingfourdispatcherCPUfixtures passed and newworker/controller syntax checked. GPU protocol sixsequentialgroups/tenrealrequests includes repeatedmixedtextimage, partialtext, partialimage, earlyEOSpair and shortrow2tokens beside128token companion. Savedactual requestframes/rawstdout/stderr plus IDs/promptIDs/timestamps required; noHTTP/performance admission. Actualworker/launcher source hashes added to runledger before launch. Original supervised batch-stream-protocol-attempt1 started remote, source/runtime frozen while live; not yetqualified. Fullscope/2x remain open.

P204 Native B2 streaming EOF qualification completed0 and independently replayed bothmachines. Archive121644bytes/43files SHA256199606d6176e4af282a1ae3e965edf24c19f6c02ca2d53636857f3e07d74e3dd. Sixgroups/tenrequests: repeatedtextimage2x128, partialtext128, partialimage128, earlyEOS2/2, shorttext2withimage128. Allactual token streams/finals/promptIDs match retained B1 oracle, inputbodies exact except deliberate2tokenlimit, imageSHAverified, monotonic timestamps, no unknown/dummy/retired-row events. Metadata/runtime/source/resource/terminal gates pass. Diagnostic peak27082674176bytes; no servingmemory/throughput admission. Actual B2 protocol nowqualified for bounded pilot; next HTTP quality/stream gates then same-worker C1/2/4/8 measurements vs matchedreferences. Tunedrefs,4Brefs,4Kprompts,broadquality,MTPbatch and full2x objective remain open.

P205 Same-worker B2 HTTPpilot prepared after protocol independent pass. Eight nonstream exactHTTP responses (repeatedmixedtextimage,earlyEOSpair,partialtext/image) plus two exactSSE ID gates precede timing. Then existing benchmark_stream C1/2/4/8,8requests,2warmupwaves,128outputs,seed3,JPEG imagefirst binarygrid on literalIPv4loopback; physicalbatch2,2msgroupdelay,MTP0,T16/T16 explicitly recorded, C4/8queued fixedbatches not physical4/8. Dispatchlog nowrecords realrowcounts/IDs/physicalcapacity beforewrite; allfour existingbridgeCPUtests passed incldispatchreceiptassertion. Worker/native math unchanged. Original batch-http-pilot-attempt1 launched, no mutation whilelive; no measurements admitted before rawHTTP/SSE/workload/source/memory independent replay. Profile differs frompriorB1MTP3pilot; target2xstillunmet.

P206 HTTPpilot original closed1 during imageC4. Archive1202111bytes/377files SHA2568aa69ee67b21a388edc197df344d29f36daf4163ec59c6578965ce5cbcf1c886 retained beforeanytransportsourcechange. Localindependentfailureclassification confirms6savedsuccessfulcohorts arithmetic,6failedimageC4requests,quality8HTTP/2SSEreportedpass,parent/resourceexit1; performance_admitted=false/full_run_passed=false. Stderr contains invalidUTF8 bytes and terminal JSONencoder ASCII ord()expectedcharacter whenserializingtext; bytewiseproof avoids decodinginvaliddiagnostic bytes. No completedcomparison/2xclaim. SimplefullUnicodefixture closed0/oldemitterallcasesdidnotfail, so no cause assumed fromit. Actual CPUtokenizer-prefixprobe then reproducedexactTypeError on partialUTF8 decodedtokenprefixes incl IDs13962,91786; nativeDecode returns rawbytes. NormalizedUTF8replacementJSONcontrol passes. Prepared+deployed narrow native_worker_io emit text-copy UTF8decode-replace, preserving caller/IDs/counts andexistingASCIIJSONwire. All4IO+4bridgeCPUtests pass; repairedactualnative tokenizerprefix probe closed0 withallprefixes validJSON and3bytepartialwrites. No nativeDLL/model math change; bothold/newcontrols retained fixedseparatefolders/ledgers. Need independentcontrolarchive/replay and freshHTTPGPUtest before performance admission. StrictB2MTP3 launcher remainsgated bypriorHTTPsuccess, notrun whilefailureunresolved.

P207 Original HTTPfailure independently replayed bothmachines asfailed:6partialcohorts arithmetic only/6failedimageC4requests; no completedbenchmark. FirstUnicodecontrolarchive39464bytes/95files SHA2568c61db4c81ddb3f93ca058ecd150a77b4bdc58d7eeb8e75fced350a6df59dba5f retainedoldhelperfromHTTPfailure plusfullUnicode,actualprefix andrepair1 CPUcases/ledgers/sources. IndependentCPython verifier rejectedrepair1:3partialprefixcases replacementcounts differ (rawF09F andF09F94, plusASCIIprefix) embeddedreplace emitsperbyteFFFD vsCPython maximalvalidsubpart. Originalnative selfcheck hadpassed; notaccepted asqualified, noGPUretry. No tolerance relaxation.
Prepared canonical transport_text normalizer withstrictvalidUTF8fastpath and explicitmaximal-subpart fallback. CPUoracle tests includevalid/partial/overlong/surrogate/outofrange/invalidbytes;5IO+4bridge tests pass. CPUfixture generated independently fromimmutablefirstarchive rawbytes/Cpythonreplacement, SHA256d98e7e7ce90e9b37506abf711587c8112f677e45853c72d1fed7318fe0ba2697. Newnativeprobe repaired-attempt2 completed0 againstfixture,3bytepartialwrites, sameoldrawinputsha/IDs. HelperonlydeployedafterallGPUjobsclosed. Needarchive2andindependentCPythonreplay before freshHTTPattempt2; nativeDLL/math unchanged3c3207. Newnormalizer andmethodsreviewable, firstrepairevidencekeptincomplete. Fullobjective/2x remainsunfulfilled.

P208 CanonicalUnicodeCPUrepair2 archive20288bytes/44files SHA256e04c3a181794b5347c0d4fe193c6c181e2257d1b18f55e9f835ea854d82d1c1c independently replayed remote+local:12prefixcases,3originalordfailures,3firstrepairreplacementmismatches,allnewactualemitterJSONtextbytesexactCPythonmaximalsubpart and3bytepartialwrites. Samexlang2e2db/DLL3c3207; helperSHA256763d61a0e7f34794bca269e4214fa03ff64b95b913778182e7c34fe40292dc84 verified. Firstrepairarchivekeptunqualified, notoverwritten.
PreparedexactfirstfailedimageC4r2/r3 requestbodies fromimmutableHTTPfailurearchive withretainednativeprefix49/48tokens. RepairedHTTPdriver addsconcurrentfailedpairtwofull128repeats,oldprefixexactandnewfullIDrepeatgates beforetiming; original8HTTP/2SSEgates retained. Fixturelineage/hashandhelperproof/runtime guarded. NativeMTP0/B2/T16,2msdelay/workloads unchanged. Original supervised batch-http-pilot-attempt2 launched afterindependentCPUproofs, separatefixedresultsfolder; noapp/native/runtime mutationwhilelive. BothCPUevidenceandoriginalHTTPfailure preserved. ActualGPUrepairquality/newcompletedperformancecomparison stillpending;2xunmet.

P209 RepairedHTTPpilot original completed0/all8cohorts, closedworker/monitorterminal beforearchive. Archive1537097bytes/460files SHA25660ee68f6db33055b16e748b82eab462d38609c61f840a35b229e5d6fa8b9cc76 independently replayed bothmachines. Original8HTTP/2SSEgoldens and4actualcrashingpairregressioncaptures pass;64measuredSSE matchednativeworker IDs/actualimages; everyrequestbyte/IPv4transport/promptoutputcount matchesbothordinaryreferences; SSEmetrics/VRAM/source/runtime/terminal replaypass. Actualcohortoccupancy reconstructed: textC2/8allpaired,textC4threepairs+twosingles;imageC2threepairs+twosingles,imageC4/8allpaired; C1physicalB2withonevalid+dummy. Thus queuedC4/8 never claimedphysical4/8.
VerifiedgreedyB2 MTP0 T16/T16: textC1decode28.1333,TTFT160.519ms;imageC1decode28.3528,TTFT341.207ms. C8aggregate54.6550text/52.4758image vsordinaryfastest445.7685/439.8643,ratios.122609/.119300. Clearlybelowreferences/2xunmet; resultsnotcherrypicked. Baselinevalidforboundedpilot, notbroadquality/4K/tunedrefs/repetitionconfidence. Exactnumerics/rawcohorts inproof. Native-to-bridge lagdiagnostic preparednotyetanalyzed; no cause/speedclaim fromremovedcrash.
Onlyafterbothindependentreplays, strict shared-mtp3-full-b2-attempt1 launchedsame3c3207. Research directB2session,T16target/T4draftdepth3,400sharedpool andMTPmetadata/headvalidation, sameordinarydecoder ID/state/last_hidden/slots hashcomparison eachoftwogroups/tworepeats. Reused<=1Mwordhosttransferbuffers avoidstatefileduplication; noHTTP/performanceadmission untilstrictgate. No nativeDLL/modelsourcechanges whilelive. Full2x/physicalhigherbatches/tunedrefs/4Brefs/4K/broadquality goals remainopen.


### B2 MTP draft builder workspace intervention (2026-10-24)

The first strict B2 MTP3 gate failed before generation: TensorRT requested 67,119,104 bytes against the default 64 MiB builder pool. The closed failure archive is 0133c76ce0cdea49b41efe5ee6897d685f5f02c3ecbbe8282e2a674f34f5cf43; independent replay passed as negative evidence, with no cases or performance admission. Its HTTP proof was external and hash-bound to the linked archive; future archives embed that dependency.

The bounded candidate adds an explicit 128 MiB MTP builder pool option; default behavior and the separate projection workspace remain unchanged. Three CPU tests pass, including invalid values and graph closure on setter failure. The guarded attempt2 is launched on the unchanged native DLL, with exact Python source hashes, prior correctness proofs, idle-GPU and fixed SSD-cache checks. It requires unchanged retained B1 IDs and complete ordinary target-state hashes on both rows across two resets. No timing or success is claimed while running; raw state arrays are not retained, limiting independent replay to producer hash witnesses.


### Physical B4 T8 candidate preparation (2026-10-24; not executed)

The admitted HTTP pilot physically batches only two rows, and queues excess clients. A separate full-model greedy correctness probe now prepares physical B4 with prefill/decode chunks8, retaining flattened M32 projection geometry. This geometry motivates the candidate but does not prove equivalence of convolution, attention, FP8 numerics, state, or timing. The probe requires all four retained text/image/short-output B1 sequences to match exactly across two resets, including unequal output limits. It reuses the shared compressed checkpoint pool and unchanged canonical policy flags. No runtime source is deployed or GPU job launched while B2 MTP attempt2 remains live. Only syntax validation has run; no B4 result or performance is claimed.


### B2 MTP128 MiB strict gate result (2026-10-24)

Original supervised attempt2 completed with exit0. Archive 3eeb4762dabd13d2c149ffff1e9a81a4465d921670985d72093f8d197e204902 contains41 files (154,295 bytes). Independent archive replay on both machines passed: all eight row/repeat full-ID comparisons match retained B1 sequences, and all four producer-observed complete ordinary/MTP target-state hash comparisons pass. The128 MiB builder intervention allowed draft construction where the prior64 MiB attempt failed. Raw target-state arrays are not retained, so independent replay validates source coverage/hash witnesses rather than reproducing state arithmetic. Native DLL remains3c3207b84b0e048f7e69736f3ae927ffb629c40ffe086994765e9a70ac5f58dd. Owned closed engine files were retired with recorded sizes/hashes. No HTTP streaming, timing, broad quality, true throughput or2x claim follows from this bounded correctness result. Next: integrate gated B2 MTP streaming with explicit equal-limit handling and unequal-limit greedy fallback, qualify protocol/IDs before timing.


### B2 MTP streaming integration candidate (2026-10-24)

A separate research worker requires the independently verified B2 MTP ID/state-hash gate and unchanged native DLL before use. Two real rows with equal output limits use the retained-target MTP3 session with128 MiB draft builder workspace. Single real rows and unequal output limits fall back to the existing greedy loop, preserving dummy-row inactivity and per-row stopping. Every frame resets both target and draft state. The first target prediction is emitted from prefill; the repeated generation callback is checked and suppressed, preventing duplicate first-token events. Completion records assert returned MTP IDs/counts against all streamed IDs including stops and retain speculative counters.

The six-group protocol candidate requires exact retained B1 prompt/output IDs for ten requests, including reset repeats, text/image singles, short color pair, and an unequal two-token/128-token pair. Expected generation mode is checked for every completion. The launcher checks prior terminal status, archive digest, native build/source identity, workspace method hashes, GPU idle and fixed SSD cache ownership. Syntax checks passed; GPU protocol results are pending. No timing or HTTP admission yet.


### B2 MTP protocol replay and matched HTTP preparation (2026-10-24)

While the original batch-mtp-stream-protocol-attempt1 worker remains supervised and live, local archival/replay methods were prepared. Replay binds the native sources, prior strict MTP archive/workspace source hashes, ten retained request/response ID pairs, image hashes, complete stream counts and monotonic arrivals, explicit MTP versus greedy fallback modes, MTP counters, terminal events and sampled adapter-wide VRAM. It does not admit performance.

A guarded HTTP pilot is prepared, not launched: it requires the independent protocol proof and terminal original job, exact workspace methods and unchanged native DLL, then reuses the repaired transport and serialization regression fixtures before matched C1/2/4/8 timing. The workload remains the existing seed3 JPEG image-first binary-grid corpus, two warmup waves, eight requests per cohort and128 output tokens. Single real rows remain greedy; this policy is explicit in metadata, so C1 cannot be mislabeled MTP speed. Actual4K prompt tests and tuned references remain separate outstanding requirements. Only syntax validation has run for these newly prepared methods.


### B2 MTP streaming independent replay passed (2026-10-24)

Original supervised protocol job completed exit0; archive9024b81cccd60a3953b5c2971d3a357d4efed7d4f4bf6ba08326090d1867d356 retains49 files/137,355 bytes. Independent replay passed on both machines: six groups/ten requests, exact prompt/output IDs, monotonic token counters/arrival records, image hashes, partial-row isolation and explicit MTP/greedy-fallback modes. Sampled adapter-wide peak31,643,979,776 bytes is not a process-attributed instantaneous peak. This is protocol admission only, not performance or broad quality. After both replays and confirmed original process closure, guarded batch-mtp-http-pilot-attempt1 was launched on the same native DLL and source/workspace hashes. Same-worker HTTP quality/regression gates precede timing; no speed result is available at launch.


### Matched B2 MTP HTTP replay preparation (2026-10-24; pilot live)

Local archive/replay methods retain the strict MTP source/workspace lineage and independent streaming proof alongside raw HTTP/SSE, worker tokens, dispatch groups, resource samples and ordinary vLLM/SGLang archives. In this equal-limit HTTP corpus, each two-real-row dispatch must record verified_mtp and valid MTP counters; single dispatches must record greedy_target. Each measured SSE completion must match worker IDs and generation mode. Per-cohort results explicitly count MTP versus greedy requests as well as actual physical group sizes. Metric replay and identical request-byte comparisons reuse the prior verified path. The current original pilot remains live; these methods are syntax-checked only and no performance is admitted until terminal archival and both independent replays.


### Guarded physical B4 T8 launch preparation (2026-10-24; not launched)

The earlier unrun B4 T8 retained-ID probe now has a fixed-folder launcher and lifecycle wrapper. It requires the current B2 MTP HTTP pilot to be terminal and independently verified with a hash-matching archive, plus unchanged native DLL/five build-source hashes and prior canonical numerical gates. GPU idle and owned SSD scratch checks remain mandatory. This prevents overlapping the live HTTP worker or moving to a larger batch without preserving its measurements. B4 T8 remains a separate unqualified numerical/memory candidate; only syntax and launch-argument checks have run.


### B2 MTP HTTP pilot independently verified (2026-10-24)

Original supervised job completed exit0, all eight cohorts. Archive bb2a7145657adaaa1ee1e694b026ef17c2db12757a35fbfbb1bc886745ae0d4e retains466 files/1,581,798 bytes. Local independent replay passed. The first remote replay failed to import compare_http_runs because the result directory lacked its module search path; a second replay with the retained methods directory on sys.path passed unchanged data and checks. Both machines verify64 measured SSE/worker pairs, matched request bytes/transport/prompt-output counts against both ordinary references, eight HTTP quality cases, two stream oracles and four serialization regression captures. No GPU rerun was needed for the replay import failure.

| Cohort | TTFT ms | Decode tok/s | Aggregate tok/s | MTP/greedy requests |
|---|---:|---:|---:|---:|
| text-c1 | 159.7317 | 28.080875 | 27.324550 | 0/8 |
| text-c2 | 2475.9712 | 39.994733 | 42.790462 | 4/4 |
| text-c4 | 2812.1804 | 51.376502 | 94.770809 | 8/0 |
| text-c8 | 4009.5516 | 53.353790 | 97.993738 | 8/0 |
| image-c1 | 339.0321 | 27.984452 | 26.254975 | 0/8 |
| image-c2 | 352.6613 | 52.178424 | 89.342210 | 8/0 |
| image-c4 | 3192.9900 | 52.237758 | 88.248960 | 8/0 |
| image-c8 | 4565.5161 | 54.433241 | 90.708603 | 8/0 |

C4/C8 queue physical B2 groups. Text C2 contains four MTP requests and four greedy fallback requests, so its cohort median is not an isolated MTP measurement. C1 is entirely greedy. C8 aggregate versus earlier greedy pilot is text 1.792949x; image 1.728579x; these are separate bounded pilots, not a controlled causal or repeated-study speedup claim. Faster ordinary reference C8 aggregate remains445.768458 text/439.864319 image tok/s. Target2x is unmet. Sampled adapter-wide measured peak reaches31,658,659,840 bytes; not an instantaneous process-attributed peak. Actual4K prompts, broad quality, repeats and tuned fastest reference configurations remain pending.


### Physical B4 T8 original launch and replay preparation (2026-10-24)

After the B2 MTP HTTP pilot closed, was archived and passed independent replay on both machines, guarded shared-full-b4-t8-attempt1 was launched through the original supervised handle. The remote source/runtime remains frozen while it lives. Local archival/replay preparation binds the prior verified HTTP and shared B2 archives, the same native DLL/five native build-source hashes, raw request/response oracles and resource records. It independently verifies that the new probe differs from the retained B2 probe only by the declared B4/T8 profile, guard and scope/docstring changes. All four retained B1 sequences must match across two resets (eight row comparisons); a partial or failed run remains failure evidence. No target-state-array, broader quality, HTTP or timing admission follows from this ID gate. Methods syntax-checked; original job remains live and no B4 result is claimed.


### Physical B4 T8 retained-ID gate passed (2026-10-24)

Original supervised job completed exit0; archivecc0359d98131a98b74c5f0672de8e3c2509e8a96dd6a69fdebdcefb14f794f70 retains36 files/107,210 bytes. Both independent replays passed all eight row/repeat full-ID comparisons against retained B1 oracles, and verified the exact declared source transformation from the archived B2 probe. No target-state hash, broad quality, HTTP or performance claim.

A separate B4 T8 greedy stream worker now requires that proof. Missing physical rows duplicate a valid prefill prompt but stay inactive during generation and emit no events. The protocol candidate uses seven groups/seventeen requests: two full four-row repeats, separate text/image singles, a two-row color group, a three-row ragged text/image/short-output group, and an unequal-limit pair. All real-row output IDs/counts and absence of dummy/retired events must match. Syntax checks passed; no stream or timing result is claimed before the real GPU gate.


### B4 streaming replay and matched HTTP candidate preparation (2026-10-24)

The original batch4-stream-protocol-attempt1 remains supervised/live. Local archive/replay methods bind its native sources, the independently verified B4 retained-ID archive, seven groups/seventeen real request IDs, exact output/prompt IDs and image hashes, greedy mode, monotonic stream counts, no dummy/retired events and sampled resource/terminal records. Ready metadata must state physicalB4, target/prefillT8 and MTP0.

A separate HTTP candidate is prepared but not launched. It requires independent protocol verification and terminal original job, uses BatchWorker capacity4 with the same2ms grouping delay, and has13 HTTP quality requests including two full four-row groups and a three-row partial group before the unchanged stream/serialization regression gates. Timing uses the unchanged seed3 corpus,128 outputs, eight requests, two warmup waves and C1/2/4/8. Actual physical group occupancy must be retained. The current GPU worker and runtime are not modified while live. New methods passed syntax validation only; no B4 stream/HTTP/timing admission is claimed.


### B4 HTTP archival/metric replay preparation (2026-10-24; not run)

While the original B4 streaming worker remains live, local HTTP archival/replay methods were prepared. They require the independently verified B4 streaming and retained-ID archives,13 HTTP quality requests, two stream oracles and four serialization regression captures. Native ready metadata must be physicalB4/T8/MTP0. The replay checks real dispatch membership and records occupancy1/2/3/4 per cohort, requires at least one actual four-row dispatch, checks every measured SSE completion against native IDs/greedy mode, and replays all raw metrics/resource samples plus identical requests versus ordinary references. Client concurrency is not treated as physical occupancy. Both methods are syntax-checked only; no B4 HTTP measurements exist yet.


### B4 streaming independently verified; HTTP pilot launched (2026-10-24)

Original protocol job completed exit0. Archive9da7cd2d64ca72439b1a7b0061fe1badba9f1deeb27f934146f8054a31f1d3dd retains43 files/138,958 bytes. Independent replay passed on both machines for seven groups/seventeen requests, exact prompt/output IDs, image hashes, partial-row isolation/no dummy events and sampled resource/terminal records. Sampled adapter-wide peak31,365,058,560 bytes is not instantaneous or process-attributed. Native DLL unchanged3c3207. Only after both replays and original closure, guarded batch4-http-pilot-attempt1 was launched in the existing fixed study folder. It requires13 same-worker HTTP quality cases and the unchanged stream/serialization regression gates before timing. No B4 speed result is available at launch.


### Physical B4 T4 candidate preparation (2026-10-24; not run)

The current B4 T8 HTTP pilot remains live. A local B4 T4 retained-ID candidate reduces both target graph token dimensions from8 to4, reusing the existing shared compressed pool/native DLL. For a four-row graph this changes flattened projection M from32 to16. Motivation is to reduce masked token-slot projection work; no speed or numerical-equivalence claim is inferred from dimensions. The same four mixed text/image/short-output oracles and two resets remain mandatory. The wrapper requires the B4 T8 HTTP pilot to finish, archive and independently verify before launch, plus unchanged canonical build/source and idle-GPU/SSD guards. Syntax checks passed; not deployed or launched. Wider prefill, MTP and actual4K prompts remain distinct unqualified profiles.


### B4 HTTP live observer note (2026-10-24)

The original pilot remains live, with all13 HTTP quality cases/two streams/four serialization captures passed and the first three text timing cohorts recorded. Sampling gaps occur in retained timing/resource data; no causal root is proven. One auxiliary read-only Windows GPUAdapterMemory CIM query was issued after the observed textC1 gap and returned before textC2 completion was observed. Its counter values and command are transcribed in evidence/batch4-http-cim-observation.json (SHA256 9f749a6b78c20d15e931fc9727de60af70c6f13f24ac46990a827d7b963ef7f3). An exact observation timestamp was not captured; this is not a raw timestamped export, process attribution, DXGI budget or eviction evidence. The extra observer differs from baseline monitoring and is retained as a possible measurement confound. Do not silently discard pauses or claim clean repeated causal speedups from this pilot. No further such observer queries are planned during the live measurements.


### B4 pilot observer retention and T4 replay preparation (2026-10-24)

Local B4 HTTP archive now requires the auxiliary CIM observation sidecar, and metric replay explicitly marks an extra observer during timing and clean-repeat requirement. This preserves the limitation rather than silently comparing a changed monitoring protocol as a clean repeat. Original job remains live; archive/replay method edits are local and do not alter the executing worker.

The unrun B4 T4 candidate has local archive/replay preparation. It binds the prior B4 T8 ID proof and latest B4 HTTP archive, checks the declared T8-to-T4 source transformation against retained source, and retains strict full output-ID equality across both resets. No numerical, memory or performance result exists for T4 yet. Syntax validation passed.


### Physical B4 T8 HTTP pilot independently replayed (2026-10-24)

Original job completed exit0/all eight cohorts. Archive 97c0c3e617c253d5d56fdbf980bb3a1052ba7714f7019b4dcaba7d1721832e13 retains471 files/1,559,766 bytes, including the explicitly transcribed auxiliary CIM observer sidecar. Both machines independently replayed64 SSE/native-worker pairs,13 HTTP quality requests/two streams/four serialization regression captures, native source/runtime lineage, actual group occupancy, metric arithmetic, resource samples and identical request bytes/transport/prompt-output counts against ordinary references. Sampling gaps and the extra observer are retained; clean_repeat_required=true. This is verified pilot evidence, not a clean repeat or causal comparison.

| Cohort | TTFT ms | Decode tok/s | Aggregate tok/s | Real-row group counts 1/2/3/4 |
|---|---:|---:|---:|---:|
| text-c1 | 325.9582 | 22.514346 | 19.385193 | 8/0/0/0 |
| text-c2 | 6301.2518 | 22.498819 | 22.702702 | 6/1/0/0 |
| text-c4 | 298.2323 | 15.436868 | 41.691157 | 1/0/1/1 |
| text-c8 | 3361.0608 | 21.763860 | 83.409857 | 0/0/0/2 |
| image-c1 | 677.6543 | 22.512011 | 20.222029 | 8/0/0/0 |
| image-c2 | 673.3359 | 22.104832 | 32.133533 | 2/3/0/0 |
| image-c4 | 676.2233 | 21.965258 | 53.139202 | 0/2/0/1 |
| image-c8 | 7227.8325 | 21.763047 | 52.382222 | 1/0/1/1 |

C8 text83.409857/image52.382222 tok/s is below the earlier B2 MTP pilot97.993738/90.708603 and the faster ordinary references445.768458/439.864319; these are different execution profiles and separate pilots, not a causal batch-size ablation. Target2x remains unmet. Only after both replays and original closure, guarded shared-full-b4-t4-attempt1 was launched on unchanged native DLL/source, reducing token slots from8 to4. No T4 numerical or speed result exists at launch. Broader quality/repeats/actual4K/tuned references remain pending.


### B4 T4 conditional streaming candidate (2026-10-24; not run)

Original shared-full-b4-t4-attempt1 remains live and unqualified. A separate local stream candidate changes only the guarded retained-ID proof and target graph/ready token dimensions from8 to4 relative to the B4 T8 greedy worker. It retains the same seven-group/seventeen-request protocol, dummy-row inactivity, reset/early-stop handling and exact prompt/output-ID requirements. The wrapper requires successful terminal original T4 job and independent T4 ID proof before construction. Native DLL, quantization, image path, capacity4096 and request corpus are unchanged. Syntax checks passed; no source is deployed over the live worker and no T4 stream/timing result is claimed.


### Physical B4 T4 retained-ID gate independently passed (2026-10-24)

Original supervised T4 job completed exit0. Archive503ec1c0cf56f85af02d3e159660b8b818cf07bea227bf3bbfb4286edcbc9712 retains37 files/105,373 bytes. Independent replay on both machines passed all eight row/repeat output-ID comparisons against retained B1 oracles, the declared source transformation, linked prior archives and native runtime/source/resource/terminal lineage. This gate does not establish full target-state equality, broad quality or performance. After both replays and original closure, guarded batch4-t4-stream-protocol-attempt1 was launched using the prepared seventeen-request corpus. No T4 streaming or timing result exists at launch.


### B4 T4 streaming replay and matched HTTP preparation (2026-10-24; not run)

Original batch4-t4-stream-protocol-attempt1 remains live. Local archive/replay preparation carries the same seven groups/seventeen requests and binds the independently verified T4 ID proof; ready metadata must be physicalB4/T4/MTP0. The next HTTP candidate requires successful original closure and independent protocol replay, then the unchanged13-case quality corpus/two stream cases/four serialization captures before identical C1/2/4/8 timing. Archive/metric replay retain actual physical group occupancy1..4, source/runtime/proof hashes, raw SSE/native IDs, metric arithmetic and all sampling gaps. Unlike the earlier T8 pilot, no auxiliary CIM observation is planned; metadata explicitly marks no auxiliary observer. Independent repeats/broad quality/actual4K and tuned reference requirements remain pending. All new methods passed syntax validation only; no T4 streaming or HTTP result is claimed. No live runtime/source mutation.


### B4 T4 streaming independently passed; matched HTTP launched (2026-10-24)

Original streaming job closed exit0. Archive04cde6f44242b09af5fc264b138d95967d31745c67cbc748bef26654cf8cd194 retains43 files/130,322 bytes. Both machines independently verified all seventeen prompt/output-ID sequences, seven full/partial/reset/unequal-limit groups, image hashes/no dummy events, native readyT4 profile and source/runtime/resource/terminal lineage. Sampled adapter-wide peak31,362,195,456 bytes is not instantaneous or process-attributed. Only after both replays and original closure, guarded batch4-t4-http-pilot-attempt1 was launched in the fixed study folder using unchanged native DLL and shared model assets. Same-worker13-case HTTP quality/two streams/four serialization captures precede identical timing. No auxiliary CIM observer is planned for this pilot; no T4 performance result exists at launch.


### Deferred canonical-state MTP candidate preparation (2026-10-24; not run)

The retained B2 MTP128 MiB report records canonical_state_backup_bytes1,391,460,352 and a separate packed replay output of1,391,460,352 bytes. Allocator tracking is disabled, so these named logical allocations do not account for whole-process peak. Source audit finds the existing default-off deferred_state_commit path: the verifier borrows unchanged canonical state until final commit and omits its separate canonical backup. This motivates a strict B2 gate before attempting larger MTP batches; no measured memory saving or speed claim follows.

The prepared probe opts into that existing flag only, retaining the same B2 T16/draftT4/128 MiB builder, quantization/runtime, four oracle rows/two resets, complete ordinary-versus-MTP state hashes and retained B1 output IDs. Hash reads additionally require no pending verifier/decoder transaction. The wrapper requires the current B4 T4 HTTP pilot to close, archive and independently verify before launch. No native or common decoder/session source is edited. Syntax checks passed; no deferred27B correctness/streaming/timing result exists.


### Deferred-state archival/replay preparation (2026-10-24; not run)

While the original T4 HTTP pilot remains live, local deferred-state replay methods were prepared. They bind its eventual verified archive and the retained non-deferred strict B2 MTP archive, require byte-identical six core Python methods plus unchanged native sources/runtime, and verify the exact declared experiment-source changes. Success still requires all retained IDs and complete ordinary/MTP state hash witnesses across two resets, closed transactions at hash reads, deferred mode metadata and zero named canonical backup bytes. The result cannot establish whole-process memory savings or performance from logical allocation metadata. Raw target-state arrays are still not retained. Syntax validation passed; no deferred GPU test has run and no live runtime/source is changed.


### B4 T4 HTTP pilot independently replayed; deferred-state gate launched (2026-10-24)

Original pilot completed exit0/all eight cohorts. Archive a065bec1839c4ce001737e274cb9038c18ba0787dd11e44c4e02b468fe98ea0d retains471 files/1,569,109 bytes. Independent replay passed on both machines for64 SSE/worker pairs, thirteen HTTP quality cases/two streams/four serialization captures, source/runtime/proof lineage, actual occupancy, metric/resource arithmetic and identical requests/transport/counts against both ordinary references. No auxiliary CIM observer ran during this timing. Large sampling gaps remain, reaching21.9991333 seconds in imageC4; retain them, do not remove pauses or infer a causal T4 speed comparison. A read-only post-run System event query over the prior hour found zero Display/NVIDIA/WHEA matches and zero query errors; its JSON is archived. This does not rule out paging, stalls or other causes.

| Cohort | TTFT ms | Decode tok/s | Aggregate tok/s |
|---|---:|---:|---:|
| text-c1 | 522.8968 | 23.045113 | 20.877911 |
| text-c2 | 3523.6284 | 22.898377 | 26.944289 |
| text-c4 | 507.9289 | 21.120722 | 78.390586 |
| text-c8 | 3650.1646 | 22.047870 | 81.724569 |
| image-c1 | 1257.6239 | 14.223868 | 12.168385 |
| image-c2 | 1339.1085 | 10.801433 | 12.073384 |
| image-c4 | 1306.5570 | 14.657596 | 20.237039 |
| image-c8 | 15146.2293 | 9.351911 | 26.815535 |

The pilot establishes no improvement: C8 text81.724569/image26.815535 tok/s remains below the earlier B2 MTP pilot and ordinary references, and2x remains unmet. Only after original closure/archival/both independent replays and the separate post-run query, guarded shared-mtp3-full-b2-deferred-attempt1 was launched on unchanged native/core sources. This is a strict retained-ID/ordinary-state-hash experiment for the existing deferred-state option, not timing. Independent repeats, broader quality, actual4K prompts and tuned references remain pending.


### Follow-up memory plan from retained allocations (2026-10-24; design only)

The retained shared-mtp3-full-b2-attempt2 external_compressed_pool records embedding_table_bytes=2,542,796,800, embedding_head_tied=false, untied_head_bytes=715,161,600 and head_precision=modelopt_nvfp4_w4a4. Source cuda_tensor_io.cpp Open uploads the full BF16 embedding table independently of the packed head; Embedding/EmbeddingChunk gather its rows on device. This is a distinct potential memory target after the currently live deferred-state gate; no native/common source is changed now.

Proposed implementation, contingent on subsequent evidence and qualification:

1. Add a default-off native host-embedding mode with persistent read-only checkpoint mapping and explicit model-open state. Require the already validated independent NVFP4 head; never substitute an invalid GPU tensor as an open-model sentinel or silently fall back to a missing tied head. Apply before target construction/capture.
2. Preserve both embedding APIs. Validate every ID and shape before mutation, gather only requested BF16 rows on CPU with exact bit-preserving FP32 conversion, and upload those rows to the existing GPU outputs. Preserve chunk time-major token input, batch-major output, inactive/padded-slot zeros and stream/lifetime rules. Include CPU lookup/H2D cost inside request timings.
3. Keep head precision, weights and full vocabulary verification unchanged. Disable/close transitions must be atomic, preserve a working prior mode on allocation failure and fail closed if a head transition would require an absent GPU table. Report host mapping versus GPU table bytes separately; mapped size is not resident CPU RSS.
4. Verify original GPU gather versus the new path using independent BF16 byte conversion, boundary/duplicate/random real checkpoint rows, B1/2/4/8 chunk shapes and invalid-ID/shape/lifetime cases. Preserve default behavior and run native memory/race checks where supported.
5. Requalify real text/image outputs and full ordinary/MTP target-state hashes under4096 capacity before streaming/HTTP/performance. Measure GPU memory, host RSS/transfer costs, TTFT/decode/real concurrency including image encoding. Do not precompute visual inputs outside measured work or reduce supported context to manufacture a gain.

The named2.54GB allocation motivates the design; no whole-process saving, equivalence, latency or2x claim is made. If the option is implemented it needs a new native build receipt and fresh gates; existing3c3207 proofs do not automatically qualify a changed DLL. Current deferred-state experiment remains isolated.


### Deferred-state B2 MTP strict gate independently passed (2026-10-24)

Original job closed exit0. Archive76602b940062fca89c22ea65266f2443e89e5a7f028df8067914cbe813aff1d0 retains43 files/145,722 bytes. Both independent replays pass eight retained B1 output-ID comparisons and four complete ordinary/MTP target-state hash witnesses, closed transactions, declared probe transformation, six byte-identical core Python methods, native build/source/proof/resource/terminal lineage. Deferred canonical backup reports zero in all four cases versus1,391,460,352 bytes in each retained non-deferred case. Sampled adapter-wide peaks are30,292,647,936 versus31,658,659,840 bytes across separate correctness runs; this observed difference is not an instantaneous/process-attributed saving or performance claim. Raw state arrays are not retained; arithmetic replay remains limited to source/hash witnesses.

A separate B2 deferred-MTP streaming worker now opts into the existing flag and requires this strict proof. It preserves T16 target/T4 draft/128 MiB builder, equal-two-real-row MTP policy, single/unequal-limit greedy fallback and the same six-group/ten-request protocol. Ready metadata records deferred mode. Syntax validation passed; no streaming or timing admission yet. No native/common algorithm/default is changed.


### Deferred-state streaming replay and HTTP preparation (2026-10-24; not run)

The original deferred B2 MTP streaming job remains live. Local archival/replay preparation binds its independently verified deferred-state strict ID/hash proof, six core method hashes and unchanged native runtime, plus the same six groups/ten streaming requests and explicit MTP-versus-greedy modes. Ready metadata must record deferred mode. A matched HTTP candidate is prepared behind successful original closure and independent protocol replay. It retains the original equal-two-row MTP/single-or-unequal greedy policy,8 quality requests/two streams/four serialization captures, seed3 image/text corpus,128 outputs, two warmup waves and C1/2/4/8 timing. Replay checks deferred metadata, actual occupancy/modes, raw SSE/native IDs and metric/resource arithmetic versus identical ordinary-reference requests. No auxiliary CIM observer is planned. Only syntax validation has run; no deferred streaming/HTTP/performance admission is claimed and no live runtime source is changed.

### Deferred-state B2 streaming gate verified; matched HTTP pilot launched (2026-10-24)

Original supervised streaming job closed exit0 before archival or deployment. Archive0773f7861e1b64a42b0b5c7ce9ccd92d24f7aaab23efb2ed4d468107ba650383 retains49 files/135,371 bytes. Local and remote independent replays both passed six groups/ten requests: exact generated/prompt IDs and image hashes, repeated mixed pairs, singles, early completion, unequal limits and explicit MTP/greedy fallback modes. Sampled adapter-wide correctness peak30,277,967,872 bytes; no timing admission. Native library remains3c3207b84b0e048f7e69736f3ae927ffb629c40ffe086994765e9a70ac5f58dd. Only isolated HTTP study methods were deployed after terminal closure; no common/native source changed.

The matched batch-mtp-deferred-http-pilot-attempt1 was dispatched on its original supervised handle. Its runner requires the above independent protocol proof and original ledger closure plus the strict deferred state/hash gate, unchanged runtime/source pins and idle GPU/SSD checks. Retains the same short text/image requests, C1/2/4/8,128 outputs, warmup and transport correctness checks. No deferred throughput claim until terminal evidence is archived and independently replayed.

A user-facing C8 comparison chart was regenerated from verified JSONs, including B2 MTP3 and B4 T8/T4 greedy pilots versus ordinary references. B4 sampling gaps and the T8 auxiliary CIM observer are disclosed; no clean repeat, causal gain or2x claim. Current best B2 C8 throughput remains97.994 text/90.709 image versus ordinary vLLM445.768/439.864 tok/s.

### Deferred HTTP verifier sampler visibility (2026-10-24; pilot still live)

Original supervised batch-mtp-deferred-http-pilot-attempt1 remains running on direct handle poll. Added local-only verifier diagnostics for per-cohort recorded sampler count and maximum consecutive monotonic interval, asserting timestamp ordering. This preserves sampling gaps explicitly without interpreting them as paging, eviction or GPU stall durations. Clean independent repeats remain required. The same arithmetic replay on retained B4 T4 image-C4 samples reproduces the recorded21.9991333-second maximum gap; this is existing evidence, not a new timing result. Syntax and whitespace checks passed. No live source, DLL, runtime or benchmark configuration was altered, and no deferred timing result has been admitted.

### Reproducible historical B1 comparison charts (2026-10-24)

Added chart_verified_b1_comparisons.py to reproduce the historical B1 wins and native model/precision tradeoffs from an explicitly supplied retained-evidence directory. It verifies existing chart-data links to the joined/4B proof bytes before plotting. Local rendering completed and the earlier visual QA remains representative of unchanged figure content. The user-facing Garnet-B1-Charts-Provenance.json separately binds generator, all five exact input JSONs and four PNG/SVG artifacts. Historical joined27B MTP3 decode72.259/74.157 tok/s exceeds ordinary faster-reference68.482/68.498 by5.5%/8.3%; text TTFT64.719ms versus152.361ms is2.354x faster, image103.715ms versus fastest156.384ms is1.508x. These are bounded B1 pilots, not2x decode/concurrency admission, tuned-reference superiority or long-context/quality parity. Full concurrency charts continue to disclose the opposite aggregate-throughput result. The original deferred HTTP job remains live on direct handle poll; no runtime changes or new performance admission.

### Default-off host embedding implemented locally; native qualification pending (2026-10-24)

Implemented local CUDATensorIO host embedding mode behind enable_host_embedding and packed decoder host_embedding_lookup=False. It requires an open checkpoint and independent native NVFP4 head; default GPU lookup is retained. An explicit model-open flag replaces table-residency sentinels for weight/projection/head APIs. Enabling prepares a persistent read-only mapping and validates BF16 geometry before device synchronization and GPU-table retirement. Disabling allocates/uploads its replacement before releasing the working mapping; disabling the native head during host mode is rejected. Host row conversion preserves BF16 bits into IEEE FP32, validates all IDs/shapes before output mutation, preserves batch/time layout and padded zeros, and completes H2D before temporary host storage expires. Memory receipts separate GPU-table bytes from mapped tensor bytes, without treating mapping size as resident RSS. Decoder opt-in enables native head/host lookup before graph construction; default constructor makes neither new call.

Four CPU constructor/policy tests pass (default, opt-in order, head/mapping failure cleanup, explicit-bool admission). Initial test harness failed on unrelated CUDADecoder class aliases; extraction was narrowed to constructor/close and rerun. This was a harness failure, not native/model evidence. Source hashes and final raw CPU output are retained in evidence/host-embedding-policy-preparation.json. Syntax/whitespace checks pass. No native compilation, byte/numerical/lifecycle GPU qualification, deployment, memory saving or performance admission is claimed. No heavy local compilation was run during the other local study's quiet window. The original remote deferred HTTP handle remains running; local source edits have not been copied there. Next build must wait for its terminal closure and independent archival/replay, preserve the3c3207 runtime/source, and require fresh default-off, embedding, full-model state/ID, streaming and HTTP gates before timing.
