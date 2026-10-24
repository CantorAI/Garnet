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
