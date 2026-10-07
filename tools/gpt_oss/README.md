# GPT-OSS GPU validation

Work from the CantorAI workspace, keeping Garnet, xlang3, and ThirdPartySDK as
siblings. The feature branch is `feature/gpt-oss-120b`. Do not add model weights
or generated engines to Git.

## Verified Linux configuration

On October 5, 2026, the original GPT-OSS-120B checkpoint completed greedy
Harmony generation on two RTX PRO 6000 Blackwell Max-Q GPUs (SM 120), using
TensorRT 10.16.1.11 and the XLang3 Release runtime at `990a745`. The automatic
planner assigned layers 0–17 to GPU 0 and 18–35 to GPU 1. The synthetic suite
and the Linux model integration runner passed, including cached two-GPU
generation, inactive batch slots and existing Qwen compatibility checks.
The pretrained smoke prompt produced a final `Hello!` response. This establishes
execution of the full checkpoint; reference-logit parity, sustained serving,
larger contexts and production throughput remain unverified.

Use the selected TensorRT SDK for both Garnet and its native operator plugin.
The tested container had CUDA 13.4, but TensorRT's generated kernel compilation
failed with its NVRTC 13.4 library. Selecting NVRTC 13.2.86 resolved that failure
without rebuilding the model code or replacing the container's CUDA toolkit:

```bash
python3 -m pip install --no-deps --target ThirdPartySDK/cuda-nvrtc-13.2 \
  nvidia-cuda-nvrtc==13.2.86
export LD_LIBRARY_PATH="$PWD/ThirdPartySDK/cuda-nvrtc-13.2/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
```

For these GPUs, pass `--cuda-architectures 120` to the build command below.
Use `test/xlang3/run_model_tests.sh` for the broader Linux integration suite.

## Optimized operator paths

The GPT-OSS plugin uses cooperative warp reductions for decode expert dot
products, router logits and paged attention. Prefill buckets routed slots by
expert and runs grouped BF16 Tensor Core GEMMs, unpacking MXFP4 weight tiles
into shared memory. It reuses each tile across 32 routed rows for shorter
prompts and 64 rows for prompts longer than 512 tokens. The full expert weights remain packed.
Single-token decode defaults to 16 splits per query head on the tested RTX PRO
6000 Blackwell GPUs, with four warps in each split and a stable softmax merge.
Set `GARNET_GPT_OSS_DECODE_SPLITS` to `8`, `16`, `32`, or `64` before starting
the runtime to tune it; set `0` for the unsplit path. The unsplit path uses
16 warps by default and accepts `GARNET_GPT_OSS_DECODE_WARPS` of `1`, `2`, `4`,
`8`, or `16`.
Router scoring defaults to 256 threads on the tested Blackwell GPUs. Set
`GARNET_GPT_OSS_ROUTER_THREADS=128` to use the previous block size.
For paired TP2 engine experiments, `GARNET_GPT_OSS_TRT_OPT_LEVEL` sets the
TensorRT builder optimization level (0–5, default 1), and
`GARNET_GPT_OSS_TRT_WORKSPACE_MB` sets its workspace cap (256–4096 MiB,
default 256). Use a separate engine cache for each configuration.
These paths require SM 80 or newer and the xModel's BF16-rounded activations;
FP32 accumulation, intermediate BF16 rounding, learned attention sinks,
sliding windows and router-order combination are preserved. Plugin version 2
uses a larger prefill workspace; rebuild engines after upgrading the plugin.

Kernel tests cover partial expert tiles, concentrated routing and ties,
multiple K tiles, batches of 1, 3, 17 and 65 rows, and attention dimensions
8, 64 and 128. Pretrained correctness and performance must also be measured
after kernel changes.

For a local decode-MoE CUDA-graph screen, build the standalone plugin's
`garnet_gpt_oss_marlin_decode_benchmark` target. Run it in separate processes
with `GARNET_GPT_OSS_FUSED_DECODE_ROUTE=0`, then `=1`, then `=1` plus
`GARNET_GPT_OSS_FUSED_MARLIN_LOCK_CLEAR=1`. It reports graph node count and
L2-flushed replay time for synthetic GPT-OSS-sized expert weights. These
flags are experimental and must be checked with all four pretrained prompts
on the target GPU before they can become defaults.

For comparisons, finish and record the reference runtime first, shut it down
and confirm GPU memory is released, then run Garnet alone. Reuse identical
Harmony input IDs, greedy sampling, stop IDs, output limits and BF16 KV precision;
disable reference prefix caching. Report the parallel topology explicitly:
Garnet currently uses sequential layer stages, while a TP2 reference splits
operators. Separate engine initialization and cold compilation from warm
inference. The driver reports `decode_wall_seconds`, including host control
updates, in addition to forward-only `decode_seconds` and
`decode_prepare_seconds` for the prefill-to-decode engine transition.

## Local build

Use an existing XLang3 Release runtime. This avoids rebuilding or changing
XLang3. Run in a Visual Studio x64 developer shell, using a CPython executable
and CMake available on your machine:

```powershell
python Garnet/tools/gpt_oss/build.py `
  --runtime-executable out/build/x64-Release/bin/xlang3.exe `
  --runtime-library out/build/x64-Release/bin/xlang3_runtime.lib `
  --runtime-dll out/build/x64-Release/bin/xlang3_runtime.dll `
  --tensorrt-root ThirdPartySDK/TensorRT `
  --build-dir out/build/gpt-oss-ready --cuda-architectures 89

python Garnet/tools/gpt_oss/verify.py `
  --runtime-dir out/build/gpt-oss-ready/bin `
  --work-dir work/gpt-oss-verification --tensorrt-root ThirdPartySDK/TensorRT
```

The build emits `garnet.dll`, `garnet_gpt_oss.dll`, the runtime executable,
and `plugins/registry.json`. The verification tool writes individual logs and
checks GPU kernels, full synthetic xModel prefill, cached batch decode, cache
reload/refit, plugin rejection cases, and greedy generation against an
independent CPU reference. It also exercises existing Qwen text/VL capture
checks and native CPU/CUDA APIs with a deliberately missing plugin registry.

The original production-capture script has a stale total-module count (27;
the current tree imports 29). The focused compatibility check reuses its
existing text/VL capture functions without changing Qwen files or that test.
ASR/TTS are outside this GPT-OSS compatibility check.

## Rented Linux GPU through SSH

Once a rented instance provides an SSH host, user, port, and key, connect using
those details. Transfer the current source changes and use the same sibling
workspace layout. No SSH host or credentials are embedded in these scripts.

For the first full checkpoint test, use a CUDA GPU with enough measured VRAM
for the packed checkpoint, KV cache, and engine workspace. Host RAM during
TensorRT compilation/refit must also be measured. The pipeline planner supplies
an estimated GPU memory budget; full-model memory use and performance still
require measurement on the target hardware.

Build an XLang3 Linux Release runtime using its repository instructions, then
build Garnet against it. TensorRT 10 headers and libraries are required:

```bash
python3 Garnet/tools/gpt_oss/build.py \
  --runtime-executable /path/to/xlang3 \
  --runtime-library /path/to/libxlang3_runtime.so \
  --tensorrt-root /path/to/TensorRT \
  --build-dir out/build/gpt-oss --cuda-architectures native

export LD_LIBRARY_PATH="$PWD/out/build/gpt-oss/bin:/path/to/TensorRT/lib:$LD_LIBRARY_PATH"
python3 Garnet/tools/gpt_oss/verify.py \
  --runtime-dir out/build/gpt-oss/bin \
  --work-dir work/gpt-oss-verification --tensorrt-root /path/to/TensorRT
```

Alternatively, the operator library builds independently:

```bash
cmake -S Garnet/plugins/gpt_oss -B out/build/gpt-oss-plugin \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native \
  -DGARNET_TENSORRT_ROOT=/path/to/TensorRT
cmake --build out/build/gpt-oss-plugin --parallel 4
ctest --test-dir out/build/gpt-oss-plugin --output-on-failure
```

After synthetic checks pass, download the official **original** checkpoint
layout separately. The Transformers checkpoint layout is not accepted:

```bash
hf download openai/gpt-oss-120b --include 'original/*' --local-dir models/gpt-oss-120b
python3 -m pip install openai-harmony
python3 Garnet/tools/gpt_oss/prepare_prompt.py 'Say hello.' work/request.json --max-new-tokens 32
out/build/gpt-oss/bin/xlang3 Garnet/tools/gpt_oss/run_tokens.py \
  models/gpt-oss-120b/original work/gpt-oss-engine-cache work/request.json work/result.json
```

The driver releases prefill before loading decode, avoiding two simultaneous
resident copies of the 120B weights. It accepts Harmony IDs and stop IDs,
generates greedily, and records token IDs and timings. It caps total context at
4096 tokens and output at 256 for the first correctness tests. It does not
start a network service or claim pretrained parity. Compare logits/tokens with
the official implementation, measure memory and latency, then evaluate larger
contexts, batch schedules, and expert-kernel optimization.

Reference architecture and weight layout:
[OpenAI model](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py),
[OpenAI weights](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/weights.py).

The GPT-OSS extension is an ordinary XLang3 native module. xModels import
`garnet_gpt_oss` and call `garnet.bind_operator_module(garnet_gpt_oss)`.
Verification includes import, repeated binding, unrelated-module rejection,
and explicit binding with a deliberately missing registry. Cached engines use
the requirement mapping to import and bind the module before deserialization.
For a standalone plugin build, provide `GARNET_XLANG_RUNTIME_LIBRARY` pointing
to the existing XLang3 runtime import library (Windows) or shared library (Linux).

## Automatic multi-GPU layer placement

`run_pipeline_tokens.py` detects visible GPUs, reads actual packed checkpoint
sizes, budgets local BF16 KV pages and working buffers, and chooses contiguous
layer ranges that fit each GPU individually. Each stage has its own engine,
weights and KV cache. Only activations and request controls cross GPUs. No
Qwen xModel files are changed.

```bash
python3 Garnet/tools/gpt_oss/verify.py \
  --runtime-dir out/build/gpt-oss/bin --work-dir work/gpt-oss-verification \
  --tensorrt-root /path/to/TensorRT --multi-gpu
out/build/gpt-oss/bin/xlang3 Garnet/tools/gpt_oss/run_pipeline_tokens.py \
  models/gpt-oss-120b/original work/gpt-oss-pipeline-cache work/request.json work/result.json
```

The request can include `"device_ids": [0, 1]`, `"memory_fraction": 0.9`, and
`"reserve_mb": 1024`. By default all visible GPUs are used; select at most one
GPU per transformer layer. The builder saves `placement.json`, generated stage
xModels and engines under the hardware/placement cache key. Runtime engine
fingerprints additionally validate source, weights, shapes and plugin binary.
Changing available memory can change the placement; identical placement on
identical hardware reuses the cache.

The batched throughput runner records batch size, actual input/output token
counts, maximum KV context, KV allocation, sampled GPU memory and separate
decode/full-request throughput. `GARNET_BATCH_CONTEXT_CAPACITY` selects the
context limit; `GARNET_BATCH_PREFILL_CHUNK` selects an opt-in chunk size and
budgets activation memory for that largest executed chunk. Match context,
BF16 KV and generated-token limits with the vLLM runner for each comparison.
Set `GARNET_BATCH_PREFILL_WARMUP=1` for a warmed prefill comparison; excluded
warmup calls remain in the result. `GARNET_BATCH_DECODE_TRIALS=3` measures
three decode windows from the same input KV prefix and validates every slot
in every window. These are decode repeats, not repeated full requests.
The continuous decode wall timer excludes progress logging and memory-query
subprocesses. Execution throughput still excludes engine loading/handoff;
consult its separate timing fields and measurement limitations.
`GARNET_GPT_OSS_PROFILE_PREFILL=1` brackets each measured prefill chunk with
the CUDA profiler API, after any enabled warmup. Use an Nsight capture that
supports repeated ranges for a chunked request. Retain separate unprofiled
trials for throughput: instrumentation can substantially perturb timing.
The vLLM runner supports `VLLM_BATCH_TRIALS=3` for three full-request trials
and `VLLM_BATCH_MAX_BATCHED_TOKENS=8192` to screen the official recipe's
larger prefill budget. Its repeats perform a fresh prefill without prefix
caching; retain the fastest supported reference configuration.

`GARNET_GPT_OSS_INLINE_BATCH_CONTROLS=1` opts into rank-thread vector updates
for batches up to 64. The generic native bridge
`garnet.tensor_update_int_vectors_async(tensors, values)` accepts 1–4 dense
CUDA INT32/INT64 tensors with 1–64 elements each and matching nested integer
lists. It validates all values before launching one kernel. Enqueue the next
tensor use on the same execution thread/stream; use Garnet's tensor lease and
completion-event path when transferring ownership to another thread.

`GARNET_GPT_OSS_DIRECT_BATCH_ALLREDUCE=1`, together with
`GARNET_GPT_OSS_DIRECT_ALLREDUCE=1`, enables an experimental two-GPU peer
reduction for hidden-state messages up to batch 64. The current scratch is
communicator-wide, so only paired, serialized rank streams are supported;
overlapping independent executions must use NCCL. Screen it with
`garnet_gpt_oss_tp_direct_benchmark <batch> 50` and
`GARNET_GPT_OSS_DIRECT_BATCH_CTAS=2|4|8|16|32` before a pretrained comparison.
The benchmark checks changing inputs on both GPUs as well as graph timing.
The additional `GARNET_GPT_OSS_DIRECT_LARGE_BATCH_ALLREDUCE=1` gate permits
batch65–128 messages for a separate capacity/throughput screen. Without it,
messages above batch64 retain NCCL dispatch. Staging grows by720KiB/GPU;
the serialized paired-stream restriction still applies. Validate changing-input
graph parity and target timing at the larger size before pretrained use.

`GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS=1` opts into rank-local original
expert constants. Routing stays global; each rank stores only its even/odd
experts and repacks those constants for Marlin. Derived refit weight names
regenerate the same bytes from the original checkpoint on engine reload;
no transformed checkpoint files are needed. This changes the engine/cache
identity and uses GPT-OSS plugin serialization version8. Validate cold and
cached graph execution, fallback kernels, Marlin and pretrained outputs
before using this candidate for throughput. The generic TensorRT weight
refitter recognizes only the explicit GPT-OSS derived-name protocol.
`screen_expert_shards_tp2.sh REQUEST EXPECTED VLLM_REFERENCE RESULT_DIR BATCH OUTPUT`
tests unchunked prefill and chunk128 with1/2/4 Marlin CTAs per SM sequentially,
validating every slot/trial before continuing. It requires a newly recorded
vLLM reference with identical input token IDs, batch, output limit and context;
it preserves failure logs and refuses to overwrite existing evidence.
Marlin row admission and prefill block size affect workspace bytes and buffer
offsets, so they are part of the placement/cache profile. Builder workspace
sizing uses the larger of grouped fallback and aligned Marlin scratch; at
4096 GPT-OSS rows, Marlin needs more scratch than the grouped path. Generate
a fresh profile if these layout settings change before loading an engine.

`GARNET_GPT_OSS_MARLIN_DECODE_BLOCK=8` or `32` independently selects the
expert tile for decode batches of at least128 rows. An unset/other value
preserves the existing row-based tile choice. The xModel passes the prefill
phase explicitly to the MoE plugin, and the override is part of the workspace
profile/cache identity. Screen both choices with actual batch routing and
matched input/output lengths; a repeated-prompt batch and a diverse batch
can favor different tiles. This is an experimental override, not a measured
default improvement.

The vLLM batch runner uses a read-only worker extension to record physical
KV backing bytes, block/layout settings and per-layer full/sliding history
estimates outside timed inference. Shared tensor views are counted once by
storage address. Logical retained history excludes page rounding and
in-flight tokens; completed requests have already freed scheduler blocks
when the RPC runs. Keep this estimate distinct from measured occupancy,
and keep sampled device memory distinct from PyTorch allocator peaks.

After a matched batch128 vLLM reference has completed and all outputs pass,
`screen_batch_tiles_tp2.sh REQUEST EXPECTED REFERENCE RESULT_DIR 128 OUTPUT CHUNK`
screens decode tiles8/32 and MoE CTAs1/2/4 sequentially. Input must divide
evenly into chunks with at most4096 flattened rows. It preserves failed
logs, validates every slot/trial before advancing, and inherits the exact
context/IDs from the reference. The peer collective uses the separately
gated128-row capacity and32 CTAs; this remains a serialized-stream experiment.

`garnet_gpt_oss_attention_batch_benchmark BATCH CONTEXT_LENGTH REPEATS` isolates
one GPU's36-layer decode attention pass (18 full,18 sliding128), with distinct
KV backing per layer and ten graph warmups before five timed replay trials.
Run each split/warp setting in a fresh process, without competing inference.
It checks sampled double-precision references, all output finiteness and an
inactive slot, with permuted pages and a missing page. Full native parity is
still required for each setting. The GPU-event timings include KV writes,
attention and merges, excluding host reference work; they are not serving
throughput. Distinct KV prevents repeatedly timing one layer entirely in L2.

The Linux build can enable the experimental two-rank GPT-OSS NCCL collectives
and run their two-GPU parity checks with `--enable-nccl`. The TP2 inference
path shards attention heads and vocabulary rows, keeps rank-local MoE work,
and uses reductions plus a final logits all-gather. Dense checkpoint weights
are still loaded from the full checkpoint on each rank, and performance and
correctness must be measured on the rented GPUs.

Backbone APIs: `cuda_devices_json()` reports per-GPU free/total memory,
architecture, PCI bus ID and P2P capability; `cuda_set_device(id)` selects the
calling thread's device and returns its previous ID; `tensor_to_device(t, id)`
performs a checked transfer; `tensor_zeros(shape, dtype)` allocates directly on
the selected GPU. The generic planner/scheduler is `python/garnet_pipeline.py`.
GPT-OSS-specific size estimates and graph instantiation are in
`tools/gpt_oss/pipeline.py` and `xModel/gpt_oss/120b/stage.py`.

The generic pipeline scheduler executes stages sequentially and fences their
per-thread streams. P2P copies are used where supported; otherwise transfers
go through host RAM. It does not overlap requests or shard arbitrary
operators. An insufficient-memory plan fails before engine generation.
Estimates include headroom but remain estimates: full 120B engine memory,
build host RAM, pretrained parity and speed must be measured on the rented
GPUs. Prefill engines are released before decode engines load; local KV caches
are retained.

The token-generation runner selects only the last prompt hidden state before
the final norm and vocabulary projection. `build_pipeline` retains full logits
by default; `last_token_logits=True` selects this generation profile with its
own engine cache directory. Both modes write the same complete prompt KV cache.
The vocabulary projection requests BF16 operands with FP32 accumulation and
returns BF16-rounded logits. This keeps the checkpoint's BF16 output weights
compressed to two bytes per element. The generic linear lowering accepts an
explicit `compute_dtype` of `bfloat16` or `float32`; its default is unchanged.

Single-token decode attention splits each head's context across eight warps
and merges stable softmax states, counting the learned sink once. Prompt
attention still parallelizes query rows. Native parity tests include long
paged decode, missing pages, sliding windows and inactive cache slots.
GPT-OSS decoder-layer fusions request TensorRT CUDA Graph replay for repeated
decode calls. Set `GARNET_DECODE_CUDA_GRAPH=0` to disable replay when diagnosing
or benchmarking this path; the setting is scoped to the GPT-OSS model runtime.

`garnet_gpt_oss_kernel_benchmark` measures full-size 2880-dimensional,
128-expert/top-8 MoE calls with L2 flushed outside each timed invocation.
It compares warp and grouped Tensor Core paths and checks their outputs.
This is a kernel diagnostic, not an end-to-end serving benchmark; grouped
expert execution remains selected for 16 or more tokens.

For the batch128 shared-KV decode candidate, run `screen_batch_attention_tp2.sh
REQUEST_JSON EXPECTED_JSON VLLM_REFERENCE_JSON RESULT_DIR 128 OUTPUT_TOKENS
PREFILL_CHUNK`. It fixes the measured expert32/CTA1 profile and screens
GQA4/8/16 sequentially, validating every slot/trial before advancing. The
reference must use exact matching input IDs, output lengths and BF16 TP2;
existing logs/results are preserved rather than overwritten.

To attribute a remaining gap, use the ordinary batch launcher with
`GARNET_BENCH_NSYS_OUTPUT=/workspace/CantorAI/work/profiles/<new-name>` and
`GARNET_GPT_OSS_PROFILE_DECODE_STEPS=32`. The default capture begins at decode
offset10; `GARNET_GPT_OSS_PROFILE_DECODE_START=480` captures the last32 steps
of a512-output request (processed context reaches input length+511). Use
`GARNET_GPT_OSS_PROFILE_PREFILL=1` for prefill instead. Preserve the chosen
kernel/environment settings and use a separate result JSON for each profile.
The launcher retains exclusive GPU access, refuses report overwrite and
stops collection at the bounded range while allowing validation to finish.
JSON records the range/report path; instrumented throughput must not be used
as the unprofiled vLLM comparison. `nsys` must already be installed.
`profile_batch_tp2.py REQUEST EXPECTED UNPROFILED_GARNET_RESULT NEW_OUTPUT_PREFIX
START STEPS` replays the recorded optimization environment, checks exact
input IDs, preserves output length/context and validates outputs after the
capture. Use the vLLM Python environment for tokenizer validation. It reduces
repeats to one diagnostic trial and refuses existing reports/results. For
input256/output512, `START=480 STEPS=32` captures processed context736..767.

`GARNET_GPT_OSS_PARALLEL_MARLIN_METADATA=1` is an opt-in batch candidate.
At 16 or more rows, it counts and stably scatters expert assignments with
one warp per expert, preserving ascending row order and padding. It reuses
router-logit scratch after routing; workspace and small-batch paths stay the
same. Run the native parity executable with `--metadata-benchmark` for exact
CPU/legacy/parallel checks and separate 36-layer CUDA graph timing. This is
an isolated diagnostic, not evidence of model throughput.

Use `screen_batch_metadata_tp2.py REQUEST EXPECTED UNPROFILED_GARNET_RESULT
NEW_RESULT_DIR` with the vLLM Python environment to replay a recorded shape.
It runs legacy metadata first and parallel metadata second, three decode
trials each, validates all answers, and requires every output token sequence
to match the legacy run. It records prefill, decode, complete-request rates,
memory and KV fields through the ordinary exclusive launcher. It refuses
existing result directories and keeps failed logs and comparisons.

`paired_batch_suite.py UNPROFILED_GARNET_PROFILE NEW_RESULT_DIR [CASE ...]`
runs memory admission without loading engines, then all requested vLLM cases,
then all Garnet cases. Default cases are saved code tracing, instruction
following and long context; the unchanged arithmetic reference can be reused.
It replays the selected Garnet flags and batch/output profile, matches input
IDs and BF16 KV, rounds longer context up to512-token capacity, validates every
slot/trial, and retains separate logs, admission plans and a progress manifest.
Use the vLLM Python environment; verify its installed version is the current
supported stable release before launch. Both engines retain exclusive GPU
access through their ordinary launchers. Setup/handoff costs are reported
separately: Garnet's execution rate is not cold full-request wall throughput.
`GARNET_BATCH_PLAN_ONLY=1` makes the Garnet batch runner write admission evidence
and exit before creating tensors, engines or inference work.

The opt-in `GARNET_GPT_OSS_ROUTER_QUERY_TILE=2` or `4` shares each expert
weight row across that many queries, preserving the original scalar256
accumulation/reduction order. It applies at16 or more rows, hidden width<=4096,
experts<=128 and the256-thread router setting; other profiles retain their
existing path. Workspace and options serialization do not change. Append
`router` to the `screen_batch_metadata_tp2.py` arguments to screen tiles0/2/4
sequentially and require exact whole token trajectories plus answer validation.
Use native `--router-benchmark` for bitwise score/top4/probability checks and
separate36-layer score-kernel timing. Kernel timing is diagnostic only.

The batch runners and profile helpers accept batch1..512 and fixed output16..2048,
with input plus output constrained by context<=4096. These are harness limits,
not tested throughput maxima. `paired_batch_suite.py PROFILE NEWDIR CASE...`
accepts explicit `--batch`, `--output`, `--prefill-chunk` and `--context` overrides.
For example use `arithmetic --batch 256 --output 512 --prefill-chunk 16 --context 1024`.
The actual Garnet planner runs before all vLLM cases, followed by all Garnet cases;
invalid/OOM admission preserves requested shapes rather than shrinking them.

Experimental exact-order prefill `GARNET_GPT_OSS_PREFILL_GQA_QUERY_TILE=2/4`
requires tiled head64 attention and8:1 GQA. It shares paged KV across query heads
and multiple queries per warp. Local timing regresses; keep it disabled pending
target isolation. `--prefill-gqa-benchmark [BATCH]` checks exact output/cache
parity and reports36-layer graph timings at short/long context and full/sliding
windows. Four cache layers are reused and query/output buffers differ perlayer.

Optional native tensor-core prefill builds with
`-DGARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL=ON` (CUDA12.9+), then opt in through
`GARNET_GPT_OSS_PREFILL_FLASHINFER=1`. Default build and runtime keep this off.
The pinned Apache-2.0 header subset and provenance live under the GPT-OSS plugin.
It needs no PyTorch/TVM/Python runtime. Invocation-owned workspace packs BF16 Q,
safe page indices and a fixed-capacity device-generated schedule; original
page masks, active rows, causal offsets, sliding windows and one learned sink
are retained. Tensor-core/probability rounding differs from scalar attention.
Native numerical and pretrained quality/performance validation is required.
Native benchmark label `tile=8` denotes FlashInfer, including conversions and
KV write, rather than an eight-query scalar tile. The optional `make_fixture.py
DEST --gqa8` fixture exercises head64/8:1 GQA through compiled_parity.py.

The large-prefill BF16 router is independently opt-in through
`GARNET_GPT_OSS_PREFILL_ROUTER_TENSORCORE=1`; it dispatches only for prefill
with at least1024 flattened rows, leaving decode on the existing router.
Reduction order can change scores and selected experts. After building,
run `bash tools/gpt_oss/verify_prefill_router.sh NEW_EVIDENCE_DIRECTORY`
on an idle two-GPU Linux target. It holds the benchmark lock, checks native
scores/sorting/probabilities and Compute Sanitizer, runs full compatibility
tests, then validates independent CPU-prefix logits cold/warm at a synthetic
batch512. The cold trace must confirm the actual router kernel ran. This
synthetic shape does not establish pretrained batch512 admission. Quality,
throughput, KV/memory and engine lifecycle comparisons remain separate.

After these gates pass, append `prefill-router-wire` to
`screen_batch_metadata_tp2.py REQUEST EXPECTED UNPROFILED_GARNET_RESULT NEW_DIR`
to replay a recorded intermediate-axis TP2 profile sequentially with
router/wire flags `0/0`, `1/0`, and `1/1`. Other recorded settings remain fixed.
Each mode validates every expected answer and records all three complete token
matrices, memory/KV, prefill and decode timing. Router reduction-order changes
are recorded; the wire-only comparison must retain identical trajectories.
The saved reference must match input IDs and context, and measurements remain
unprofiled. A new batch/length shape still needs optimized vLLM first.

For a separate Nsight diagnostic, `GARNET_GPT_OSS_PROFILE_PREFILL=1` captures
one timed prefill chunk after its optional warmup. Select its zero-based index
with `GARNET_GPT_OSS_PROFILE_PREFILL_CHUNK` (default0). Later long-context chunks
can expose attention costs that the first chunk misses; an invalid index fails
before execution. The index and instrumentation flag are recorded in the raw
result. Profiled timings stay separate from throughput comparisons.

The BF16 pack/unpack memcheck retains explicit CUDA API reporting and uses
`nccl_initialization_api_suppressions.xml` only for observed NCCL initialization
backtraces: unavailable-kernel209 attribute probes and already-enabled-peer704
clearing. Every suppression requires the specific API, error code and NCCL
caller frames. Device-memory errors and Garnet callers are not suppressed.
Full unsuppressed diagnostic XML remains in the research evidence; the final
run also saves XML. New errors must be investigated rather than broadening
these exclusions to make a gate pass.

The `prefill-router` screen mode runs just router0/1 with BF16 communication
explicitly off. It can isolate router quality/performance after its own native,
memory and compiled-model gates, while an independent transport experiment is
still pending. It retains the same three-trial answer and trajectory evidence.
