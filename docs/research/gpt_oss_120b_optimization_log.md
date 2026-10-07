# GPT-OSS-120B optimization research log

This is an append-only record of performance hypotheses, implementations, correctness checks, and measured outcomes. Entries below were reconstructed on 2026-10-06 from [the contemporaneous optimization notes](../gpt-oss-120b-optimization.md) and raw benchmark artifacts; dates and commit IDs are marked unknown where the evidence does not establish them. Future experiments should be appended in chronological order, without rewriting earlier entries. Measurements from different hosts are not interchangeable. “Prefill” in Garnet's runner is execution time; vLLM's time to first token (TTFT) includes a wider serving boundary. Unless noted, rates are warmed wall-clock decode for one request, greedy sampling, batch 1, TP2, and two sequential trials per saved prompt on the specified host.

## Optimization summary

| ID | Component | Optimization | Baseline | Result | End-to-end impact | Decision |
|---|---|---|---:|---:|---|---|
| OPT-0001 | TP inputs | Keep decode inputs resident on each rank | 188.91–190.91 tok/s | 199.59–200.51 tok/s | About 5% arithmetic gain on Japan host | ACCEPT, opt-in |
| OPT-0002 | MoE prefill | Marlin block 8 → 32 | ~0.505 s long prefill | ~0.446 s | ~59 ms long-prefill reduction | ACCEPT, opt-in |
| OPT-0003 | Attention prefill | More warps per query | 0.4464 s | 0.4475–0.4501 s | None | REJECT |
| OPT-0004 | Attention prefill | Fast exponential | 0.4463–0.4467 s | 0.4389–0.4390 s | ~1.7% prefill gain, changed tokens | INCONCLUSIVE, opt-in |
| OPT-0005 | Attention prefill | Tiled query/KV reuse | 0.655/0.706 s | 0.462/0.465 s | Large long-prefill gain on Workstation | ACCEPT, opt-in |
| OPT-0006 | TensorRT | Level 3 / larger workspace | Prior setting | 6.3% / 8.6% slower | Regression on measured host | REJECT |
| OPT-0007 | NCCL | Protocol overrides | ~200 tok/s | 175–178 tok/s or failure | Regression | REJECT |
| OPT-0008 | TP collective | Direct peer reduction | 198.18–199.69 tok/s | 202.95–204.22 tok/s | 1–3% arithmetic gain on Japan host | ACCEPT, opt-in |
| OPT-0009 | Decode controls | Batch four scalar updates | 72.5–78.0 µs/cycle, isolated | 25.1–27.2 µs/cycle | Full-model impact measured later | ACCEPT, opt-in |
| OPT-0010 | Decode controls | Inline rank-local update kernel | 189.27–190.90 tok/s | 192.86–193.84 tok/s | 1–3% on Workstation | ACCEPT, opt-in |
| OPT-0011 | MoE decode | Fuse routing and lock clear | 188.47/189.69 tok/s | 193.84/192.86 tok/s | Small arithmetic gain, combined flags | ACCEPT, opt-in |
| OPT-0012 | Attention decode | 8 or 32 splits vs 16 | 193.84/192.86 tok/s | 187.41–187.98 / 191.42–193.92 | No reliable gain | REJECT |
| OPT-0013 | MoE decode | Warp-local top-K | 193–196 tok/s | 193–196 tok/s | Noise-level differences | REJECT |
| OPT-0014 | Attention prefill | Stage 32 vs 16 KV positions | 0.462/0.465 s | 0.470/0.461 s | No gain | REJECT |
| OPT-0015 | MoE prefill | cuBLAS FP32 router GEMM | ~0.46 s | 0.520/0.510 s | Regression; profiler failure | REJECT |
| OPT-0016 | TP communication | BF16 attention-prefill all-reduce | 0.4622–0.4650 s | 0.4582–0.4589 s | ~4.2 ms paired median gain | ACCEPT, opt-in |
| OPT-0017 | Layer fusion | Residual add + RMSNorm | 193–196 tok/s short | 193–195 tok/s short | No material gain | REJECT |
| OPT-0018 | TensorRT | Builder level 5 | 193–196 tok/s short | 197–199 tok/s short | ~2% decode gain; changed token sequences | INCONCLUSIVE |
| OPT-0019 | Dense decode | GPT-OSS BF16 GEMV plugin | 193–196 short, 170–171 long tok/s | First integration regressed; revised 195–198 short, 173–174 long tok/s | Revised path gave small gains, changed instruction tokens; far below vLLM | INCONCLUSIVE, opt-in |
| OPT-0020 | Batch serving | Matched batch/input/output/KV profile | Batch 1 only | Warmed Garnet b8 989 vs vLLM 1226 decode tok/s; long b4 Garnet 574 vs vLLM 512 but full-request 131 vs 470 tok/s | Found batch and long-prefill gap; auto workspace fit added | MEASUREMENT IN PROGRESS |
| OPT-0021 | Batched long prefill | Extend Marlin from 4096 to 8192 rows | 3.03 s long b4 prefill | 1.37 s, but 2/4 slots missed/wrong final answer | Correctness failure despite 55% prefill gain | REJECT, revert |
| OPT-0022 | Batched prefill | Chunk long input below Marlin row cutoff | 3.030 s prefill, 130.8 full-request tok/s | 1.431 s, 202.2 tok/s; all four answers pass, token sequences differ | Faster complete request; decode regressed, repeat pending | INCONCLUSIVE, opt-in |
| OPT-0023 | Batched TP reduction | Peer reduction with CTA-count screen | NCCL b8/32/64: 9.19/22.64/32.66 µs/op; corrected b32 decode 2596–2623 tok/s | Direct16 plus inline controls: 2760–2792; all 96 trial/slot token sequences match | ~4.0% median decode gain over controls alone; full execution 961 → 974 tok/s; other shapes pending | INCONCLUSIVE, opt-in |
| OPT-0024 | Batched decode controls | Inline integer-vector updates on rank threads | Corrected b32 decode 2596–2623 tok/s | 2637–2670; synthetic suite and all 96 trial/slot checks pass | ~1.8% median decode gain; full execution 959 → 961 tok/s; other shapes pending | INCONCLUSIVE, opt-in |
| OPT-0025 | TP2 weight storage | Store only owned original-layout experts per rank | Full constants plus half-model Marlin repack; chunk128 b32 OOM | Full synthetic suite and96 arithmetic outputs/mode pass; sampled peak falls97095→68207MiB/GPU; chunk128 prefill0.828–0.842s | Memory fix enables chunked prefill; token trajectories differ and other workloads pending | INCONCLUSIVE, opt-in |
| OPT-0026 | Batch128 expert scheduling | Separate decode8/32-row tile and screen CTAs1/2/4 | Legacy tile selected by flattened row count | Six modes pass384 arithmetic checks each; best median3238.51 decode tok/s vs vLLM5394.05 | Best complete execution2821.48 tok/s; still behind reference | INCONCLUSIVE, opt-in |
| OPT-0027 | GQA decode attention | Share paged KV tiles across eight query warps | Matched b128 median3238.51 decode tok/s | First GQA4 median4580.18, all384 arithmetic checks pass; other settings pending | +41.4% decode, complete execution3745.44; still below vLLM | INCONCLUSIVE, opt-in |

## Cumulative accepted-stage history

These rows show the measurable serving trajectory within each host. Changes in host, baseline flags, or benchmark boundary start a new series. Only accepted optimizations advance a series; opt-in status does not imply production readiness.

| Stage | Host / accepted change | Arithmetic warm tok/s | ITL | Long prefill | Memory |
|---|---|---:|---:|---:|---:|
| J0 | Japan TP2 baseline before rank-local inputs | 189.94/190.91/188.91 | ~5.24–5.29 ms | Not measured here | Not recorded |
| J1 | + OPT-0001 rank-local inputs | 199.59/200.51/199.82 | ~4.99–5.01 ms | Not measured here | Not recorded |
| J2 | + OPT-0008 direct reduction (paired NCCL baseline 198.18/198.29/199.69) | 204.22/203.72/202.95 | ~4.90–4.93 ms | Not measured here | Not recorded |
| W0 | British Columbia Workstation, batched-copy controls and optimized MoE/attention flags | 190.90/189.27 | ~5.24–5.28 ms | Not recorded for this exact control path | Not recorded |
| W1 | + OPT-0010 inline controls, OPT-0011 fused routing/lock clear | 193.84/192.86 | ~5.16–5.18 ms | 0.462/0.465 s | Not recorded |
| W2 | + OPT-0016 BF16 prefill collective; decode path unchanged | 197.81/193.39 | ~5.06–5.17 ms | 0.4582–0.4589 s paired long trials | Not recorded |

The W2 arithmetic decode spread is run variation; the BF16 path applies to prefill only. The optimized vLLM 0.31.0 comparison on the Workstation was about 273 tok/s warm decode, with ~27–36 ms short TTFT and ~115–116 ms long TTFT. Garnet prefill is not identical to TTFT. Raw results and Nsight reports live outside Git under `D:/CantorAI/work/vast-54543362/`; earlier Japan artifacts are under `D:/CantorAI/work/gpt-oss-benchmark-evidence-2026-10-05/`.

To reproduce the current Workstation comparison from the built runtime and staged checkpoint, run `GARNET_BENCH_OUTPUT=/workspace/CantorAI/work/<new-directory> GARNET_TP_RANK_LOCAL_INPUTS=1 GARNET_GPT_OSS_DIRECT_ALLREDUCE=1 GARNET_GPT_OSS_INLINE_CONTROL_KERNEL=1 GARNET_GPT_OSS_MARLIN_PREFILL_BLOCK=32 GARNET_GPT_OSS_MARLIN_MAX_TOKENS=4096 GARNET_GPT_OSS_PREFILL_TILED_64=1 GARNET_GPT_OSS_PREFILL_QUERY_TILE=16 GARNET_GPT_OSS_FUSED_DECODE_ROUTE=1 GARNET_GPT_OSS_FUSED_MARLIN_LOCK_CLEAR=1 bash Garnet/tools/gpt_oss/benchmark_tp2.sh` from `/workspace/CantorAI`, with no competing GPU process. Add `GARNET_GPT_OSS_TRT_OPT_LEVEL=5` only for OPT-0018, or `GARNET_GPT_OSS_BF16_PREFILL_ALLREDUCE=1` only for OPT-0016. The script records its environment beside JSON and logs; the saved decoded answers were produced afterward with the tokenizer helper. The optimized vLLM configuration and three-trial JSON results are under `D:/CantorAI/work/vast-54543362/vllm-benchmarks/`.

## Experiments

### OPT-0001: Rank-local TP2 decode inputs

**Date / Commit:** 2026-10-05 or earlier; exact commit not reconstructed, `feature/gpt-oss-120b`. Relevant files: `python/garnet_pipeline.py`, `tools/gpt_oss/` runner.
**System / Workload:** GPT-OSS-120B, Japan 2× RTX PRO 6000 Max-Q, BF16 dense/MXFP4 MoE, TP2, batch 1; saved arithmetic and four-prompt suite, warm decode. Other configuration and memory figures: see contemporaneous notes; not fully reconstructed.
**Baseline / Observed bottleneck:** Per-token copies/allocations for rank 1's token and control tensors. Arithmetic old path 189.94/190.91/188.91 tok/s.
**Hypothesis / Proposed optimization:** Keep token and controls resident per rank, then update only values. Before: host prepares and copies each rank's inputs each token. After: each rank reuses its own device tensors.
**Implementation:** Added a rank-local execution path selected by `GARNET_TP_RANK_LOCAL_INPUTS=1`.
**Correctness validation:** All four expected final-answer checks passed. Exact token comparison and numerical error were not recorded.
**Performance result:** Arithmetic 199.59/200.51/199.82 tok/s; about 5% faster. TTFT, memory, GPU utilization, and kernel counts not measured for this isolated change.
**Decision / Analysis:** ACCEPT as opt-in. Copy/allocation avoidance helped, but GPU graph and TP communication remained dominant.
**Next step:** Reduce TP collective and per-token GPU work.

### OPT-0002: Increase Marlin prefill row block

**Date / Commit:** 2026-10-05 or earlier; exact commit not reconstructed, `feature/gpt-oss-120b`. Relevant files: `plugins/gpt_oss/cuda/` Marlin path, GPT-OSS runner.
**System / Workload:** GPT-OSS-120B, Japan TP2 Max-Q, BF16/MXFP4, batch 1; short and 2,005-token long prompt, repeat prefill.
**Baseline / Observed bottleneck:** Eight-row Marlin scheduling contributed to prefill. Short ~0.055 s and long ~0.505 s.
**Hypothesis / Proposed optimization:** Increase rows processed per block from 8 to 32, amortizing scheduling and improving expert matrix utilization.
**Implementation:** Added opt-in `GARNET_GPT_OSS_MARLIN_PREFILL_BLOCK=32` and shape limit `GARNET_GPT_OSS_MARLIN_MAX_TOKENS=4096`; corrected the shape-compatible column tile after an initial 256-column candidate corrupted token 0.
**Correctness validation:** Final four answers passed after correction. The corrupted first variant was not accepted.
**Performance result:** Short ~0.050 s, long ~0.446 s. A 64-row follow-up stayed ~0.446 s and was rejected. TTFT and memory not measured at a comparable boundary.
**Decision / Analysis:** ACCEPT 32-row opt-in; REJECT 64-row follow-up. The larger tile helped until scheduling ceased to be the dominant cost.
**Next step:** Optimize attention's long-prompt KV traversal.

### OPT-0003: Split prefill queries across additional warps

**Date / Commit:** 2026-10-05 or earlier; exact commit not reconstructed. Relevant file: `plugins/gpt_oss/cuda/gpt_oss_kernels.cu`.
**System / Workload:** Japan TP2, 2,005-token saved prompt, batch 1; two repeated prefill runs per setting.
**Baseline / Observed bottleneck:** One warp per query walks prior keys serially; prefill 0.4464/0.4464 s.
**Hypothesis / Proposed optimization:** Two or four warps per query might shorten the serial attention scan.
**Implementation:** Experimental warp splitting in the GPT-OSS prefill kernel.
**Correctness validation:** Same first token, `200005`, across variants; full numerical error not recorded.
**Performance result:** Two warps 0.4496/0.4501 s; four warps 0.4478/0.4475 s. No gain. Raw data: `D:/CantorAI/work/gpt-oss-benchmark-evidence-2026-10-05/tp2-prefill-warp-df6bf28/`.
**Decision / Analysis:** REJECT and revert. Extra coordination/work outweighed the shorter per-warp scan.
**Next step:** Reuse KV data across neighboring queries rather than subdividing one query.

### OPT-0004: Fast exponential in prefill softmax

**Date / Commit:** 2026-10-05 or earlier; exact commit not reconstructed. Relevant file: `plugins/gpt_oss/cuda/gpt_oss_kernels.cu`.
**System / Workload:** Japan TP2, saved 2,005-token prompt, batch 1, two paired repeated prefill runs.
**Baseline / Observed bottleneck:** Exponential evaluation is repeated throughout online softmax; baseline 0.4463/0.4467 s.
**Hypothesis / Proposed optimization:** CUDA `__expf` might reduce arithmetic latency without changing final task answers.
**Implementation:** Opt-in `GARNET_GPT_OSS_PREFILL_FAST_EXP=1`.
**Correctness validation:** CUDA parity and all four final-answer checks passed; first token matched. Full token sequences and lengths differed in later full runs, so strict equivalence is unproven.
**Performance result:** 0.4390/0.4389 s, ~1.7% lower prefill. Raw data: `D:/CantorAI/work/gpt-oss-benchmark-evidence-2026-10-05/tp2-prefill-fastexp-19c13e1/`.
**Decision / Analysis:** INCONCLUSIVE for default use; retain opt-in. The approximate math helps a narrow kernel path but changes some generation trajectories.
**Next step:** Test numerical behavior on broader prompts before promotion.

### OPT-0005: Tiled attention prefill with staged KV offsets

**Date / Commit:** 2026-10-05 to 2026-10-06; exact implementation commits not reconstructed. Relevant file: `plugins/gpt_oss/cuda/gpt_oss_kernels.cu`.
**System / Workload:** Local RTX 4080 isolated screen, then British Columbia 2× RTX PRO 6000 Workstation TP2 full model; 2,005-token prompt, batch 1, two runs per end-to-end variant.
**Baseline / Observed bottleneck:** Untiled attention revisited KV for neighboring queries. Untiled Workstation long prefill 0.655/0.706 s; warmed Nsight attention summed ~239.9/237.4 ms across GPUs on an earlier host.
**Hypothesis / Proposed optimization:** Reuse a shared-memory KV tile across 8–16 neighboring queries and stage each page-table offset once per 16-position tile.
**Implementation:** Opt-in `GARNET_GPT_OSS_PREFILL_TILED_64=1`; hardware-choice query tile `GARNET_GPT_OSS_PREFILL_QUERY_TILE=16`. Preserved online softmax and learned sink behavior.
**Correctness validation:** CUDA parity for paged full/sliding cases; 99.9988% exact element matches on a 4,106,240-element synthetic output, max absolute difference one BF16 step (0.00390625), none outside tolerance. All six Workstation final answers passed.
**Performance result:** Local isolated full attention 20.50 → 10.80 ms/call at eight queries; Workstation end-to-end long prefill untiled 0.655/0.706 s, 8-query 0.468/0.469 s, 16-query 0.462/0.465 s. The improved warm Workstation profile still summed ~99.3 ms attention over 36 layers. vLLM long TTFT was ~0.116 s at a different boundary.
**Decision / Analysis:** ACCEPT as opt-in. KV reuse delivered a large full-model gain; remaining attention, Marlin, router and communication work limits prefill.
**Next step:** Larger attention redesign and paired TTFT instrumentation.

### OPT-0006: TensorRT level 3 and larger workspace

**Date / Commit:** 2026-10-05 or earlier; no source change required. Relevant file: `tools/gpt_oss/pipeline.py`.
**System / Workload:** Japan TP2 GPT-OSS-120B, batch 1 warm decode; exact trial counts and raw metrics not reconstructed from surviving notes.
**Baseline / Observed bottleneck:** BF16 projection GEMMs were a substantial decode kernel group. The existing TensorRT builder level/workspace was the comparison.
**Hypothesis / Proposed optimization:** A broader tactic search or workspace might choose faster matrix kernels.
**Implementation:** Experimental builder level 3 and larger workspace, one variable at a time.
**Correctness validation:** Not recorded in the surviving summary; this limits interpretation.
**Performance result:** Level 3 about 6.3% slower; larger workspace about 8.6% slower. Absolute rates, TTFT, memory, and tactic names not recorded.
**Decision / Analysis:** REJECT those settings on the tested host. More search space did not imply better execution for these shapes.
**Next step:** Profile tactic selection on the actual target hardware.

### OPT-0007: Force NCCL protocols

**Date / Commit:** 2026-10-05 or earlier; environment-only trials. Relevant file: `plugins/gpt_oss/cuda/tp_collective.cpp`.
**System / Workload:** Japan TP2 GPT-OSS-120B, batch 1 warm decode; exact trial count not reconstructed.
**Baseline / Observed bottleneck:** TP all-reduce appeared in each layer; default NCCL selection yielded ~200 tok/s.
**Hypothesis / Proposed optimization:** LL128 or Simple protocol might reduce small-message latency.
**Implementation:** NCCL protocol overrides; Tree also screened.
**Correctness validation:** LL128 and Simple produced benchmark rates; final-answer details not preserved. Tree caused TensorRT `enqueueV3` failure.
**Performance result:** LL128 ~177.7 tok/s; Simple ~175.3 tok/s; Tree failed. No comparable TTFT or memory figures.
**Decision / Analysis:** REJECT overrides and keep NCCL selection. Protocol tuning based solely on message size missed full-graph behavior.
**Next step:** Test a controlled direct reduction, including graph capture and full-model timing.

### OPT-0008: Direct two-GPU peer reduction

**Date / Commit:** 2026-10-05 or earlier; exact commit not reconstructed. Relevant files: `plugins/gpt_oss/cuda/tp_direct.cu`, `plugins/gpt_oss/cuda/tp_collective.cpp`.
**System / Workload:** Japan TP2 RTX PRO 6000 Max-Q, BF16 dense/MXFP4 MoE, batch 1, 1,000-op microbenchmarks then four saved full requests.
**Baseline / Observed bottleneck:** NCCL all-reduces cost ~7.92 µs/op in the standalone test and ~0.66–0.70 ms/token in an earlier graph trace.
**Hypothesis / Proposed optimization:** For 2,880 FP32 values across two peer-access GPUs, a specialized vectorized reduction could avoid NCCL protocol overhead.
**Implementation:** System-scope release/acquire signals, 128-bit vector reads, peer staging; opt-in `GARNET_GPT_OSS_DIRECT_ALLREDUCE=1` for 2,880-element decode reductions. An initial scalar prototype at 14.3 µs was replaced by a 2-CTA vectorized version.
**Correctness validation:** Standalone outputs matched NCCL; all four final answers passed. Concurrent-request scratch safety remains unvalidated.
**Performance result:** Kernel-only NCCL 7.92 → direct 4.85 µs/op; direct with staging copy 6.23–6.24 vs NCCL 7.89–7.96 µs/op. Full arithmetic NCCL 198.18/198.29/199.69 vs direct 204.22/203.72/202.95 tok/s; code 198.52 → 202.62, instruction 198.82 → 201.33, long 173.95 → 176.69. Raw: `D:/CantorAI/work/gpt-oss-benchmark-evidence-2026-10-05/tp2-direct-5a7a78d/`.
**Decision / Analysis:** ACCEPT opt-in for the present single-request path. Graph-launch and staging overhead reduced the isolated win to 1–3% end to end.
**Next step:** Make staging request-safe and test concurrent decode.

### OPT-0009: Batch rank-local scalar control updates

**Date / Commit:** 2026-10-05 to 2026-10-06; exact commit not reconstructed. Relevant files: Garnet native tensor update API, `python/garnet_pipeline.py`, runner.
**System / Workload:** Local RTX 4080 isolated CUDA copy test, then target TP2 correctness suite; four scalar controls per rank, 10,000 rounds per local trial.
**Baseline / Observed bottleneck:** Four `tensor_update_from_host` calls per rank each synchronized the stream: eight waits per token.
**Hypothesis / Proposed optimization:** Enqueue four copies under one tensor lease and synchronize once per rank.
**Implementation:** Experimental batched scalar-update API and benchmark flag `GARNET_GPT_OSS_BATCH_CONTROL_UPDATES=1`.
**Correctness validation:** Batched scalar check passed in the full synthetic suite. Full-model result was later measured as the W0 stage with batched copies.
**Performance result:** Local copy-only path 72.5–78.0 → 25.1–27.2 µs/rank per cycle. Target isolated contribution cannot be separated from other W0 flags in surviving data.
**Decision / Analysis:** ACCEPT as an opt-in safe stepping stone. CUDA copy savings do not equal model throughput savings.
**Next step:** Enqueue all controls with one rank-local kernel, avoiding host copies.

### OPT-0010: Inline rank-local control kernel

**Date / Commit:** 2026-10-06; exact commit not reconstructed. Relevant files: native scalar-update API and `python/garnet_pipeline.py`.
**System / Workload:** British Columbia 2× RTX PRO 6000 Blackwell Workstation, BF16/MXFP4 TP2, batch 1; four saved prompts, two trials each, same engines and flags apart from control update.
**Baseline / Observed bottleneck:** 32-token Nsight window recorded 320 host-to-device copies and 160 stream synchronizations with batched-copy control, and 193.64 ms GPU-0 span.
**Hypothesis / Proposed optimization:** Enqueue one small kernel per rank before its model graph to write token, position, context length and slot, removing CPU copy/wait overhead.
**Implementation:** `GARNET_GPT_OSS_INLINE_CONTROL_KERNEL=1`, selected in the rank-local runner.
**Correctness validation:** All four expected final answers passed twice; synthetic scalar-update check passed.
**Performance result:** Arithmetic 190.90/189.27 → 193.84/192.86; code 190.61/190.78 → 195.86/194.78; instruction 190.31/190.46 → 194.30/194.50; long 169.27/168.45 → 170.22/170.69 tok/s. Profiled 32-token GPU-0 span 193.64 → 188.31 ms while summed kernels stayed ~120.5 → 120.4 ms.
**Decision / Analysis:** ACCEPT opt-in; reduced host staging but left per-layer GPU work essentially unchanged.
**Next step:** Reduce graph nodes or expensive per-layer kernel groups.

### OPT-0011: Fuse MoE routing and Marlin lock clearing

**Date / Commit:** 2026-10-06; routing candidate `0a5e1cb`, later lock-clear commit not reconstructed. Relevant files: `plugins/gpt_oss/cuda/gpt_oss_kernels.cu`, Marlin plugin path.
**System / Workload:** British Columbia Workstation TP2, batch 1 arithmetic, two full-model runs per variant; isolated L2-flushed graph screen.
**Baseline / Observed bottleneck:** Separate router-score, metadata and lock-clear graph nodes per MoE layer; no-fusion arithmetic 188.47/189.69 tok/s with direct reduction and inline controls.
**Hypothesis / Proposed optimization:** Have router-score CTAs write routing metadata and clear their own Marlin lock slice, reducing launches and dependencies.
**Implementation:** `GARNET_GPT_OSS_FUSED_DECODE_ROUTE=1` and `GARNET_GPT_OSS_FUSED_MARLIN_LOCK_CLEAR=1`.
**Correctness validation:** CUDA parity in sharded/unsharded cases, full synthetic suite, expected arithmetic final answer in each paired run.
**Performance result:** Isolated graph nodes 9 → 7 → 6; L2-flushed medians 0.7935 → 0.7912 → 0.7917 ms (no useful isolated gain). Full arithmetic no fusion 188.47/189.69, routing 191.59/191.59, routing+locks 193.84/192.86 tok/s.
**Decision / Analysis:** ACCEPT both as opt-in. The full-model improvement matters more than the almost flat isolated screen; direct-reduction scratch and concurrency remain caveats.
**Next step:** Profile the full graph and target a larger kernel group.

### OPT-0012: Change decode attention split count

**Date / Commit:** 2026-10-06; exact commit not reconstructed. Relevant file: `plugins/gpt_oss/cuda/gpt_oss_kernels.cu`.
**System / Workload:** British Columbia Workstation TP2, batch 1 arithmetic, two runs per split count, other flags fixed.
**Baseline / Observed bottleneck:** Split attention is ~0.23 ms/token in the captured kernel sum; baseline 16 splits 193.84/192.86 tok/s.
**Hypothesis / Proposed optimization:** Eight or 32 splits might better balance parallelism and merge cost.
**Implementation:** Experimental split-count override.
**Correctness validation:** Six expected final answers passed; generated token counts differed across 8/16/32 (94/90/91).
**Performance result:** 8 splits 187.41/187.98; 32 splits 193.92/191.42 tok/s. No reliable improvement over 16.
**Decision / Analysis:** REJECT alternatives. Eight underutilized the GPU, while 32's extra merge work and variability erased any benefit.
**Next step:** Improve attention algorithm without changing split count alone.

### OPT-0013: Warp-local decode top-K

**Date / Commit:** 2026-10-06; candidate reverted, exact commit not reconstructed. Relevant file: `plugins/gpt_oss/cuda/gpt_oss_kernels.cu`.
**System / Workload:** British Columbia Workstation TP2, BF16/MXFP4, batch 1; four saved prompts, two trials each.
**Baseline / Observed bottleneck:** Existing top-K route used cross-warp reductions and repeated block barriers; route group ~0.19 ms/token in one trace.
**Hypothesis / Proposed optimization:** Keep each lane's four expert scores in registers through eight selections, avoiding some barriers.
**Implementation:** Opt-in single-warp candidate; later reverted.
**Correctness validation:** Full synthetic CUDA/TP2 suite including Qwen compatibility; all eight final answers passed.
**Performance result:** Arithmetic 192.95/193.57 vs 193.84/192.86; code 195.67/195.69 vs 195.86/194.78; instruction 194.92/195.46 vs 194.30/194.50; long 171.64/171.45 vs 170.22/170.69 tok/s. Raw: `D:/CantorAI/work/vast-54543362/garnet-inline-warp-topk/`.
**Decision / Analysis:** REJECT; differences were noise-level and insufficient to justify specialized complexity.
**Next step:** Fuse more substantial MoE work or improve Marlin itself.

### OPT-0014: Stage 32 KV positions per prefill tile

**Date / Commit:** 2026-10-06; candidate reverted, exact commit not reconstructed. Relevant file: `plugins/gpt_oss/cuda/gpt_oss_kernels.cu`.
**System / Workload:** British Columbia Workstation TP2, 2,005-token long prompt, two repeated prefill runs per setting.
**Baseline / Observed bottleneck:** Tiled attention still summed ~99 ms/36 layers on GPU 0; baseline staged 16 KV positions.
**Hypothesis / Proposed optimization:** Stage 32 positions to reduce per-query KV tile setup.
**Implementation:** Experimental 32-position shared-memory stage.
**Correctness validation:** Full synthetic TP2/Qwen suite passed; all six long final answers passed across related screens.
**Performance result:** 32-position 0.470/0.461 s vs 16-position 0.462/0.465 s. With fast exponentials, 32-position 0.453/0.455 vs 16-position 0.451/0.453 s.
**Decision / Analysis:** REJECT and revert. Larger stage did not improve the balance of occupancy, KV reuse and synchronization.
**Next step:** Redesign the full attention loop, not only its stage size.

### OPT-0015: cuBLAS FP32 router GEMM for prefill

**Date / Commit:** 2026-10-06; candidate reverted, exact commit not reconstructed. Relevant files: GPT-OSS router plugin/CUDA path.
**System / Workload:** British Columbia Workstation TP2, 2,005-token long prompt, two end-to-end trials, router change limited to prefill ≥128 tokens.
**Baseline / Observed bottleneck:** Router scores summed ~36.9 ms over 36 layers in the warm profile; tiled long prefill ~0.46 s.
**Hypothesis / Proposed optimization:** cuBLAS GEMM might use tensor cores more efficiently than router-score CTAs.
**Implementation:** Opt-in FP32 cuBLAS router scores for large prefill; decode unchanged.
**Correctness validation:** Both expected final answers passed. A separate Nsight CUDA profiler-range capture hit TensorRT enqueue failure.
**Performance result:** 0.520/0.510 s, slower than ~0.46 s. Raw: `D:/CantorAI/work/vast-54543362/garnet-router-sgemm-long/`.
**Decision / Analysis:** REJECT and revert. Launch/layout conversion and shape-specific GEMM behavior likely exceeded any arithmetic gain; the profiler failure adds fragility.
**Next step:** Benchmark a shape-specialized router in isolation before a full-model change.

### OPT-0016: BF16 attention-prefill TP collective

**Date / Commit:** 2026-10-06, `f9551e1`, `feature/gpt-oss-120b`. Relevant files: `plugins/gpt_oss/cuda/tp_collective.cpp`, GPT-OSS plugin and xModel.
**System / Workload:** British Columbia Workstation TP2, BF16 dense/MXFP4 MoE; batch 1, 2,005-token long prompt; three alternating warm BF16/FP32 pairs plus four saved full prompts twice.
**Baseline / Observed bottleneck:** GPU-0 warm prefill summed ~52.7 ms NCCL FP32 for 72 calls; MoE and decode collectives remained FP32.
**Hypothesis / Proposed optimization:** Attention projection is BF16-rounded before reduction, so BF16 communication can halve bytes for large attention-prefill messages.
**Implementation:** `GARNET_GPT_OSS_BF16_PREFILL_ALLREDUCE=1` for attention prefill ≥128 tokens; pack BF16, NCCL BF16 sum, unpack FP32 in per-invocation plugin workspace.
**Correctness validation:** Rank-paired FP32/BF16 collective test, full synthetic TP2/Qwen suite; all eight full answers and token sequences identical to FP32 baseline.
**Performance result:** Paired long prefill BF16 0.4586/0.4589/0.4582 vs FP32 0.4622/0.4628/0.4650 s; ~4.2 ms (0.9%) median gain. BF16 profile NCCL ~12.5 ms/36 attention calls plus FP32 ~29.1 ms/36 MoE calls, pack/unpack ~0.9 ms total; profiled app boundary 0.312 vs earlier FP32 0.325 s, with nonidentical capture conditions. Decode unchanged. Raw: `D:/CantorAI/work/vast-54543362/garnet-bf16-*/`.
**Decision / Analysis:** ACCEPT opt-in. Reduced attention communication, but conversion and unaffected groups made the end-to-end gain small. Broader numerical tests remain.
**Next step:** Attack larger attention, GEMM or MoE groups.

### OPT-0017: Fuse attention residual add with RMSNorm

**Date / Commit:** 2026-10-06, candidate `f8a9c7f` and dispatch fix `76ad77a`, reverted by `dca0103` and `aea84c8`; branch `feature/gpt-oss-120b`. Relevant files: GPT-OSS xModel, plugin/CUDA, TensorRT lowering.
**System / Workload:** British Columbia Workstation TP2, BF16 dense/MXFP4 MoE, batch 1; four saved prompts twice, same other optimized flags as unfused baseline. BF16 prefill collective off to isolate fusion.
**Baseline / Observed bottleneck:** ~873 GPU kernels/token; residual add and RMSNorm are separate graph operations in each layer. Short decode ~193–196 tok/s, long ~170–171.
**Hypothesis / Proposed optimization:** Fuse rounded residual add and RMSNorm into one GPT-OSS native op while preserving both the rounded residual and normalized output, reducing graph nodes/intermediate traffic.
**Implementation:** Experimental two-output CUDA/TensorRT plugin and xModel lowering, opt-in during the test.
**Correctness validation:** Fused CUDA outputs exactly matched separate GPU operations; full synthetic TP2 and Qwen checks passed. Eight pretrained token sequences matched unfused baseline exactly.
**Performance result:** Arithmetic 194.80/192.70 vs 193.84/192.86; code 195.05/194.81 vs 195.86/194.78; instruction 195.14/194.59 vs 194.30/194.50; long 172.09/171.08 vs 170.22/170.69 tok/s. Short prefill ~0.21 s; long 0.475/0.464 s vs ~0.46 s. Raw: `D:/CantorAI/work/vast-54543362/garnet-fused-addnorm-all-four/`.
**Decision / Analysis:** REJECT and revert. One small fusion per layer yielded no repeatable full-model improvement; larger kernel groups and dependencies dominate.
**Next step:** Examine projection tactics and layer-level graph work.

### OPT-0018: Raise TensorRT builder optimization level to 5

**Date / Commit:** 2026-10-06; configuration-only, no source commit; tested at branch `293b4ae`. Relevant file: `tools/gpt_oss/pipeline.py`.
**System / Workload:** British Columbia Workstation TP2, BF16 dense/MXFP4 MoE, batch 1; four saved prompts twice, greedy sampling, same other optimized flags as level-1 baseline, `GARNET_GPT_OSS_TRT_OPT_LEVEL=5`. Fresh prefill/decode engines compiled for each prompt shape.
**Baseline / Observed bottleneck:** TensorRT BF16 projection GEMMs summed ~1.07–1.19 ms/token in warmed GPU-0 traces; level-1 arithmetic 193.84/192.86 tok/s.
**Hypothesis / Proposed optimization:** Wider TensorRT tactic search may find a better projection implementation on Blackwell.
**Implementation:** Builder level 5 via existing environment setting; no model or kernel source change.
**Correctness validation:** Eight decoded final answers passed. Code-tracing trial 0 had identical token IDs to level 1; arithmetic, instruction and long prompts changed token sequences/lengths (87 vs 90, 147 vs 146, 126 vs 109 respectively). Exact numerical equivalence is therefore not established.
**Performance result:** Arithmetic 197.74/197.25 vs 193.84/192.86; code 198.12/198.79 vs 195.86/194.78; instruction 198.14/198.65 vs 194.30/194.50; long 174.83/174.11 vs 170.22/170.69 tok/s. Short prefill ~0.209–0.218 s; long 0.473/0.462 s vs ~0.462/0.465 s. Fresh engine generation took several minutes per new shape. TTFT, memory delta, and tactic-level causal attribution not measured. Raw: `D:/CantorAI/work/vast-54543362/garnet-trt-level5-all-four/`.
**Decision / Analysis:** INCONCLUSIVE for default use; the setting is available for further paired tests. The ~2% rate improvement is too small to meet the vLLM goal, and changed generation trajectories require broader quality checks. This differs from OPT-0006's level-3 regression on another host.
**Next step:** Profile level-5 GEMM tactics against level 1 and target a larger per-layer improvement; preserve the full-model answer checks.

#### OPT-0018 follow-up profile, 2026-10-06/07

A matched 32-token arithmetic Nsight Systems capture on the same Workstation held all non-builder flags fixed. On GPU 0, level 1 recorded **27,936 kernels**, **120.36 ms** summed kernel time and a **188.31 ms** captured span; level 5 recorded **24,448 kernels**, **118.36 ms** summed time and a **183.44 ms** span. That is 3,488 fewer kernels, or 109 fewer per token, while summed GPU work fell only ~0.063 ms/token. The principal TensorRT BF16 GEMM group was **34.26 ms** at level 1 versus **34.38 ms** at level 5 over the window; level 5 selected a fused-named GEMM tactic and removed a separate ~1.56 ms split-K group plus some pointwise nodes. This suggests the end-to-end improvement mainly comes from graph-node/adjacent-operation fusion, not faster core GEMM math. Nsight profiling changes timing, so these captured spans are explanatory evidence, not a replacement for the unprofiled paired rates above. Raw trace and SQLite export: `D:/CantorAI/work/vast-54543362/profiles/garnet-trt-level5-arithmetic.*`. The changed generated token sequences and small remaining speedup leave the decision INCONCLUSIVE.

### OPT-0019: Specialized BF16 decode projection GEMV

**Date / Commit:** 2026-10-06/07, isolated screen `e99de89`, integrated candidate not yet measured, `feature/gpt-oss-120b`. Relevant files: `tools/gpt_oss/gemv_screen.cu`, `plugins/gpt_oss/cuda/gpt_oss_gemv.cu`, GPT-OSS TensorRT plugin/lowering.
**System / Workload:** British Columbia Workstation, 2× RTX PRO 6000 Blackwell, BF16 dense/MXFP4 MoE, TP2. Isolated screen uses 36 pairs of QKV (2,560×2,880) and attention output (2,880×2,048) BF16 projections, 911.2 MiB distinct weights, batch 1; 20 CUDA-graph warmups and five sets of 100 replays. Full-model four-prompt result pending.
**Baseline / Observed bottleneck:** In a 32-token Garnet arithmetic trace, TensorRT BF16 projection GEMMs summed 34.26 ms on GPU 0 (~1.07 ms/token); level 5 removed adjacent graph work but left the main GEMM group at 34.38 ms. This is profiled time and cannot be directly equated with unprofiled microbenchmarks.
**Hypothesis / Proposed optimization:** A batch-1 matrix-vector kernel can stream BF16 weight rows, accumulate in FP32 and round to BF16 without TensorRT's general matrix-multiply tactic. Before: TRT matrix multiplication of 1×K by K×M. After: one warp per output row, four rows per block in a GPT-OSS-only native operator. Preserve existing FP32 cast, bias add, and following model-level BF16 rounding.
**Implementation:** The isolated graph screen was written locally, committed/pushed, pulled and compiled on Vast. A four-row CUDA kernel and opt-in TensorRT plugin/lowering path are under local development; only GPT-OSS TP2 attention QKV/out decode projections are eligible. Prefill, MoE and Qwen are unchanged.
**Correctness validation:** On the isolated screen, max absolute error against FP32 CPU reference over 16 sampled rows was 0.004859 and max relative error 0.002896 for four- and eight-row kernels. Integrated synthetic parity, full pretrained outputs and exact token comparisons are pending.
**Performance result:** On Blackwell, graph replay of 72 distinct projections: four rows/block **0.7266 ms/decode**, eight rows/block **0.7536 ms/decode**. On local RTX 4080, 1.6594 and 1.6541 ms respectively. These exclude the full model and are not a measured speedup over TensorRT under identical instrumentation. TTFT, ITL, GPU memory delta and end-to-end tok/s pending.
**Decision / Analysis:** INCONCLUSIVE. The screen is promising enough to test in the model, but no claim of serving gain is justified yet.
**Next step:** Complete the opt-in plugin, run CUDA/TP2 parity and all four saved requests sequentially, compare token outputs and warm rates to level-1 TensorRT on the same machine; reject/revert if it does not produce a meaningful end-to-end gain.

#### OPT-0019 first integration and trace, 2026-10-07

The first opt-in TensorRT integration used rank-sharded weight tensors produced by TensorRT slices/concatenation. The full synthetic suite, Qwen compatibility and direct GEMV parity passed. All eight pretrained final answers passed, but warm wall decode regressed: arithmetic **189.02/186.38** vs baseline **193.84/192.86**, code **188.78/189.74** vs **195.86/194.78**, instruction **188.07/188.51** vs **194.30/194.50**, and long **168.99/166.31** vs **170.22/170.69 tok/s**. Arithmetic, code and long token sequences matched the baseline; instruction changed from 146 to 149 tokens while preserving the expected final answer. Long prefill **0.476/0.463 s** showed no gain. Raw JSON, logs and decoded answers: `D:/CantorAI/work/vast-54543362/garnet-decode-gemv-all-four/`.

A matched 32-token arithmetic Nsight capture explained the regression. GPU 0's first GEMV graph had **29,088 kernels**, **128.07 ms** summed kernel time and a **194.04 ms** span, compared with **27,936**, **120.36 ms** and **188.31 ms** for the prior TensorRT graph. GEMV kernels themselves summed **9.48 ms** over 2,304 calls, but two TensorRT `Move` groups materializing the sharded QKV and output weights summed **11.49 + 9.43 ms** over 1,152 calls each. These extra copies were absent from the baseline and outweighed the arithmetic savings. The full-model result REJECTS this first implementation, not the broader direct-weight hypothesis. Raw trace: `D:/CantorAI/work/vast-54543362/profiles/garnet-decode-gemv-arithmetic.*`.

The revised candidate leaves the checkpoint's full refittable weight bound to the plugin and computes rank-specific QKV row selection or output-projection column offsets inside GEMV. It avoids per-token TensorRT weight-slice materialization without duplicating the checkpoint. Dedicated parity cases now cover both TP ranks and boundary rows. It has not yet been built or benchmarked; no revised speedup is claimed.

#### OPT-0019 revised full-weight integration, 2026-10-07

The revised candidate was committed as `0db099c`, built on the same Workstation, and passed direct CUDA parity (including QKV rank boundaries and output-projection column offsets) plus the full synthetic TP2 and Qwen-compatibility suite. With the same optimized flags, all eight pretrained final answers passed. Arithmetic and code tracing token sequences were identical to the saved level-1 baseline, as were both long-context sequences; instruction following changed to 149 rather than 146 tokens while still giving the expected answer. The eight warm wall rates were arithmetic **194.71/195.03**, code **197.28/197.35**, instruction **197.64/197.41**, and long **174.07/172.99 tok/s**, versus baseline **193.84/192.86**, **195.86/194.78**, **194.30/194.50**, and **170.22/170.69** respectively. Short prefill was ~0.21–0.22 s except one 0.495 s outlier; long prefill was **0.478/0.465 s**. Raw outputs and logs: `D:/CantorAI/work/vast-54543362/garnet-decode-gemv-v2-all-four/`.

The revised kernel is directionally better than the first weight-sliced integration and yields only ~1–3% full-model decode gains over the saved baseline. This is insufficient against the ~273 tok/s optimized vLLM reference; no target achievement is claimed. Keep the path opt-in pending a matched second profiler capture to confirm removal of per-token weight moves and broader numerical validation for the instruction token divergence. Next, measure the user's required aggregate batch output throughput across batch, input length, output length and BF16 KV-cache use; a batch-1 GEMV improvement cannot stand in for that serving metric.

### OPT-0020: Matched aggregate batch output-throughput profile

**Date / Commit:** 2026-10-07, batch runners `3c2d9af`, memory/length protocol `195c8d0`, fixed-context profile `5fb22a2`, single-slot controls `f604625`, `feature/gpt-oss-120b`. Relevant files: `tools/gpt_oss/run_vllm_batch_throughput.py`, `tools/gpt_oss/run_tp_batch_throughput.py`, `python/garnet_pipeline.py`.
**System / Workload:** British Columbia Workstation TP2, BF16 KV, greedy fixed-output generation with early EOS disabled, no prefix cache on vLLM, no simultaneous GPU processes. The initial sweep uses the saved 256-token arithmetic prompt with 128 output tokens at batches 1/2/4/8, then 512 outputs at batch 8 and the saved 2,005-token prompt with 128 outputs at batch 4. The Garnet comparison fixes its maximum KV context to 4,096 tokens per request, as on vLLM. Each duplicate batch slot receives a disjoint page-table range.
**Baseline / Observed bottleneck:** The earlier four-prompt benchmark measured only batch 1, while the user's primary serving metric is maximum aggregate output tokens per second. Neither single-request latency nor a microkernel screen establishes that metric.
**Hypothesis / Proposed optimization:** Establish a matched batch/input/output/KV-cache profile, then target the stage where Garnet falls behind vLLM. Before: no measured multi-request TP2 throughput. After: separate decode aggregate, full-request throughput, per-request completion, GPU memory and KV allocation, with raw token sequences retained for batch-isolation checks.
**Implementation:** Added `greedy_batch` sampling to the Python TP2 wrapper without changing its default single-request path, plus exclusive-GPU benchmark scripts. The first Garnet smoke run used request-sized KV and host-updated controls; it is a functional diagnostic, not part of the fixed-context comparison.
**Correctness validation:** The batch-1 Garnet smoke run generated all requested tokens. Cross-slot identity and answer checks are pending multi-slot Garnet results.
**Performance result:** Optimized vLLM 0.31.0 yielded **272.82/411.49/727.56/1225.93 aggregate decode tok/s** for batches 1/2/4/8 on the 256-token input with 128 outputs. Batch 8 with 512 outputs gave **1139.65 tok/s**; the 2,005-token input at batch 4 with 128 outputs gave **511.91 tok/s**. Its full-request output rates were **258.64/396.10/692.90/1192.11**, then **1128.90** and **469.80 tok/s**, respectively. vLLM used ~79.4–79.7 GiB GPU memory and reported ~40.08 GiB available KV-cache memory per GPU (about 1.165 million cached tokens at the configured profile). Raw JSON/logs: `D:/CantorAI/work/vast-54543362/vllm-batch-throughput/`.

The fixed-4,096-context Garnet batch-1 run gave **145.21 aggregate tok/s including the first decode step**, **178.71 tok/s after that one-time step**, and **116.92 full-request output tok/s**. It allocated **150,994,944 bytes (144 MiB) KV per GPU** and used **63.3 GiB GPU memory** after loading decode engines. The earlier request-sized KV smoke run gave 140.21/170.83 tok/s with slower host-updated controls. Both lag vLLM substantially; decode engine warmup and API timing boundaries differ, so retain the raw step timings. Batch 2 and higher Garnet runs are pending. Raw fixed batch-1 result: `D:/CantorAI/work/vast-54543362/garnet-batch-throughput/arithmetic-fixed-b1-o128.json`.
**Decision / Analysis:** MEASUREMENT IN PROGRESS. The b1 result establishes that memory headroom alone does not imply throughput; profile batched decode after multi-slot correctness and scaling are measured.
**Next step:** Complete Garnet batch 2/4/8, then matched 512-output and long-input cases; compare token sequences across duplicate slots, profile GPU/host bottlenecks, and optimize before claiming any serving gain.

#### OPT-0020 batch-2 follow-up, 2026-10-07

At fixed 4,096-token KV capacity, Garnet batch 2 allocated **288 MiB KV per GPU**, used about **63.5 GiB GPU memory**, and delivered **242.66 aggregate decode tok/s including the first decode step**, **287.60 tok/s after that one-time step**, and **198.05 full-request output tok/s**. The matching vLLM batch-2 decode/full-request rates were **411.49/396.10 tok/s**. Both Garnet duplicate slots generated 128 tokens, but their token IDs first diverged at output index 34; vLLM's duplicate slots also diverged (first at index 17), so lane identity alone is not a correctness criterion. A first-final-answer check per slot is being added. Raw Garnet result: `D:/CantorAI/work/vast-54543362/garnet-batch-throughput/arithmetic-fixed-b2-o128.json`. The scaling from b1 to b2 is real, but the vLLM gap remains large; batch 4/8 are the next measured points.

#### OPT-0020 batch-4/8 and length follow-up, 2026-10-07

Garnet's fixed-context arithmetic batch 4 gave **483.92 aggregate decode tok/s including the first decode step**, **572.87 after that one-time step**, **382.17 full-request tok/s**, **576 MiB KV per GPU**, and ~**63.8 GiB GPU memory**. Batch 8 gave **812.12 / 932.51 / 624.32 tok/s** by those three measures, **1,152 MiB KV per GPU**, and ~**64.4 GiB GPU memory**. Matching vLLM aggregate decode was **727.56** at batch 4 and **1225.93** at batch 8; full-request rates were **692.90** and **1192.11**. All four and eight Garnet slots respectively passed their first-final JSON check. Garnet batch 8 with 512 fixed output tokens gave **918.51 / 953.13 tok/s** (first-step-inclusive / later-only), versus vLLM's **1139.65** first-step-inclusive; all eight first-final answers passed. Raw files are under `D:/CantorAI/work/vast-54543362/garnet-batch-throughput/`.

The 2,005-token input at batch 4 initially failed TensorRT compilation: `gpt_oss_moe_mxfp4_65` required **759,916,596 bytes** scratch but the builder allowed **268,435,456 bytes**. `GARNET_GPT_OSS_TRT_WORKSPACE_MB=1024` let it build. The user-visible answer passed in all four slots; prefill was **3.033 s**, aggregate decode **459.68 tok/s including first step / 540.42 later-only**, and full-request output **123.74 tok/s**. Matching vLLM was **511.91 aggregate decode / 469.80 full-request tok/s**. Thus Garnet's later-only decode can exceed vLLM on this shape, but prefill dominates its full-request deficit. This was a cold prefill execution; a repeat and matched warmed-decode run are pending. The successful raw file is `long-fixed-b4-o128-workspace1024.json`; the failed log is `long-fixed-b4-o128.log` in the same work directory.

After observing that exact failure, commit `8bfd8d7` added an automatic builder-workspace estimate mirroring `GptOssMoeWorkspace()`; a local check returned **759,916,596 bytes** for the failing 8,020-row shape and selects a **1,024 MiB** builder cap. It also warms one Garnet decode step outside the timed request, matching vLLM's prior warmup batch, then overwrites that same KV slot in the measured request. The pre-change results above preserve first-step-inclusive and later-only rates rather than silently discarding cold cost. The auto-fit and warmed comparison are being verified on Vast; they do not imply a speed gain by themselves.

#### OPT-0020 warmed matched results, 2026-10-07

After the one-step Garnet warmup, identical 256-token input and 128-token fixed output gave aggregate decode **176.74, 302.35, 568.78, 989.30 tok/s** at batches 1, 2, 4 and 8, versus optimized vLLM **272.82, 411.49, 727.56, 1225.93**. Garnet full-request output rates were **137.4, 238.0, 435.9, 729.2 tok/s**, versus vLLM **258.6, 396.1, 692.9, 1192.1**. The Garnet rate scales with batch but has not caught vLLM. All Garnet batch slots passed the first-final expected-answer check after warmup; duplicate token sequences still need not match one another. These results use the same 4,096-token maximum KV context and BF16 KV type. Garnet's KV allocation was **144/288/576/1152 MiB per GPU**, with GPU-0 resident memory around **63.4/63.5/63.8/64.4 GiB**. The vLLM allocation was substantially larger, about 79.5 GiB resident per GPU, with ~40 GiB available KV space; the allocation policy differs but the dtype and context limit match.

With 512 fixed output tokens at batch 8, warmed Garnet gave **949.22 aggregate decode / 874.6 full-request tok/s**, versus vLLM **1139.65 / 1128.9**; all eight first-final answers passed. With the 2,005-token input at batch 4, Garnet's automatic builder workspace selected **1,024 MiB**, compiled or loaded successfully, and yielded **3.030 s prefill**, **574.37 aggregate decode / 130.8 full-request tok/s**, versus vLLM **511.91 / 469.8**. All four first-final answers passed. This is a shape where Garnet now exceeds vLLM's aggregate *decode-only* rate by ~12% but loses the full request by ~72%, principally due to prefill. The next optimization priority is batched long-input prefill (including the grouped MoE fallback above the Marlin row limit), with batch-8 decode profiling in parallel as a smaller-gap target. Raw JSON, validation and logs are in `D:/CantorAI/work/vast-54543362/garnet-batch-throughput/` and `vllm-batch-throughput/`.

### OPT-0021: Permit 8,020-row Marlin prefill

**Date / Commit:** 2026-10-07, candidate `3b8227d` on `feature/gpt-oss-120b`; reverted after pretrained failure. Relevant file: `plugins/gpt_oss/cuda/gpt_oss_marlin.cu`.
**System / Workload:** British Columbia Workstation TP2, long 2,005-token input repeated four ways (8,020 rows), 128 or 256 fixed generated tokens, 4,096-token maximum context. Same built plugin and otherwise identical optimized flags; `GARNET_GPT_OSS_MARLIN_MAX_TOKENS=4096` baseline versus `8192` candidate, sequential A/B.
**Baseline / Observed bottleneck:** Marlin rejected 8,020 rows at its 4,096-row opt-in cap, falling back to grouped MoE and producing ~3.03 s prefill. The vLLM full batch request is ~1.09 s.
**Hypothesis / Proposed optimization:** Marlin's packed MXFP4 expert kernel might process the larger row count faster. The candidate only expands the opt-in supported cutoff; no model graph or decode kernel code was changed.
**Implementation:** One-line 8,192-row opt-in allowance. CUDA plugin rebuilt on Vast from the locally pushed commit; the full synthetic suite passed (kernel parity, TP graph, multi-GPU, Qwen compatibility), but those tests do not exercise the 8,020-row pretrained shape.
**Correctness validation:** Debug logs confirm baseline `shape rejected (8020, 2880)` and candidate `decode candidate (8020, 2880)`. All four baseline slots passed expected final JSON. At 128 output tokens, only candidate slots 2/3 passed, while 0/1 had no final JSON. Extending to 256 tokens did not recover them: decoded text showed unrelated arithmetic-task analysis and a wrong long-prompt answer with units `[5, 32, 70]` instead of `[15, 42, 70]`. Candidate token sequences diverged from baseline as early as output index 5 in slots 0–2. This is a genuine correctness failure, not merely an insufficient output cap.
**Performance result:** Baseline prefill **3.031 s** versus candidate **1.369 s** (~55% reduction). Baseline warmed aggregate decode **538.80 tok/s** versus candidate **449.65** in the first A/B; the cutoff should not alter decode dispatch, so this latter difference may include unrelated run variability or state effects. Full-request output throughput improved **128.83 → 204.89 tok/s** but remained below vLLM **469.8** and cannot be accepted with wrong answers. Raw JSON, validation files, decoded slot text and logs: `D:/CantorAI/work/vast-54543362/garnet-batch-throughput/long-marlin-{4096,8192}-b4-o128.*` and `long-marlin-8192-b4-o256.*`.
**Decision / Analysis:** REJECT and revert the expanded cutoff. The 4,096-row path remains in force. The exact numerical or indexing defect inside the Marlin large-row path is not yet isolated; merely raising its supported limit is unsafe.
**Next step:** Optimize batched long prefill through chunking or a corrected large-row kernel with direct parity at 8,020 rows before a repeat full-model test. Keep vLLM/Garnet batch-16/32 throughput and KV-memory profiling separate from this rejected candidate.

#### OPT-0020 batch-16/32 and KV-capacity follow-up, 2026-10-07

At the same 4,096-token maximum context per request, 256 input tokens and 128 fixed output tokens, optimized vLLM aggregate decode throughput was **1,989.87 / 2,832.10 tok/s** at batch 16/32; full-request output throughput was **1,942 / 2,785 tok/s**. Garnet delivered **1,448.21 / 2,606.43 decode tok/s** and **1,022 / 967.7 full-request tok/s**. All slots passed their first-final expected-answer check. Garnet allocated **2,304 / 4,608 MiB BF16 KV per GPU** and used about **65.5 / 67.9 GiB** GPU memory at decode; vLLM resided near **79.7 GiB**. The batch-32 decode gap is only about 8%, but Garnet's prefill expanded to **2.673 s** and erased that scaling in the complete request. Raw per-slot tokens, logs and validation are under `D:/CantorAI/work/vast-54543362/{garnet,vllm}-batch-throughput/`.

For subsequent serving profiles, record **batch size, actual input length, fixed output length, maximum context/KV reservation, allocated KV bytes, resident and peak GPU memory, prefill/first-token latency, warm aggregate decode throughput, and full-request throughput** together. A shorter maximum context can admit a larger batch, while a longer input increases prefill work and a longer output weights decode more heavily. Report the maximum throughput only with the shape and memory budget that produced it. Keep BF16 KV and matched context/output settings across Garnet and vLLM. The next matched sweep uses a 512-token per-request context for the 256+128-token arithmetic workload, beginning with batch 64, with the engines run sequentially.

### OPT-0022: Chunked long-input batched prefill

**Date / Commit:** 2026-10-07, opt-in runner `2b059a4`, `feature/gpt-oss-120b`. Relevant file: `tools/gpt_oss/run_tp_batch_throughput.py`.
**System / Workload:** British Columbia Workstation TP2, four identical 2,005-token long prompts, 128 fixed output tokens, 4,096-token BF16 KV capacity per slot; otherwise the same optimized flags as the unchunked baseline.
**Baseline / Observed bottleneck:** One 8,020-row prefill invokes grouped MoE fallback and takes **3.030 s**, limiting full-request output throughput to **130.8 tok/s** despite **574.37 aggregate decode tok/s**. The rejected 8,192-row Marlin allowance was faster but failed pretrained correctness.
**Hypothesis / Proposed optimization:** Divide prefill into model calls small enough to keep each Marlin expert operation below the validated 4,096-row cutoff. Preserve the same KV tensors, page table, absolute positions and accumulated context length across chunks.
**Implementation:** Opt-in `GARNET_BATCH_PREFILL_CHUNK=512` uses chunks of **512+512+512+469 tokens**; their respective measured prefill steps were **0.3913, 0.2546, 0.2896 and 0.4959 s**. The last chunk requires a separately shaped TensorRT engine. No GPT-OSS operator or Qwen implementation changed for this candidate.
**Correctness validation:** All four slots passed the first-final expected-answer check. Full output token sequences match the unchunked baseline in slot 2, and first diverge at output indexes 5, 22 and 22 in the other slots. The differing tokens warrant further numerical and quality checks; a passing final JSON alone is not exact parity.
**Performance result:** Prefill **3.030 → 1.431 s** and full-request output **130.8 → 202.2 tok/s**. Aggregate decode regressed **574.37 → 461.29 tok/s**, so the full-request improvement is smaller than the prefill reduction. Both runs used about **93.3 GiB GPU memory during decode**; the memory near device capacity is not unique to chunking. Chunked execution also rebuilt two prefill engine shapes, adding substantial setup time excluded from the steady-request metric. Raw JSON, log and validation: `D:/CantorAI/work/vast-54543362/garnet-batch-throughput/long-chunk512-b4-o128.*`.
**Decision / Analysis:** KEEP EXPERIMENTAL. This improves the complete long batch request but remains well below vLLM's **469.8 tok/s** on the matched shape. Repeat with cached engines, check numerical/answer stability and profile the decode regression before treating it as a serving optimization. Test whether a chunk size that divides the input more evenly reduces the expensive final 469-token step and memory pressure.
**Next step:** Run sequential matched batch/context/KV sweeps, then profile batch-8 decode and long-batch prefill to identify the expensive layer groups. Maintain output-quality validation for every candidate.

#### OPT-0020 512-context batch-64 follow-up, 2026-10-07

To test a higher-throughput point without reserving 4,096 KV tokens for each slot, both engines used a **512-token maximum context**, **256-token actual input**, **128 fixed output tokens**, **batch 64**, TP2 and BF16 KV. The vLLM run completed first, then Garnet ran without a competing GPU process. All 64 slots passed the first-final answer check in both runs. vLLM delivered **3,579.54 aggregate decode / 3,518.16 full-request output tok/s** and used about **79.7 GiB** resident memory per GPU. Garnet delivered **3,064.23 decode / 1,865.23 full-request tok/s**, with **1.739 s** prefill, **1,207,959,552 bytes (1,152 MiB) allocated KV per GPU**, about **64.4 GiB** resident after engine loading and about **91.7 GiB** sampled during decode. The decode gap is ~14%; the full-request gap is ~47% because of prefill. This is the fastest Garnet aggregate decode result measured so far, not evidence that it has beaten vLLM on maximum throughput.

An initial unchunked Garnet batch-64 attempt was rejected by the memory planner before engine loading; it budgeted a 16,384-row prefill. Commit `17c9e9e` makes the opt-in chunked runner pass its largest actual prefill chunk to that planner. Four 64-token chunks, each 4,096 rows, then fit the stated 90% planning budget without changing operator code or increasing the budget. This is an important input-length/KV distinction: a 512-token context reduces KV reservation, while prefill row count and its activation workspace depend on **batch × current input chunk**. Results and validation files: `D:/CantorAI/work/vast-54543362/{vllm,garnet}-batch-throughput/arithmetic-ctx512-b64-o128*`.

#### OPT-0020 batch-8 decode kernel trace, 2026-10-07

Nsight Systems captured **16 warmed decode steps** of Garnet's arithmetic **batch 8**, with **256 input / 128 requested output tokens** and **4,096-token BF16 KV capacity per slot**. The capture was bounded by CUDA profiler start/stop calls after warmup. The report and SQLite export were copied to `D:/CantorAI/work/vast-54543362/profiles/garnet-batch8-arithmetic.*`. The profiled process was interrupted after the report was saved because Nsight made the remainder of the fixed-output run unusually slow; this trace is for kernel attribution, while the separate unprofiled batch-8 run supplies throughput and correctness.

The combined two-GPU CUDA kernel-duration sums in the 16-step report were **NCCL all-reduce 46.32 ms (25.1%)**, TensorRT BF16 GEMMs **34.68 ms (18.8%)**, the two Marlin expert kernel variants **46.67 ms (25.3%)**, and split attention partial **15.19 ms (8.2%)**. There were **2,304 NCCL all-reduce kernels** (two per 36 layers per step per GPU). The enabled direct-reduction flag applies only to the 2,880-element batch-1 message; at batch 8 the 23,040-element messages use NCCL. Approximate per-GPU per-step sums are 1.45 ms NCCL, 1.08 ms BF16 GEMM, 1.46 ms Marlin and 0.47 ms attention, before other kernels and graph scheduling. These are GPU-duration sums, not additive shares of unprofiled wall latency. The next batch-decode candidate should investigate a correct, concurrency-safe reduction for batched messages and compare end-to-end output quality and throughput; changing the existing batch-1 peer scratch size alone would be unsafe.

### OPT-0023: Screen direct TP reduction for serialized batches

**Date / Commit:** 2026-10-07, candidate under development on `feature/gpt-oss-120b`. Relevant files: `plugins/gpt_oss/cuda/tp_direct.cu`, `tp_collective.cpp`, `test2026/gpt_oss/tp_direct_benchmark.cu`.
**System / Workload:** Target Workstation TP2, FP32 hidden states of 2,880 elements per request, batches up to 64; paired rank streams and no overlapping independent model executions.
**Baseline / Observed bottleneck:** The batch-8 trace attributes 25.1% of combined kernel time to NCCL all-reduce; the existing peer reduction only handles batch 1.
**Hypothesis / Proposed optimization:** Vectorized peer reads and a two-rank system-scope handshake may reduce batched message latency. Screen CTA count and changing-input CUDA graph replay before pretrained evaluation.
**Implementation:** A separate opt-in `GARNET_GPT_OSS_DIRECT_BATCH_ALLREDUCE=1`, together with the existing direct flag, permits hidden-state messages up to batch 64. Per-CTA start/end/epoch storage and loop bounds cover the complete message; `GARNET_GPT_OSS_DIRECT_BATCH_CTAS` screens 2/4/8/16/32 CTAs while batch-1 behavior stays at two. The benchmark verifies every element on both ranks, changing inputs between graph operations and trials, then times 72 reductions per graph. Local CUDA 12.0 compilation passed.
**Correctness validation / Performance result:** Target compilation, eager/changing-input graph parity and matched NCCL timing pending. No speedup is claimed.
**Decision / Analysis:** EXPERIMENTAL SCREEN ONLY. Communicator-wide staging requires serialized execution; this does not establish concurrent-serving safety. Do not make it a default or claim general serving support without per-execution context isolation.
**Next step:** Build and screen on Vast, reject slow or incorrect CTA variants, then evaluate any viable variant on the saved pretrained batch workloads sequentially.

#### OPT-0023 target screen and regression validation, 2026-10-07

Commit **`8b96f47`** compiled for SM120 with CUDA13.4 on the same British Columbia Workstation. The C++ benchmark captures **72 reductions per graph**, warms ten graph replays, then times **five trials of 50 replays**. Separate correctness graphs add exactly representable FP32 values to each rank's input before every reduction; five trials of three replays verify every output element on both GPUs with **zero absolute/relative error** against the exact host sum, covering 1,080 changing-input operations per configuration. The eager check and final outputs of every timed trial also match exactly. No overlapping independent executions were tested.

| Batch / FP32 elements | NCCL median µs/op | Direct 2 CTAs | Direct 4 CTAs | Direct 8 CTAs | Direct 16 CTAs | Direct 32 CTAs |
|---|---:|---:|---:|---:|---:|---:|
| 8 / 23,040 | 9.192 | 10.116 | 7.558 | 7.419 | 6.913 | 6.970 |
| 32 / 92,160 | 22.637 | 27.441 | 16.347 | 14.110 | 13.846 | 13.852 |
| 64 / 184,320 | 32.657 | 50.037 | 27.230 | 23.719 | 23.166 | 23.254 |

The earlier batch-8 NCCL trial was **8.302 µs/op**, so the exact size of that small-message gain varies with the run; retain both logs. NCCL batch1 was **6.218 µs/op**. Larger messages need more parallel peer reads: merely enlarging the two-CTA batch1 kernel regresses badly, whereas 16 CTAs is the best screened balance here. This is a kernel/graph screen and includes staging-copy and host graph launch costs; no full-model gain is established yet. Raw trial logs: `D:/CantorAI/work/vast-54543362/tp-direct-batch-screen/`. Reproduce with `GARNET_GPT_OSS_DIRECT_ALLREDUCE=1 GARNET_GPT_OSS_DIRECT_BATCH_ALLREDUCE=1 GARNET_GPT_OSS_DIRECT_BATCH_CTAS=16 out/build/gpt-oss/bin/garnet_gpt_oss_tp_direct_benchmark <batch> 50`; set the existing direct flag to zero for NCCL.

The complete synthetic verification suite passed with both direct flags and 16 CTAs: CUDA kernel parity, batched scalar updates, cold/warm compiled graph parity, TP collectives and MoE graph, two-GPU generation, native module bridge/API, and Qwen compatibility. Logs remain at `/workspace/CantorAI/work/tp-direct-batch-verification/`. The pretrained arithmetic **batch32/input256/output128/context4096** test is now running with the same saved optimized flags plus the new batch gate. Its baseline is Garnet **2,606.43 aggregate decode / 967.7 full-request tok/s** versus vLLM **2,832.10 / 2,785**. Decision remains **INCONCLUSIVE** until pretrained answer checks and matched model measurements complete.

### OPT-0024: Inline batched decode-control updates

**Date / Commit:** 2026-10-07, candidate under development, `feature/gpt-oss-120b`. Relevant files: `src/cuda/int_scalar_update.*`, `src/entry/garnet.*`, `python/garnet_pipeline.py`, `tools/gpt_oss/run_tp_batch_throughput.py`.
**System / Workload:** British Columbia Workstation TP2, batched GPT-OSS-120B decode; first target batch32/input256/output128/context4096, BF16 KV, FP32 activations/MXFP4 experts.
**Baseline / Observed bottleneck:** Code inspection shows the batched runner calls `tensor_update_from_host` separately for token, position, length and slot on each rank, synchronizing after every copy. The batch1 inline update cannot handle vectors. This creates eight host waits before the rank graphs on each step; the earlier single-request trace and OPT-0010 established a small end-to-end cost from this pattern.
**Hypothesis / Proposed Optimization:** Pass up to four integer vectors as kernel parameters and write them with one CUDA kernel on each rank's execution thread before its graph. Eliminate temporary host copies and their synchronizations while preserving per-slot values for future heterogeneous lengths.
**Implementation:** The generic backbone API `tensor_update_int_vectors_async(tensors, values)` accepts 1–4 dense CUDA INT32/INT64 tensors, each with 1–64 elements and a matching value list. It validates the complete group before writing; the by-value kernel parameter fits the local CUDA12.0 parameter limit. `forward_rank_local(..., vector_values=...)` schedules this on the same rank thread as the next graph, following the existing tensor lease/event mechanism. The benchmark enables it through `GARNET_GPT_OSS_INLINE_BATCH_CONTROLS=1`; larger batches keep the existing path. Qwen code is unchanged.
**Correctness Validation:** Added both-device cases for batch1/8/64, mixed integer widths, distinct token/position/length/slot values, INT32 overflow, wrong group/list sizes, oversized vectors, and no partial mutation after rejection. Local CUDA12.0 compilation and Python syntax checks passed; target API/parity and pretrained checks pending.
**Performance Result:** Not measured. Test this independently after the reduction-only run, retaining identical input/output/KV settings and raw per-step timings.
**Decision / Analysis:** INCONCLUSIVE, opt-in. This is a synchronization hypothesis, not evidence of a full-model improvement.
**Next Step:** Build remotely from the locally committed source, run API and model parity, then compare controls alone and controls plus direct batch reduction to the saved baseline.

#### OPT-0023 first pretrained batch-32 result, 2026-10-07

The reduction-only run at **`8b96f47`**, batch32/input256/output128/context4096, produced **2,756.94 aggregate decode tok/s** versus **2,606.43** in the saved NCCL baseline (~5.8% gain). Later-step diagnostic throughput was **2,799.71 tok/s**; retain the first-step-inclusive rate as the matched comparison. Prefill was **2.67277 s**, essentially unchanged, and execution-only complete-request throughput was **987.73** versus **967.71 tok/s** (~2.1%). All 32 first-final answer checks passed, and **all 32 complete 128-token sequences exactly matched the baseline slot for slot**. Sampled GPU memory was unchanged, about **97,061 MiB** on GPU0 during decode, leaving less than 1 GiB of device headroom. Raw JSON, validation and log: `D:/CantorAI/work/vast-54543362/garnet-batch-throughput/arithmetic-direct16-b32-o128.*`. Optimized vLLM still leads at **2,832.10 decode / 2,785 full-request tok/s**. Decision remains **INCONCLUSIVE** until repeat trials and other shapes; serialized scratch remains an explicit limitation.

#### OPT-0020 benchmark boundary correction, 2026-10-07

Code inspection found that the old Garnet batch runner timed the first prefill execution after engine loading, while vLLM warmed the whole request before its timed batch. It also summed decode call durations while running GPU-memory subprocesses and progress logging between calls; its `decode_wall_seconds` therefore was not a continuous wall window. Preserve the old raw results, but do not silently treat them as fully matched warmed serving latency. The revised runner opts into prefill warming with **`GARNET_BATCH_PREFILL_WARMUP=1`**, records excluded warmup calls, and measures a continuous decode wall window without those diagnostic subprocesses/logs. It retains the summed call time as a separate diagnostic. New trials must establish an unchanged-operator baseline under this boundary before claiming a new full-request gain.

**`GARNET_BATCH_DECODE_TRIALS=3`** repeats decode from the same first prefill token while preserving only the original input KV prefix. Every trial overwrites generated-token KV slots in order and bounds attention with its current context length. These repeated windows are decode stability checks, not extra full-request trials; every slot in every trial is validated independently. The result now saves the source commit, performance environment, hardware/budget, BF16 KV bytes per logical token, allocated versus logically filled KV, sampled memory maximum and explicit measurement limitations. Engine loading and the prefill-to-decode engine handoff remain excluded from the execution throughput; the existing build/load time fields retain that substantial cost. This boundary does not prove production serving latency with simultaneously resident prefill/decode engines.

The target build at **`37d72ed`** passed the complete synthetic suite, including the new vector-update cases on both GPUs and Qwen compatibility. Before the next Garnet A/B, refresh the optimized vLLM reference with **three actual full-request trials** (`VLLM_BATCH_TRIALS=3`) and check both **4,096 and 8,192 max batched prefill tokens** (`VLLM_BATCH_MAX_BATCHED_TOKENS`). The [official release list](https://github.com/vllm-project/vllm/releases) still marks **0.31.0**, released October 5, as latest stable when checked October 7 UTC. The [official GPT-OSS recipe](https://github.com/vllm-project/recipes/blob/main/OpenAI/GPT-OSS.md#recipe-for-nvidia-blackwell--hopper-hardware) recommends the larger 8,192-token budget; retain whichever supported configuration is faster rather than selecting a weaker baseline. Keep BF16 KV, TP2, greedy output128, input256, batch32, context4096 and no prefix cache matched; the recipe's FP8 KV option would change the agreed precision. VLLM's repeats include a fresh prefill with no prefix-cache reuse, whereas Garnet's extra decode-only repeats do not supply independent complete-request samples.

#### OPT-0020/0023/0024 repeated batch-32 results, 2026-10-07

**Source / configuration:** Garnet `e591a7e`, native plugin/backbone built at `37d72ed`; British Columbia TP2 Workstation, homogeneous arithmetic batch32, actual input256/output128, BF16 KV capacity4096 per slot. One identical-KV prefill warmup and one decode-step warmup excluded. All GPU inference ran sequentially: refreshed vLLM first, then Garnet unchanged-operator baseline, inline vector controls alone, and controls plus direct16 reduction. Environment and hardware are preserved in each Garnet JSON.

| Configuration | Decode trial 0 / 1 / 2, aggregate tok/s | Prefill execution, s | Complete execution output tok/s, first trial |
|---|---:|---:|---:|
| vLLM 0.31.0, max batched tokens4096 | 2755.74 / 2688.92 / 2883.18 | See per-request first-token times in raw results | 2694.24 |
| vLLM 0.31.0, max batched tokens8192 | 2907.38 / 2941.49 / 2933.26 | See per-request first-token times in raw results | 2857.43 |
| Garnet baseline, batch NCCL + host vector copies | 2595.53 / 2622.43 / 2622.50 | 2.70634 | 958.78 |
| Garnet inline vector controls, batch NCCL | 2637.40 / 2670.44 / 2668.63 | 2.72229 | 960.78 |
| Garnet inline vector controls + direct16 | 2759.80 / 2775.95 / 2791.60 | 2.73462 | 973.57 |

**Validation:** All 96 slot/trial answers pass for each Garnet configuration and each vLLM configuration. All 96 complete 128-token sequences for each Garnet configuration exactly equal the corresponding slot in the Garnet baseline's first trial. This supplements the full synthetic suite's vector-update API, graph, collective, and Qwen compatibility checks. Garnet repeats measure decode only from the original input KV; vLLM repeats are independent complete requests. Do not compare their trial counts as independent full-request sample counts.

**Analysis / decision:** Median inline-controls decode improves **2622.43 → 2668.63 tok/s (~1.8%)**; adding direct16 improves **2668.63 → 2775.95 (~4.0%)**; combined gain over baseline is **~5.9%**. Use the stronger 8192-token vLLM configuration going forward: its median **2933.26** remains **~5.7% faster than Garnet**, and its complete-request rates are **2857.43 / 2885.52 / 2842.44 tok/s**. Garnet's ~2.7-second prefill still dominates its complete-execution deficit. OPT-0023 and OPT-0024 remain **INCONCLUSIVE, opt-in** pending other batch/length/prompt shapes and independent repeat sessions. Direct scratch still requires serialized paired streams.

**Memory / measurement limits:** Each Garnet configuration allocated **4608 MiB BF16 KV per GPU** for 32×4096 reserved tokens. Actual input KV occupied **288 MiB**, rising to **430.875 MiB** after the last generated token's required cache write (383 cached positions/slot; the final sampled token is not processed). Sampled device memory peaked at **97095 / 97093 MiB**, leaving about 792 MiB on GPU0. The planner's estimated footprint does not yet account accurately for all lazy Marlin repacking; a successful estimate is not evidence of safe peak headroom. Decode engine load/handoff was **16.40–16.49 s**, excluded from the execution rates above; production request latency with this engine lifecycle would be substantially worse. These samples do not establish simultaneous resident-engine serving capacity or a guaranteed allocation peak.

**Evidence:** `D:/CantorAI/work/vast-54543362/garnet-batch-throughput/arithmetic-wall-{baseline,controls,combined}-b32-o128.{json,validation.json}` and `D:/CantorAI/work/vast-54543362/vllm-batch-throughput/arithmetic-refresh-mbt{4096,8192}-b32-o128.{json,validation.json}`. Raw files retain all slot token IDs and trial timings. No candidate is made default and the overall vLLM-beating goal remains unmet.

#### Workload profile contract: batch, lengths and KV, 2026-10-07

The user clarified that the primary objective is **maximum aggregate batch output throughput**, with KV cache and actual input/output lengths included in the profile. Record each performance point with GPU/topology, TP degree, kernel flags, batch/concurrency, actual per-request input and generated token counts, reserved maximum context, KV dtype/page allocation, logically occupied KV, and sampled resident/peak memory. Record prefill/first-token latency, warm decode output tok/s, complete-request output tok/s, correctness and timing exclusions alongside those fields. A maximum is the best measured point within the tested workload and memory envelope, not a universal hardware constant.

Input length drives prefill work and initial KV occupancy; output length increases KV occupancy during decode and changes how heavily complete-request throughput weights decode. Reserved context sets memory capacity, which can exceed the actual input-plus-output length by a large amount. KV capacity must admit every slot's input and requested output while leaving room for weights, repacking, activations, workspace and graph/runtime allocations. Longer contexts can lower the feasible batch or change the best kernel configuration. Keep matched precision, actual lengths and output work when comparing vLLM, while exposing different allocation policies. The current repeated-prompt runner only measures homogeneous batches; mixed-length requests and continuous admission are still unverified. Tune across short/long inputs and outputs and report the throughput/latency/memory tradeoff rather than choosing a shorter context solely to improve one headline rate.

#### OPT-0022 batch-32 chunk128 failure, 2026-10-07

At `1ca34e3`, the combined decode candidate with batch32/input256/output128/context4096 tried two128-token prefill chunks, keeping each MoE call at4096 rows. The first chunk's warmup failed with TensorRT `pluginV2DynamicExtRunner` followed by `compiled_engine_execution_failed: TensorRT enqueueV3 failed`; no output JSON or throughput result was produced. The process exited and both GPUs were confirmed idle before any follow-up. Preserve `arithmetic-wall-combined-chunk128-b32-o128.log` rather than overwriting this failure. This is **not an accepted optimization**.

The native plugin omitted the underlying CUDA status, so the failure cannot yet be attributed to OOM, indexing, or another runtime cause. Memory pressure is a hypothesis: unlike the8192-row grouped baseline prefill, this4096-row Marlin path also allocates lazy packed weights while retaining the large prefill engine/workspace and4096-context KV. Code inspection and checkpoint headers show about69.58GB of conservatively counted full-checkpoint weights (including doubled BF16 constants); each TP2 plugin currently retains full original expert constants and separately repacks its owned half. Correcting this duplication is a potential memory/serving improvement, not yet implemented or measured.

Added bounded CUDA-error diagnostics (operator kind, rank, device and row count) on plugin failure and a `GARNET_GPT_OSS_PROFILE_PREFILL=1` switch that brackets only measured prefill calls after warmup. Instrumented timing must stay separate from unprofiled benchmarks. The next action is to reproduce the terminal failure with diagnostics, then use matched shorter-context trials or fix the identified cause. Do not restart merely because an observation timed out.

#### OPT-0022 diagnostic confirmation, 2026-10-07

The diagnostic repeat at **`8dc8084`** reproduced the failure and then exited. Both MoE ranks reported **CUDA error2 (out of memory), kind2, rows4096**. Sampled memory before prefill warmup was **68253 / 68235MiB**; lazy Marlin packing then failed. This establishes OOM rather than a merely inferred memory cause. No performance or output validation can be claimed for this shape. Raw log: `arithmetic-wall-combined-chunk128-diagnostic-b32-o128.log` in the same evidence directory. Both GPUs were confirmed idle before native source synchronization/build.

### OPT-0025: Rank-local original expert constants and refit regeneration

**Date / source:** 2026-10-07, local candidate on `feature/gpt-oss-120b`; opt-in `GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS=1`. Relevant files: GPT-OSS TensorRT lowering/refitter helper, plugin options/kernels, xModel stage and TP2 planner.

**Hypothesis / bottleneck:** Every rank retains all original expert weights even though it only executes its even/odd half, and then creates a packed copy of that half. Removing the unused original half should free roughly half the original expert-constant footprint, making large prefill and batch/KV shapes feasible. This is primarily a memory/lifecycle hypothesis; no speed gain is assumed before measurement.

**Implementation:** Gather only the owned expert-axis rows as TensorRT constants while keeping router weights, selected expert IDs and routing global. Warp, grouped fallback and Marlin bias/weight indexing map owned global expert IDs to local storage. Refittable constants carry a versioned GPT-OSS derived name; engine reload regenerates exactly the same bytes from the original safetensors, without extra checkpoint files. Plugin serialization advances7→8 and the sharded storage flag changes the placement/cache identity. The sharded planner includes padded MXFP4/E8M0 Marlin repacking alongside original constants; the legacy unsharded estimator retains its known omission and must not be treated as a guaranteed peak fit.

**Validation / result:** Local CPU byte mapping and refit-regeneration tests pass for even/odd expert counts, one-/two-/multi-byte rows and invalid inputs. Synthetic TP2 memory-admission/cache-identity tests pass, including padded layout and one-byte-under-budget rejection. Python syntax checks pass. Extended CUDA parity compares rank-local original weights with full original weights exactly in warp/grouped paths and Marlin; expanded Marlin checks cover every enabled synthetic shape, not just token counts≤8. Cold and cached TP2 graph tests exercise derived refit names, including an odd5-expert fixture. Target compilation, CUDA results and pretrained validation are pending.

**Decision:** INCONCLUSIVE, opt-in. Keep default storage and Qwen behavior unchanged. Before throughput, require target parity and successful cold/cached refitting, then measure peak memory and all saved pretrained answers. Removing duplicate weights does not by itself solve complete-request engine handoff or heterogeneous scheduling.

#### OPT-0025 target correctness gate, 2026-10-07

At **`84ecc3d`**, the target backbone, plugin and new tests compiled successfully for SM120. The complete verification suite passed with Marlin4096/block32 and direct16 enabled: exact rank-local versus full-constant CUDA comparisons in warp/grouped/Marlin paths, synthetic TP2 memory admission and byte gathering, scalar/vector updates, cold/warm compiled execution, TP collectives, cold and cached sharded TP2 MoE graphs, two-GPU pipeline generation, module/API requirements and Qwen compatibility. Logs: `/workspace/CantorAI/work/expert-weight-shard-verification/` and its parent summary log. This verifies synthetic correctness/refitting, not pretrained serving performance.

The next screen records a fresh vLLM reference first, retaining the exact input token IDs in both engines' results. It then runs sharded constants with unchunked prefill (same operator dispatch as the previous baseline) and chunk128 with1/2/4 Marlin CTAs per SM, sequentially. Each configuration validates all96 slot/trial outputs before the next starts. This combines the memory fix and workload-specific prefill/kernel screen in one verification cycle, while keeping each mode's evidence separate. Decision remains INCONCLUSIVE until measured pretrained results.

#### OPT-0023 larger-batch preparation, 2026-10-07

While the sharded batch32 screen is running, prepare a separate **`GARNET_GPT_OSS_DIRECT_LARGE_BATCH_ALLREDUCE=1`** gate for batch65–128 hidden-state messages. This doubles staging capacity from64×2880 to128×2880 FP32 elements, adding720KiB/GPU. The existing gate alone still sends messages above64 to NCCL. Keep the same serialized paired-rank handshake and CTA choices; do not infer concurrent-serving support. The microbenchmark now permits batch128 and refuses a claimed direct test when the necessary gate is absent. Target changing-input parity and NCCL/direct timing at128 are pending; do not pull/build this source over the already running pretrained screen. The motivation is to test maximum aggregate output throughput with the larger batches potentially admitted after OPT-0025, not to redefine success around batch32.

#### OPT-0025 preliminary pretrained memory result, 2026-10-07

The first sharded run at **`58cd9fb`**, native build `84ecc3d`, preserves the previous unchunked batch32/input256/output128/context4096 dispatch. All96 answers across three decode windows pass. Sampled GPU0 memory before prefill execution fell **68721 → 39589MiB** relative to the corrected full-constant baseline; sampled peak fell **97095 → 68207MiB**, a reduction of **28888MiB (~28.2GiB)**. Warm unchunked prefill is **2.66277s**, versus2.70634s previously, and decode windows are **2786.82 / 2534.24 / 2804.06 aggregate tok/s**. The low middle trial is retained rather than hidden; no statistical serving-speed improvement is established by the memory change alone. The fresh vLLM reference gives **2998.45 / 2976.32 / 2945.51**, with all96 expected answers passing and exact input IDs recorded. The complete Garnet execution rate is **993.92 tok/s**, so the overall goal remains unmet. Subsequent chunk/CTA runs are still active; full token-sequence comparisons and independent repeats remain pending.

#### OPT-0020 workspace-profile correction, 2026-10-07

Code inspection found that Marlin's row-admission flag and block size change `getWorkspaceSize()` and internal buffer offsets, but were absent from the TP2 placement/cache identity. Builder auto-sizing also considered only grouped fallback scratch. Add normalized Marlin workspace settings to the profile/cache identity and record them in results; reject a settings change between planning and engine construction. The builder allowance now covers **max(grouped scratch, aligned Marlin scratch)**, retaining the exact16-byte allocation alignment. A native `--workspace` report exposes the actual C++ allocation methods for CPU cross-checks at1/8/32/128/513/4096/8020 rows and both8-/32-row tiles. Local cache-identity/admission and Python checks pass; target cross-checks pending. This is a cache/memory-contract correction, not evidence that the previously rejected8192-row experiment failed for this reason. Keep that experiment rejected unless its own numerical defect is resolved and validated.

#### OPT-0025 sequential pretrained screen, 2026-10-07

Source58cd9fb/native84ecc3d, TP2 BF16, homogeneous batch32/input256/output128/context4096, sharded original experts and direct16, inline controls, tiled16 attention, Marlin4096/block32. Fresh optimized vLLM0.31 ran first with8192 max batched tokens, prefix reuse disabled, three full requests: decode2998.45/2976.32/2945.51 and full2950.49/2919.83/2896.33 aggregate output tok/s. Both engines retain exact input IDs. Garnet's three trials per mode repeat decode from the original input KV, with one warmed prefill and engine loading/handoff excluded.

| Mode | Warm prefill s | Decode aggregate output tok/s, three trials | First complete-execution tok/s | Sampled peak MiB/GPU |
|---|---:|---|---:|---|
| Unchunked, CTA1 |2.66277|2786.82 /2534.24 /2804.06|993.92|68207 /68189|
| Chunk128, CTA1 |0.82795|2434.08 /2443.89 /2443.35|1639.99|68713 /68695|
| Chunk128, CTA2 |0.83473|2522.26 /2540.68 /2493.45|1674.58|68749 /68731|
| Chunk128, CTA4 |0.84205|2414.72 /2431.29 /2412.94|1622.14|68749 /68731|

All96 expected arithmetic answers per mode passed. Chunking changes token trajectories: CTA1 matches the unchunked full128-token sequence in only1/32 slots in each trial, CTA2 in0/32; do not assume decode times represent identical expert routing. The low middle unchunked trial is retained without an invented cause. Peak memory falls about28888MiB (~28.2GiB) compared with the earlier full-original baseline. Chunk128 no longer OOMs and prefill falls about69%, but decode regresses and full execution throughput remains below vLLM. KV allocation is4608MiB/GPU; logically occupied KV is288MiB after input and430.875MiB after383 processed positions/slot. Sampled memory is not a guaranteed allocation maximum. Raw results, validation and logs are retained locally in D:/CantorAI/work/vast-54543362/garnet-expert-weight-shard-screen and remotely in /workspace/CantorAI/work/garnet-expert-weight-shard-screen. Decision remains INCONCLUSIVE pending broader prompt/length/batch correctness and serving measurements.

#### Independent batch-decode tile candidate, 2026-10-07

The existing Marlin prefill-block flag applies by flattened row count and therefore also selects32 for batch128 decode. Add an opt-in GARNET_GPT_OSS_MARLIN_DECODE_BLOCK=8/32, propagate the MoE prefill phase explicitly from xModel, and include the override in workspace/cache identity. Unset preserves existing dispatch. Homogeneous routing can favor larger expert tiles, while diverse routing may waste padded rows; neither tile is declared faster without matched model measurements. Add128-row decode parity and native workspace cross-checks for prefill8/32 and decode inherited/8/32 across dispatch boundaries. Local Python/cache-admission checks pass; target validation and performance pending.

#### Batch128 preparation and KV profile evidence, 2026-10-07

Native build7272984 and the full synthetic suite passed, including the C++/Python workspace-byte cross-check for inherited/8/32 decode tiles,128-row MoE decode and sharded cached graphs. Native128 peer-reduction timing and override8 parity are the next isolated gates. Add a read-only vLLM worker extension to capture actual unique CUDA KV backing storage (deduplicating alias views), per-layer dtype/window/block settings, logical full and sliding-retained history estimates, plus hardware/topology and memory samples. Capture RPCs outside timed inference. Logical history at completion is an estimate, not sampled live scheduler occupancy; page rounding and in-flight work are excluded, and completed requests have been freed. PyTorch peak counters exclude external allocators. Refuse result overwrite to preserve evidence. Keep reserved context and actual input/output lengths separate in every profile. This instrumentation does not change vLLM kernels, cache policy or timed sampling.

### OPT-0026: Independent batch128 decode expert tiles

**Date / Commit:** 2026-10-07; native7272984, read-only KV profiles26e30fe, screenb28ad1a, feature/gpt-oss-120b. Files: gpt_oss_llm.py, gpt_oss_lowering.cpp, gpt_oss_marlin.cu, pipeline.py, kernel_parity.cpp, tp_weight_placement.py, screen_batch_tiles_tp2.sh.

**System:** British Columbia2x RTX PRO6000 Blackwell Workstation, SM120, TensorRT11.2.1.2/CUDA13.4, TP2 attention/EP2 expert ownership, MXFP4 experts, BF16 activations/KV. Direct peer reduction uses serialized paired streams. Retain hardware/topology in result JSON.

**Workload:** Saved256-token arithmetic input repeated128 times,512 fixed generated tokens/slot,1024 maximum context. One identical-KV prefill warmup per32-token chunk, three decode trials reusing original input KV. vLLM0.31 TP2,8192 max batched tokens, chunked prefill/no prefix reuse, greedy/no EOS stopping, one full-batch warmup and three complete measured requests. vLLM runs first, then Garnet with no competing GPU inference.

**Baseline / Observed bottleneck:** At batch32 chunked prefill is faster but decode falls to2413-2541 aggregate tok/s, versus fresh vLLM2946-2998. The Marlin prefill32 flag also controls batch128 decode by flattened row count. Existing direct collective capacity ends at64 rows. These dispatch limits and the earlier batch8 profile (Marlin25.3%, NCCL25.1% of combined GPU kernel-duration sums) motivate measuring larger batches and expert scheduling. Profile duration sums are not unprofiled wall shares.

**Hypothesis / Proposed optimization:** Permit separate8/32 decode tiles while retaining32-row prefill, and screen1/2/4 CTAs per SM. Smaller tiles may reduce padding for diverse routing; larger tiles may reuse expert weights better in homogeneous batches. Extend peer staging to128 behind a separate gate and select CTAs from isolated parity/timing. No tile is declared best beforehand.

**Implementation:** Pass prefill phase in MoE xModel attributes through the backbone and plugin. GARNET_GPT_OSS_MARLIN_DECODE_BLOCK=8/32 applies only to decode rows>=128; unset preserves legacy selection. Include the override in workspace/cache identity and exact byte estimates. The Linux screen checks exact IDs/context against the reference, runs six modes sequentially, refuses to overwrite logs/results and validates every slot/trial before advancing. Direct128 adds720KiB staging/GPU and remains opt-in/serialized-only. At batch128 the inline-vector control path remains disabled because its existing capacity is64; results record the actual fallback.

**Correctness validation:** Full target synthetic suite passed at7272984, including cold/cached sharded graphs and all native/Python workspace sizes. Override8 dedicated CUDA parity passed, including128-row decode and exact rank-local/full-original expert comparisons. Direct128 validates every element through changing-input graph replays, five rounds of216 reductions (1080 total). Native and screen logs are retained. Pretrained Garnet screen remains in progress.

**Performance result:** The direct-only128 message (368640 FP32 elements,72 graph reductions per replay,50 timed replays/trial,five trials) measured NCCL51.287us median, direct16 42.213us, direct32 41.715us. This is about18.7% less isolated collective latency at32 CTAs, not a model speedup. The refreshed vLLM batch128 decode rates are5403.49/5394.05/5307.65 aggregate tok/s; full-request rates5390.98/5379.66/5292.63. All384 answers pass. Garnet full-model before/after numbers are pending.

**Memory profile:** vLLM actual unique backing allocation is42567008256 bytes/GPU (39.64GiB), matching its startup log; shared views are counted once. Full-history data at767 processed tokens/slot is3619160064 bytes/GPU (3.37GiB), while18 sliding layers reduce logical retained history to2111569920 bytes (1.97GiB). The latter is an estimate excluding page rounding and in-flight tokens, not live scheduler occupancy. Sampled memory peak is79811/79813MiB. Garnet reserves4608MiB/GPU at128x1024 and retains full history for sliding layers; expected logical data at767 positions is3.37GiB. Actual Garnet peak is pending. Different cache allocation policies are visible; actual IDs, lengths, context limit and precision match.

**Decision:** INCONCLUSIVE. Native gates pass, but neither a microbenchmark gain nor a single arithmetic shape establishes the serving target.

**Analysis / Next Step:** Use each result's token trajectories to identify numerical/routing changes and compare complete-execution rates as well as decode. Broaden all four prompts and input/output lengths after the screen; mixed lengths and continuous admission remain unverified. Engine handoff is excluded and remains a production-latency limitation. Raw reference: D:/CantorAI/work/vast-54543362/vllm-batch-throughput/arithmetic-kv-profile-b128-o512.json; target screen: /workspace/CantorAI/work/garnet-batch128-tile-screen.

#### OPT-0026 first pretrained modes, 2026-10-07

At sourceb28ad1a/native7272984, decode block8/CTA1 gives2956.42median aggregate output tok/s (2982.82/2956.42/2928.46),3.30221s warmed prefill and2597.50first complete-execution tok/s. Block8/CTA2 gives2918.44median (2952.16/2918.44/2909.44),3.35355s prefill and2569.08full execution. All384 expected answers per mode pass. Sampled peaks are68717/68699 and68753/68735MiB respectively. Both reserve4831838208KV bytes/GPU, with3619160064logical history bytes at767 positions/slot. Tile8 is not a win against vLLM5394.05median decode/5379.66median full requests, and comparing it with the yet-unmeasured tile32 on this exact shape remains pending. Other four modes are still running; preserve all results before choosing a profile.

#### OPT-0012 batch/length-specific follow-up hypothesis, 2026-10-07

The earlier8/32 split rejection was for batch1; it cannot establish the best batch128 configuration. Code inspection of the currently enabled16-split attention launches128x32x16=65536partial CTAs per rank per layer, each with four warps. A128-token sliding range therefore gives only two keys per warp. This suggests measuring the already-supported unsplit path with4/8/16warps and the8-split path on the matched batch128 input256/output512/context1024 profile, after the MoE tile screen is terminal. The unsplit path launches4096CTAs/rank/layer, but each CTA does more serial work. Fewer CTAs may reduce scheduling/merge overhead at high batch, or longer serial loops may regress; no performance claim yet. Run native parity with each exact setting before pretrained validation and retain all token trajectories. A batch-aware GQA reuse redesign is a later candidate if profiling still identifies attention as substantial. Keep the baseline and candidate settings explicit in the hardware/workload profile.

#### OPT-0012 isolated batch-attention screen preparation, 2026-10-07

Add attention_batch_benchmark.cu and its standalone CMake target to measure a36-layer alternating full/sliding128 attention graph, with independent paged BF16 KV per layer (avoiding a repeatedly L2-cached single layer). CLI records batch, processed context length and allocated KV. Ten graph warmups precede five GPU-event replay trials; timers include KV writes and attention/merge kernels, excluding host checks. Synthetic nonzero GQA32:4/head64 data uses reversed pages, one missing page and an inactive slot. Validate sampled double-precision full/sliding outputs, all-output finiteness and inactive zeros before and after timing. This sampled check complements, rather than replaces, the full native parity executable. Local CUDA12/SM89 compilation passes; target SM120 build and split/warp screen wait until the active pretrained MoE screen is terminal. No plugin/backbone binary or existing inference behavior changes in this benchmark-only commit.

#### OPT-0012 local attention screening, 2026-10-07

The new benchmark passed a CUDA12/RTX4080 smoke check at batch2/context64 with zero sampled reference error. A first Windows-to-WSL shell-loop invocation accidentally passed empty split/warp environment values; five27.6-27.8ms medians all used the fallback16-split path and are INVALID as an A/B screen. Preserve the failure and add strict setting validation in the benchmark to reject blank/unsupported values before timing. Re-run each variant with direct env arguments (no shell-variable interpolation).

The corrected, sequential RTX4080 batch128/processed-context767/repeats20 screen,36 independent KV layers (3623878656bytes,18full/18sliding128), gives GPU-event medians: split16=27.6077ms, split8=26.5630ms, unsplit4warps=27.8236ms, unsplit8warps=26.0817ms, unsplit16warps=38.9338ms. Sampled full/sliding double-reference maximum absolute error is zero in the first four modes and4.76837e-7 for unsplit16; all outputs finite and inactive slot zero. This is ~5.5% isolated improvement for unsplit8 and a41% regression for unsplit16 on another GPU, not a Blackwell or full-model result. Target compilation, full native parity and performance still pending. A large reduction in CTA count alone does not guarantee proportional speedup; the longer serial softmax walk can regress. Next collect the SM120 screen and full-model/kernel attribution before implementing a larger GQA reuse or expert-distribution redesign.

#### OPT-0026 completed six-mode pretrained screen, 2026-10-07

Sourceb28ad1a/native7272984, same batch128/input256/output512/context1024 arithmetic profile and chunk32. All384 expected answers per mode passed; this does not establish identical token trajectories or broader prompt correctness.

| Decode block / CTAs per SM | Warm prefill s | Decode aggregate output tok/s, three trials | First complete-execution tok/s | Sampled peak MiB/GPU |
|---|---:|---|---:|---|
|8 /1|3.30221|2982.82 /2956.42 /2928.46|2597.50|68717 /68699|
|8 /2|3.35355|2952.16 /2918.44 /2909.44|2569.08|68753 /68735|
|8 /4|3.38830|2853.75 /2826.86 /2813.78|2491.07|68753 /68735|
|32 /1|3.30244|3282.69 /3238.51 /3230.35|2821.48|68717 /68699|
|32 /2|3.36538|3083.30 /3061.55 /3046.22|2666.34|68753 /68735|
|32 /4|3.39621|2931.78 /2913.40 /2890.66|2549.43|68753 /68735|

Block32/CTA1 is the next screening configuration; its median remains40.0% below vLLM5394.05 decode, and complete execution remains below vLLM5379.66 median full requests. Garnet trials reuse original input KV, and its execution rates exclude engine build/load/handoff. The initial block8/CTA1 cold prefill224.28s and decode engine setup220.14s are retained; cached block8/CTA2 setup42.12/40.62s and CTA4 setup42.25/40.82s also remain outside execution timing. This is a substantial production latency problem. Reserved KV4.5GiB/GPU versus3.37GiB logical history at767 processed positions remains explicit. Decision INCONCLUSIVE; no accepted-stage advancement. Local evidence: D:/CantorAI/work/vast-54543362/garnet-batch128-tile-screen and batch128-tile-evidence.tgz.

#### OPT-0012 target isolated attention screen, 2026-10-07

Targetd8206a4, SM120/CUDA13.4, batch128/processed context767,36 independent alternating full/sliding128 KV layers, five trials of20 replays after10 warmups. Full native kernel parity passed separately for each configuration. Sampled double-reference checks, finiteness and inactive zeros passed: split16 median21.9461ms; split8 20.3816ms; unsplit4warps19.4534ms; unsplit8warps19.3774ms; unsplit16warps21.6441ms. Maximum sampled absolute error0 except unsplit16=4.76837e-7. The best existing knob gives11.7% isolated improvement, not a measured model win; it cannot alone prove closure of the40% batch decode gap. Remote logs: /workspace/CantorAI/work/attention-{split16,split8,unsplit4,unsplit8,unsplit16}-{parity,benchmark}.log. Next test shared GQA KV staging rather than infer proportional gains from CTA reduction.

### OPT-0027: Shared paged KV tiles for grouped-query decode

**Date / Commit:** 2026-10-07, local candidate on feature/gpt-oss-120b. Files: plugins/gpt_oss/cuda/gpt_oss_kernels.cu, test2026/gpt_oss/kernel_parity.cpp, attention_batch_benchmark.cu.

**System / Workload:** Initial CUDA12/SM89 RTX4080 local screen; batch128, processed context767, qHeads32/kvHeads4/headDim64,36 independent BF16 KV layers (18 full,18 sliding128). Five20-replay GPU-event trials,10 graph warmups. Target Blackwell results pending.

**Baseline / Observed bottleneck:** Current attention launches one partial CTA per query head and split, repeatedly loading the same KV for eight query heads. Default16-split local36-layer timing27.6077ms; target21.9461ms. Best existing target unsplit8 is19.3774ms. These synthetic kernel times exclude projections/MoE/TP/host and are not additive model wall shares.

**Hypothesis / Proposed implementation:** Stage16 paged KV positions once per KV head and split, sharing FP32-converted K/V across eight query warps. Reduce repeated global loads, page lookup and conversion; longer warp loops and barriers may erase gains. Keep FP32 scores/online softmax, ordinary expf, BF16 final output and sink added once.

**Implementation:** Opt-in GARNET_GPT_OSS_DECODE_GQA_TILED=1, head64 and GQA8:1 only; other geometries fall back. GARNET_GPT_OSS_DECODE_GQA_SPLITS=4/8/16 defaults8. Eight warps per CTA; existing partial scratch stride130 and merge preserve workspace/ABI. No approximate exponential or quantized attention probabilities. Native parity now includes full output checks for8:1 and32:4 heads with301-position paged full/sliding histories, missing page, inactive slot and strong positive/negative sinks. Benchmark rejects invalid/blank GQA labels. Fix benchmark scratch from64x66 to64x130 floats/head so supported64 splits fit; previous8/16-split results stayed within allocation bounds.

**Correctness validation:** Local smoke batch2/context64 passes; batch128/context767 sampled full/sliding double reference maximum error0 at4/8 splits,4.76837e-7 at16; all outputs finite and inactive slot zero. Full native target gate and pretrained quality pending. Local compilation passed.

**Performance result:** Local medians9.11836/9.13295/9.7028ms at4/8/16 splits, compared with existing default27.6077ms. No target model gain claimed. Raw logs D:/CantorAI/work/vast-54543362/attention-gqa{4,8,16}-local.log.

**Decision / Analysis / Next Step:** INCONCLUSIVE, opt-in. Validate full native parity for each exact setting on SM120, then target attention timing; only if those gates pass run best expert32/CTA1 pretrained profile sequentially against the saved optimized vLLM reference. Preserve all four prompts, batches, input/output lengths, KV/memory and complete-request boundaries; one arithmetic screen cannot complete the goal. Source is authored locally, with no Qwen edits or copied external attention source.
#### OPT-0027 target native gates and length screen, 2026-10-07

Nativefae4735 built on SM120. Full kernel parity passes separately atGQA4/8/16 splits, including new8:1 and32:4 full-output paged301-position fixtures, learned sinks, missing pages, inactive-cache preservation, full/sliding and fallback geometries. The full verify.py --multi-gpu suite passes atGQA8: cold/warm compiled graphs, refit/workspace, scalar controls, token generation, TP collectives/MoE/sharded graphs, cold/warm two-GPU pipeline/generation, plugin requirement/native bridge and Qwen compatibility. These are synthetic gates; pretrained results still pending.

Blackwell batch128/context767 medians are7.09845/7.11113/7.72328ms atGQA4/8/16, compared with existing split16=21.9461ms and best existing unsplit8=19.3774ms. Sampled maximum absolute error0/0/4.76837e-7. At processed context256 medians2.79646/2.68648/3.19933ms, and512 medians5.24150/5.09944/5.94648ms. The latter short-duration trials vary widely (for example context256/GQA8 ranges1.83228-4.29082ms), so they do not establish4 versus8 as a robust winner. Keep raw trials and run all three settings on the full model. Added screen_batch_attention_tp2.sh for sequential fixed expert32/CTA1, GQA4/8/16, matching reference IDs/context/output, exclusive GPU access and384 answer validations per mode. Target native evidence archive: D:/CantorAI/work/vast-54543362/attention-native-fae4735-evidence.tgz; full verify remote /workspace/CantorAI/work/gqa-fae4735-verify. Decision remains INCONCLUSIVE until pretrained correctness and model timing.

#### OPT-0026 token trajectory comparison, 2026-10-07

Compared every512-token slot across all three decode trials and six modes against block32/CTA1 trial0. Same-mode repeats match128/128 slots in every trial for every mode. Across modes, exact slots are32/CTA1=128,32/CTA2=0,32/CTA4=0,8/CTA1=1,8/CTA2=2,8/CTA4=0 (each count repeats in all three trials). All modes have exact matched256 input IDs and512 output tokens per slot and pass final-answer validation, but expert scheduling changes numerical trajectories/routing; this limits causal attribution of speed differences. Local evidence token-trajectory-comparison.json beside the six modes. Do not describe the six-mode answer success as exact token equivalence.
#### Batch128 latency boundary and profiling preparation, 2026-10-07

The saved optimized vLLM batch128 reference records per-request first-token timestamps, not just first delivery. Trials have min/median/max TTFT0.0487/0.7770/1.4379s,0.0529/0.7665/1.4500s,0.0557/0.9159/1.4374s; complete wall12.1566/12.1822/12.3825s. Its aggregate decode window starts at the earliest first token and includes staggered prefill/first-token delivery for later requests. Garnet completes homogeneous prefill before decode; its earlier3.302s warm prefill and separate engine handoff cannot be compared only with vLLM's minimum TTFT. Preserve all latency distributions and complete execution as well as decode. The workload/input/output/precision is matched, but scheduling differs.

Added optional non-overwriting Nsight wrapper to benchmark_batch_tp2.sh and a recorded, bounded GARNET_GPT_OSS_PROFILE_DECODE_START offset (default10), so early/late KV contexts can be profiled without changing measured input/output. Explicit decode-range gating also prevents a prefill-only job from starting another decode capture. Local shell/Python syntax checks pass; target profiling awaits terminal GQA model screen. No inference kernel/runtime behavior changes in this instrumentation. Keep instrumented rates separate from unprofiled comparisons.

#### OPT-0027 first pretrained setting, 2026-10-07

Source4bd4822/nativefae4735, Blackwell TP2/EP2, unchanged expert32/CTA1/direct128, batch128/input256/output512/context1024/chunk32. GQA4 yields4608.77/4580.18/4562.24 aggregate decode tok/s (median4580.18), versus earlier3238.51, a41.4% median improvement. Complete execution3745.44 versus2821.48 (+32.7%); warm prefill3.30548s, unchanged memory peak68717/68699MiB/GPU, reservedKV4831838208bytes and logicalfullhistory3619160064bytes at767positions. Engine build/load excludes223.82s prefill and219.04s decode. All384 arithmetic answer checks pass. Exact256 input IDs and512 output lengths match; none of the128 full token sequences matches the prior baseline in any trial, while repeats within this mode match128/128 slots in all trials. Preserve the changed numerical/routing trajectories; this is not exact pretrained token equivalence or an isolated causal timing of attention alone.

vLLM median5394.05 decode/5379.66 full requests still leads: GQA4 decode15.1% lower, complete execution30.4% lower. Scheduling/timing boundaries described above remain applicable. GQA8/16 settings are still running sequentially, and code/instruction/long-prompt plus other batch/length correctness/performance are pending. Do not extrapolate the batch128 gain to small batch or promote the option to default. Raw evidence is backed up/extracted at D:/CantorAI/work/vast-54543362/gqa4-fae4735-evidence.tgz and garnet-batch128-gqa-fae4735; full native verify included. Decision INCONCLUSIVE until remaining gates; optimization goal stays active and unmet.
#### OPT-0027 completed pretrained GQA split screen, 2026-10-07

All three settings completed sequentially atsource4bd4822/nativefae4735 and passed384 arithmetic checks each. The workload, BF16KV allocation/logical history and fixed expert profile match the first result above.

| GQA splits | Warm prefill s | Decode aggregate output tok/s, three trials | First complete-execution tok/s | Sampled peak MiB/GPU |
|---|---:|---|---:|---|
|4|3.30548|4608.77 /4580.18 /4562.24|3745.44|68717 /68699|
|8|3.33960|4525.03 /4546.41 /4526.63|3682.98|68753 /68735|
|16|3.35235|4363.63 /4365.73 /4342.79|3573.06|68753 /68735|

GQA4 remains the next measured configuration; its approximately1.2% median lead over8 is small enough to warrant matched repeat/profile confirmation. All settings repeat128/128 complete512-token sequences within each mode, but GQA8/16 match0/128 GQA4 sequences in all trials. Cached build/load timingsGQA8=41.84/40.34s andGQA16=41.72/40.25s remain excluded. Full local evidence backed up/extracted atbatch128-gqa-fae4735-evidence.tgz and garnet-batch128-gqa-fae4735, including token-trajectory-comparison.json generated locally. Decision remains INCONCLUSIVE because other prompts/lengths and actual serving lifecycle are unverified and vLLM still leads.

Next run recorded-configuration Nsight captures at decode offsets10..41 (processed context266..297) and480..511 (context736..767), keeping384-slot screen evidence separate from the one-trial diagnostic128-slot validation. Use profile_batch_tp2.py with exact reference settings/IDs and no concurrent GPU job. Then attribute remaining per-rank kernel/collective/host costs and broaden the saved four prompts before selecting a larger next implementation; no unmeasured TP expert-distribution redesign is accepted.
#### OPT-0027 early/late Nsight attribution, 2026-10-07

Source82c3770/nativefae4735, GQA4/expert32/CTA1, same batch128/input256/output512/context1024. Both one-trial diagnostic runs pass128 arithmetic checks. Capture32 decode steps at offsets10..41 (processed context266..297) and480..511 (736..767). GPU0 spans793.425/1103.639ms, or24.795/34.489ms per step under tracing. Per-GPU kernel-duration sums below are not additive wall shares; TP waits/overlap and profiling overhead apply. Instrumented full-run decode3729/3701tok/s is not the unprofiled reference comparison.

| GPU0 group | Early32 steps ms | Late32 steps ms | Observation |
|---|---:|---:|---|
|Direct TP reduction,2304 calls|143.198|136.240|About4.3-4.5ms summed/step; synchronization included|
|Marlin experts,2304 calls|130.283|290.232|Routing distribution may increase padded expert work; requires counts/profiling before redesign|
|Shared GQA attention,1152 calls|85.616|215.905|Longer context increases attention work despite shared KV|
|Router scores,1152 calls|79.226|86.262|About2.5-2.7ms summed/step|
|Vocabulary all-gather,32 calls|50.849|50.675|About1.59ms/step to move full logits before greedy|
|Expert metadata,1152 calls|35.785|39.032|About1.1-1.2ms summed/step|
|Dense projection GEMMs,2304 calls|33.631|35.811|Attention/other projections|
|Vocabulary projection,32 calls|19.564|23.504|Necessary local shard scoring|
|Full-vocabulary top1,32 calls|9.026|9.068|About0.28ms/step scan after all-gather|

Runtime memcpy/synchronize duration sums649.4/637.2ms early and956.7/944.4ms late include waits and overlap GPU execution; do not call them independently removable host overhead. The scripts formerly used for batch8 had a hardcoded per32 label despite16 captured steps; no old result is reused as a new profile. Here counts verify32 top1/all-gather calls and1152attention/2304reductions per rank. Complete reports/SQL/JSON/validation are backed up/extracted atD:/CantorAI/work/vast-54543362/batch128-gqa-context-profiles-82c3770.tgz and profiles/garnet-gqa4-b128-{early,late}.*.

### OPT-0028: Compact vocabulary candidates for exact TP2 greedy sampling

**Date / Commit:** 2026-10-07 local candidate on feature/gpt-oss-120b. Source: GPT-OSS plugin/lowering/xModel, pipeline planner and generic optional TensorParallel pair merge; Qwen code untouched.

**System / Workload / Baseline:** Blackwell batch128/input256/output512/context1024. Prior GQA4 unprofiled median4580.18decode,3745.44complete execution; vLLM5394.05/5379.66. Early/late traces show~1.59ms/step full-logit all-gather plus~.28ms full top1. Candidate runtime/target model performance unmeasured.

**Hypothesis / Implementation:** Greedy needs only the best rounded score and global token ID on each rank. Opt-in GARNET_GPT_OSS_COMPACT_VOCAB_GREEDY=1 adds gpt_oss_vocab_top1 (new kind7), reuses the existing TP all-gather for two FP32 words/rank/row, then merges the two pairs through the existing tensor_to_cpu bridge with ties to lower global ID. At128 rows, collective input falls51,478,528 to1024bytes/rank. No logits/weight precision change or approximate math. IDs remain exactly representable up to2^24; equal vocabulary shards required. Full-logit model stays the default; compact prefill requires a last-token row per request. Compile/cache identity and operator requirements distinguish compact output. Existing kind0-6 options serialization/ABI stay unchanged; manifest minor version0.9.0 advertises the new operator. CPU transfer/merge overhead may offset some benefit and is included in decode wall time.

**Correctness:** Local CUDA12/RTX4080 full native parity passes, including128-row vocabularies7/65/100544 with cross-rank ties, all NaN, all negative infinity, positive infinity and signed zero versus full-vocabulary greedy semantics. Cache-identity/admission test passes after restoring the test's intentionally reduced VRAM budget; initial test attempt correctly rejected that leftover budget rather than indicating a planner defect. Python syntax checks pass. Real TP2 compiled full/compact exact ID/value tests and target full model still pending. No pretrained speed/equivalence claim yet.

**Decision / Next Step:** INCONCLUSIVE. Push locally, build on idle target, run native/compiled TP2 exact gates and complete fixture generation cold/warm. If those pass, compare matched full-logit/compact pretrained outputs and speed sequentially, then all four prompts and length/batch profiles. Do not infer vLLM superiority from removed bytes alone.
#### OPT-0028 first target native/compiled gate, 2026-10-07

Native09bdda0 builds on SM120, and full native parity passes, including the compact greedy special-value cases. Real TP2 compiled full/compact sampling passes exact IDs/scores atbatch3/local width7 on both repeats. The batch128/local width100544 baseline fixture then fails during TensorRT construction: its default64MiB workspace cannot hold102,957,056bytes of full all-gather scratch. This is a test-harness budget failure before inference; production TP2 already sets256MiB (or larger prefill) workspace. Retain the failed log/cache and explicitly set128MiB/optimization1 for both full/compact fixture engines. Re-run in a new cache/log; no candidate performance claim until the large shape and full-generation gates pass.
#### OPT-0028 compiled/generation gates and independent-oracle mismatch, 2026-10-07

At aa28f92/native09bdda0, real TP2 compiled full/compact exact IDs/scores and signed-zero semantics pass twice for3x7 and128x100544 local vocabulary. Full verify.py --multi-gpu passes, including Qwen compatibility and cold/warm compiled/sharded/vertical-pipeline generation. Separate full-model TP2 free-generation gives [25,27,11] for full-cold,compact-cold,full-warm,compact-warm: the candidate preserves all three tokens exactly. However the independent unsharded CPU fixture expects [25,60,16], so the parent assertion failed. Preserve the failed assertion and all four JSONs/logs; do not call this independent-oracle gate passed. The mismatch is shared by both TP modes and therefore is not introduced by compact sampling, but its cause is not established. Add teacher-forced CPU-prefix logits/margins to the fixture and a TP diagnostic at the existing .025*(1+abs(reference)) compiled tolerance, without changing expected.generated or weakening the exact assertion. Establish whether this is bounded BF16 partition-rounding near ties or a TP defect before advancing correctness claims. Pretrained compact speed is still unmeasured.