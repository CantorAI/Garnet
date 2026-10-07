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
