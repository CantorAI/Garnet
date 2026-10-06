# GPT-OSS-120B optimization log

This log records measurements, decisions, and tradeoffs for GPT-OSS-120B tensor-parallel inference in Garnet. The target is to exceed a current, optimized vLLM run on the same rented two-GPU machine without changing answers. A standalone kernel improvement is only a candidate until an end-to-end model run confirms it.

## Test method and hardware

- Machine: 2 × RTX PRO 6000 Blackwell Max-Q (96 GB each), connected through PCIe/PHB, with 300 GB instance disk. This is tensor parallelism (TP2): each GPU holds part of the model, and both participate in each token.
- Run one server/benchmark at a time. Run vLLM first, then Garnet, with the same saved request IDs, prompt text, output caps, decoding settings, model weights, and BF16 KV cache. Record correctness for arithmetic, code tracing, instruction following, and long context.
- Separate cold first-request time (which can include one-time MXFP4 packing) from repeat prefill and warm decode. Report wall-clock decode including host updates and both ranks; a forward-only rate omits meaningful overhead. Keep raw logs and profiler traces in `D:/CantorAI/work/gpt-oss-benchmark-evidence-2026-10-05/` (outside Git).
- `tools/gpt_oss/run_tp_prefill.py` repeats the same TP2 request in one loaded process and records the first (potentially cold) pass separately from the subsequent warm passes; use it for prefill A/B runs with all other environment settings held constant.
- vLLM reference: 0.31.0, TP2, BF16 KV, FlashInfer attention, Marlin experts, and its automatic all-reduce selection. A different vLLM configuration would require a fresh baseline.

## Current end-to-end results

| Case | Optimized vLLM warm decode | Garnet with rank-local inputs and direct TP, warm wall decode | Status |
|---|---:|---:|---|
| Arithmetic | ~248–249 tok/s | ~203–204 tok/s | Correct final answer |
| Code tracing | ~248–249 tok/s | ~203 tok/s | Correct final answer |
| Instruction following | ~248–249 tok/s | ~201 tok/s | Correct final answer |
| Long context | ~248–249 tok/s | ~177 tok/s | Correct final answer |

These Garnet numbers use the experimental direct reduction and rank-local inputs. The best local option remains roughly 18–19% behind vLLM on short requests and 29% behind on the long request. The goal is **not yet met**. Some full runs produce different output token counts despite passing the final-answer checks, so per-token timings and correctness are recorded separately.

vLLM time to first token was about 0.029–0.030 s for short prompts and 0.116 s for long context. Garnet's repeated prefill was about 0.050 s for short prompts and 0.446 s for long context with the 32-row Marlin prefill option; the opt-in fast-exponential trial reduced the repeated long prefill to about 0.439 s. The cold first pass was about 0.215 s for short prompts and 0.628 s for long context in the earlier run because it also packed weights. Garnet's prefill measurement and vLLM's time to first token do not cover exactly identical work, so this is a directional comparison rather than a claimed latency win.

## Decisions and tradeoffs

| Change or trial | Measurement and correctness | Decision / tradeoff |
|---|---|---|
| Persistent rank-local decode inputs | Alternating arithmetic A/B runs: old path **189.94, 190.91, 188.91 tok/s**; rank-local **199.59, 200.51, 199.82 tok/s**. All four final-answer checks passed. | Keep as opt-in `GARNET_TP_RANK_LOCAL_INPUTS=1` while validating on more workloads. It removes per-token GPU-1 input copies and allocations; it does not fix kernel or collective time. |
| Marlin prefill tile, 8 to 32 rows | Repeat short prefill ~0.055 to ~0.050 s; long ~0.505 to ~0.446 s. All four answers correct after adjusting shape-compatible column tiles. | Keep opt-in (`GARNET_GPT_OSS_MARLIN_PREFILL_BLOCK=32`, `GARNET_GPT_OSS_MARLIN_MAX_TOKENS=4096`). The first 256-column variant corrupted token 0, so output checking is mandatory. |
| Marlin 64-row prefill tile | Correct first token, but long prefill remained ~0.446 s. | Reverted: no measured gain over 32 rows. |
| Split each prefill query across 2 or 4 warps | On the same 2,005-token prompt, repeated prefill was **0.4464 / 0.4464 s** with the original kernel, **0.4496 / 0.4501 s** with 2 warps, and **0.4478 / 0.4475 s** with 4 warps. All variants produced the same first token (`200005`). | Reverted. Extra warp parallelism did not reduce the total work enough to beat the original one-warp-per-query kernel. The next prefill path needs tiling/data reuse or a tested fused-attention implementation. Raw trial logs are under `D:/CantorAI/work/gpt-oss-benchmark-evidence-2026-10-05/tp2-prefill-warp-df6bf28/`. |
| CUDA fast exponential in prefill softmax | On the same 2,005-token prompt and build, repeated baseline prefill **0.4463 / 0.4467 s** versus `__expf` **0.4390 / 0.4389 s**. The first token matched (`200005`), the CUDA parity suite passed with the option enabled, and all four full final-answer checks passed. | Keep opt-in as `GARNET_GPT_OSS_PREFILL_FAST_EXP=1`. This is only ~1.7% faster for long prefill. Full token sequences and output lengths can differ, despite matching expected final JSON, so broader numerical validation is needed before making approximate exponentials the default. Raw results are under `D:/CantorAI/work/gpt-oss-benchmark-evidence-2026-10-05/tp2-prefill-fastexp-19c13e1/` and `tp2-fastexp-19c13e1/`. |
| Eight-query tiled prefill attention, 16 staged KV positions | Local RTX 4080 kernel-only screening at 2,005 tokens and 32 query heads: full attention **20.50 → 10.80 ms/call**; 128-token sliding window **2.29 → 1.13 ms/call** (five timed calls after two warmups). The opt-in kernel compiled under WSL CUDA 12.0 and the CUDA parity suite passed, including a 40-token paged full/sliding test. | Candidate behind `GARNET_GPT_OSS_PREFILL_TILED_64=1` for head dimension 64. It reuses each shared-memory KV tile across eight adjacent queries; all warps keep their own online softmax and learned sink. This is a different GPU and an isolated kernel timing, so target Blackwell TP2 prefill and full-answer tests are still required before adopting it. The local benchmark harness is `D:/CantorAI/work/gpt-oss-prefill-tile-local-bench.cu`. |
| Sixteen-query tiled prefill variant | On the same local RTX 4080 synthetic shape, full attention **10.03 ms/call** versus **10.80 ms/call** with eight queries, while the 128-token sliding case regressed **1.13 → 1.23 ms/call** in the first screening. Both paged parity cases passed. A second 2,005-token run with 4,106,240 output elements per window found **99.9988% exact matches** to the untiled output for both tile sizes and both full/sliding windows; maximum absolute difference was one BF16 step (**0.00390625**), with none exceeding the parity tolerance. | Expose `GARNET_GPT_OSS_PREFILL_QUERY_TILE=16` only when tiled prefill is enabled; retain eight queries as the provisional default. Pick the tile on the actual hardware using the full model's alternating full/sliding layers, not the best isolated number. |
| TensorRT optimization level 3 and larger workspace | About 6.3% and 8.6% slower, respectively, in the measured trials. | Rejected; builder settings need measurement on this shape and hardware. |
| NCCL protocol overrides | `LL128` ~177.7 tok/s, `Simple` ~175.3 tok/s, versus ~200 tok/s with NCCL-selected protocol; `Tree` failed TensorRT `enqueueV3`. | Retain NCCL's selection. A protocol name alone does not predict end-to-end performance. |
| Direct two-GPU reduction, first scalar prototype | ~14.3 µs median kernel versus ~5.55 µs NCCL kernel in the same profiler trial. | Reworked; a custom collective is not inherently faster. |
| Direct two-GPU reduction, 2-CTA vectorized prototype | **NCCL 7.92 µs/op → direct 4.85 µs/op** for 1,000 standalone operations on this machine, with matching numerical output. Profiler medians over 200 operations: ~5.6 → ~4.45 µs/kernel. | This is a kernel-only result with preallocated input and peer buffers. The later staging-copy and model trials below give the relevant serving result. |
| Direct reduction with a device-to-device staging copy | Two 1,000-operation standalone runs: NCCL **7.96 / 7.89 µs/op**; direct plus copy **6.24 / 6.23 µs/op**. Outputs matched. | The copy consumed much of the original gain, but left ~1.7 µs/op in this microbenchmark. The later model trial below shows a smaller end-to-end gain. |
| Graph-captured direct reduction plus staging copy | One reduction per replay: NCCL **9.09, 8.47, 8.22 µs/op** versus direct **8.29, 8.22, 8.20 µs/op** in three trials, almost a wash. With 48 reductions captured in each graph and 100 replays: NCCL **7.36 / 7.34 µs/op** versus direct **5.76 / 5.73 µs/op** in two trials; outputs matched. | Host graph-launch overhead masks small kernels when each graph contains only one operation. The 48-operation graph is synthetic and repeats fixed inputs; the model trial below is the deciding measurement. |
| Direct reduction inside Garnet | Same build, rank-local inputs, paired sequential runs. Arithmetic NCCL **198.18 / 198.29 / 199.69** versus direct **204.22 / 203.72 / 202.95 tok/s** (warm wall). Code **198.52 → 202.62**, instruction **198.82 → 201.33**, long context **173.95 → 176.69 tok/s**. All four final answers passed. | Keep behind `GARNET_GPT_OSS_DIRECT_ALLREDUCE=1`. This is a modest 1–3% end-to-end gain, far smaller than the synthetic kernel gain. The shared staging area is only suitable for the current single-request, batch-1 decode trial; concurrent-request safety is not established. NCCL remains the default. |

The direct reduction uses system-scope release/acquire signals, peer-access memory, and 128-bit vector reads. It is a hardware-specific candidate for small TP all-reduces, not a general NCCL replacement. The prototype lives in `D:/CantorAI/work/custom-allreduce-proto.cu`; the opt-in implementation is `plugins/gpt_oss/cuda/tp_direct.cu`. The measured model outputs and logs are in `D:/CantorAI/work/gpt-oss-benchmark-evidence-2026-10-05/tp2-direct-5a7a78d/` locally and `/workspace/CantorAI/work/tp2-direct-5a7a78d/` on the rented machine.

## Where time goes

A steady decode trace showed roughly 4.58 ms of GPU graph work per token plus ~0.66 ms between host graph launches. Per-GPU kernel groups were approximately TensorRT BF16 GEMMs 1.19 ms, Marlin experts 0.96–0.99 ms, NCCL 0.66–0.70 ms, attention 0.36 ms, and remaining operations ~1.07 ms. A faster collective helps only one group; reaching ~4.0 ms/token, the rough vLLM target, also requires reducing graph work and launch gaps. The long-context prefill gap is a separate problem.

A CUDA graph trace of the warmed 2,005-token prefill measured 0.465 s wall time under the profiler. After separating the second pass from the first pass's one-time repacking, the main per-GPU kernel totals were:

| Warm prefill group | GPU 0 | GPU 1 | Interpretation |
|---|---:|---:|---|
| Attention | 239.9 ms | 237.4 ms | Largest cost; current prefill kernel walks prior keys serially per query warp. |
| NCCL all-reduce | 90.9 ms | 96.7 ms | Large hidden-state messages across PCIe, twice per layer. |
| Marlin experts | 59.2 ms | 56.2 ms | MXFP4 expert matrix products. |
| Router scores | 31.4 ms | 30.9 ms | Routing matrix/vector work. |
| Expert metadata | 16.4 ms | 16.6 ms | Sorting and padding expert assignments. |

These are kernel-duration sums on each GPU, not percentages of wall time; some work overlaps and profiler overhead affects latency. The complete trace and SQLite export are saved under `D:/CantorAI/work/gpt-oss-benchmark-evidence-2026-10-05/tp2-prefill-warm-profile/`. The attention loop in `plugins/gpt_oss/cuda/gpt_oss_kernels.cu` is the first prefill redesign target. Reducing that cost alone will not erase the collective and expert costs, so a vLLM-beating prefill still needs an end-to-end rerun.

Garnet's `xModel/gpt_oss/120b/profiles/tensorrt_mxfp4.json` describes backend/capability choices. The hardware planner in `tools/gpt_oss/pipeline.py` inspects available memory, reserves headroom, sizes KV cache, selects placement, and builds shapes. That is memory/placement fitting; the measured kernel tiles, TensorRT tactics, collective protocol, and graph behavior still need workload-specific profiling. Avoid encoding a single 2-GPU result as a universal default.

## Next checks

1. Keep the direct collective experimental until changing per-token activations, repeated requests, and concurrent-request safety are validated. The four saved requests passed and showed a small speed gain, but the shared scratch design is not ready as a general serving default.
2. Measure the opt-in tiled prefill path on the rented Blackwell TP2 machine with the 2,005-token prompt and all four final-answer checks; compare repeated prefill and warm decode with the saved baseline. Keep it opt-in if the full model does not improve or token correctness changes.
3. Profile and shorten the host graph-launch gap and remaining norm/router/metadata work without changing output.
4. Re-run the four saved cases, then compare against the same vLLM baseline. Keep rejected results in this log so later hardware choices can be revisited with evidence.

Marlin-derived source attribution and Apache-2.0 terms are in the repository `NOTICE` and the GPT-OSS plugin provenance/license files. Qwen code is unchanged by this work.
