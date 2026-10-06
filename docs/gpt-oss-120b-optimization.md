# GPT-OSS-120B optimization log

This log records measurements, decisions, and tradeoffs for GPT-OSS-120B tensor-parallel inference in Garnet. The target is to exceed a current, optimized vLLM run on the same rented two-GPU machine without changing answers. A standalone kernel improvement is only a candidate until an end-to-end model run confirms it.

## Test method and hardware

- Machine: 2 × RTX PRO 6000 Blackwell Max-Q (96 GB each), connected through PCIe/PHB, with 300 GB instance disk. This is tensor parallelism (TP2): each GPU holds part of the model, and both participate in each token.
- Run one server/benchmark at a time. Run vLLM first, then Garnet, with the same saved request IDs, prompt text, output caps, decoding settings, model weights, and BF16 KV cache. Record correctness for arithmetic, code tracing, instruction following, and long context.
- Separate cold first-request time (which can include one-time MXFP4 packing) from repeat prefill and warm decode. Report wall-clock decode including host updates and both ranks; a forward-only rate omits meaningful overhead. Keep raw logs and profiler traces in `D:/CantorAI/work/gpt-oss-benchmark-evidence-2026-10-05/` (outside Git).
- vLLM reference: 0.31.0, TP2, BF16 KV, FlashInfer attention, Marlin experts, and its automatic all-reduce selection. A different vLLM configuration would require a fresh baseline.

## Current end-to-end results

| Case | Optimized vLLM warm decode | Garnet with rank-local inputs, warm wall decode | Status |
|---|---:|---:|---|
| Arithmetic | ~248–249 tok/s | ~200 tok/s | Correct final answer |
| Code tracing | ~248–249 tok/s | ~199 tok/s | Correct final answer |
| Instruction following | ~248–249 tok/s | ~199 tok/s | Correct final answer |
| Long context | ~248–249 tok/s | ~173 tok/s | Correct final answer |

The best local option remains roughly 20% behind vLLM on short requests and 30% behind on the long request. The goal is **not yet met**. Some full runs produce different output token counts despite passing the final-answer checks, so per-token timings and correctness are recorded separately.

vLLM time to first token was about 0.029–0.030 s for short prompts and 0.116 s for long context. Garnet's repeated prefill was about 0.050 s and 0.446 s with the 32-row Marlin prefill option; the cold first pass was about 0.215 s and 0.628 s because it also packed weights. Garnet's prefill measurement and vLLM's time to first token do not cover exactly identical work, so this is a directional comparison rather than a claimed latency win.

## Decisions and tradeoffs

| Change or trial | Measurement and correctness | Decision / tradeoff |
|---|---|---|
| Persistent rank-local decode inputs | Alternating arithmetic A/B runs: old path **189.94, 190.91, 188.91 tok/s**; rank-local **199.59, 200.51, 199.82 tok/s**. All four final-answer checks passed. | Keep as opt-in `GARNET_TP_RANK_LOCAL_INPUTS=1` while validating on more workloads. It removes per-token GPU-1 input copies and allocations; it does not fix kernel or collective time. |
| Marlin prefill tile, 8 to 32 rows | Repeat short prefill ~0.055 to ~0.050 s; long ~0.505 to ~0.446 s. All four answers correct after adjusting shape-compatible column tiles. | Keep opt-in (`GARNET_GPT_OSS_MARLIN_PREFILL_BLOCK=32`, `GARNET_GPT_OSS_MARLIN_MAX_TOKENS=4096`). The first 256-column variant corrupted token 0, so output checking is mandatory. |
| Marlin 64-row prefill tile | Correct first token, but long prefill remained ~0.446 s. | Reverted: no measured gain over 32 rows. |
| TensorRT optimization level 3 and larger workspace | About 6.3% and 8.6% slower, respectively, in the measured trials. | Rejected; builder settings need measurement on this shape and hardware. |
| NCCL protocol overrides | `LL128` ~177.7 tok/s, `Simple` ~175.3 tok/s, versus ~200 tok/s with NCCL-selected protocol; `Tree` failed TensorRT `enqueueV3`. | Retain NCCL's selection. A protocol name alone does not predict end-to-end performance. |
| Direct two-GPU reduction, first scalar prototype | ~14.3 µs median kernel versus ~5.55 µs NCCL kernel in the same profiler trial. | Reworked; a custom collective is not inherently faster. |
| Direct two-GPU reduction, 2-CTA vectorized prototype | **NCCL 7.92 µs/op → direct 4.85 µs/op** for 1,000 standalone operations on this machine, with matching numerical output. Profiler medians over 200 operations: ~5.6 → ~4.45 µs/kernel. | Promising **microbenchmark only**. The direct path uses preallocated input and peer buffers. Garnet integration could need a staging copy and synchronization, which may erase the ~3.07 µs wall-clock gain. Test graph capture and the complete model before adopting it. |
| Direct reduction with a device-to-device staging copy | Two 1,000-operation standalone runs: NCCL **7.96 / 7.89 µs/op**; direct plus copy **6.24 / 6.23 µs/op**. Outputs matched. | The copy consumed much of the original gain, but left ~1.7 µs/op in this microbenchmark. Dynamic model inputs and full-request effects remain unmeasured. |
| Graph-captured direct reduction plus staging copy | One reduction per replay: NCCL **9.09, 8.47, 8.22 µs/op** versus direct **8.29, 8.22, 8.20 µs/op** in three trials, almost a wash. With 48 reductions captured in each graph and 100 replays: NCCL **7.36 / 7.34 µs/op** versus direct **5.76 / 5.73 µs/op** in two trials; outputs matched. | Host graph-launch overhead masks small kernels when each graph contains only one operation. The 48-operation graph is synthetic and repeats fixed inputs; Garnet interleaves layers and changes activations. The model-level result remains unknown. |

The direct reduction uses system-scope release/acquire signals, peer-access memory, and 128-bit vector reads. It is a hardware-specific candidate for small TP all-reduces, not a general NCCL replacement. Its prototype lives in `D:/CantorAI/work/custom-allreduce-proto.cu`; it has not been added to Garnet's production path.

## Where time goes

A steady decode trace showed roughly 4.58 ms of GPU graph work per token plus ~0.66 ms between host graph launches. Per-GPU kernel groups were approximately TensorRT BF16 GEMMs 1.19 ms, Marlin experts 0.96–0.99 ms, NCCL 0.66–0.70 ms, attention 0.36 ms, and remaining operations ~1.07 ms. A faster collective helps only one group; reaching ~4.0 ms/token, the rough vLLM target, also requires reducing graph work and launch gaps. The long-context prefill gap is a separate problem.

Garnet's `xModel/gpt_oss/120b/profiles/tensorrt_mxfp4.json` describes backend/capability choices. The hardware planner in `tools/gpt_oss/pipeline.py` inspects available memory, reserves headroom, sizes KV cache, selects placement, and builds shapes. That is memory/placement fitting; the measured kernel tiles, TensorRT tactics, collective protocol, and graph behavior still need workload-specific profiling. Avoid encoding a single 2-GPU result as a universal default.

## Next checks

1. Integrate the graph-capture-compatible direct collective behind an opt-in switch, retaining NCCL fallback. Verify changing per-token activations, numerical results, no deadlock on both ranks, and complete-request speed; reject it if the model does not improve.
2. Profile and shorten the host graph-launch gap and remaining norm/router/metadata work without changing output.
3. Re-run the four saved cases, then compare against the same vLLM baseline. Keep rejected results in this log so later hardware choices can be revisited with evidence.

Marlin-derived source attribution and Apache-2.0 terms are in the repository `NOTICE` and the GPT-OSS plugin provenance/license files. Qwen code is unchanged by this work.
