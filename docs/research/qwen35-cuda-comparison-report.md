# Qwen3.5 CUDA development report — in progress

Generated from saved artifacts at 2026-10-23T19:58:25.379355+00:00.

**Incomplete: Garnet has not achieved the 2× latency target. Broad correctness, isolated peak/residency coverage, fresh local-source Metal build preservation, and the attention prototype remain open.**

The original screenshots came from another machine. They are context, not a quantitative baseline for this RTX4080 GPU0 run.

## Protocol and precision

- Qwen3.5-2B and 4B pinned BF16 source weights; the shared embedding/head is counted once. Original Garnet cohorts exclude unused MTP weights and have speculation OFF. Experimental Garnet MTP cohorts include their MTP engines/buffers.
- Correction (October23): saved LM Studio2B batches1/2/4/8 actually use MTP, with28/56/112/224 draft-acceptance events. The requested false flag did not establish execution. Saved LM4B logs have no drafting events, so their speculation status remains unresolved. Saved vLLM configuration disables speculation. Exact proposal settings are not matched. See [saved-comparator-speculation-audit.json](../../../work/qwen35-cuda/saved-comparator-speculation-audit.json). Earlier contrary descriptions are superseded; raw timing logs are unchanged.
- Garnet uses BF16 projection input/output with FP32 accumulation and reduced-precision reductions disabled; residual, norm, RoPE, attention and recurrent state are FP32. Other engines have different intermediate arithmetic. Equal BF16 source weights do not imply bit-exact execution.
- Current fixed workload: 320×192 real-image variants, 60 merged visual tokens, 85 prompt tokens, 32 output-token limit, reasoning off, greedy sampling. A unique8×8 corner tile is added to each invocation; source/JPEG/decoded-RGB hashes are saved. Four related image families are not a broad quality dataset.
- Seven rounds of four groups: two warmup rounds, five measured rounds. One model/engine/profile at a time. Batch groups contain B simultaneous requests inside that engine.
- Current partitioned LM cohorts verify B slots,512-token slot capacity and `kv_unified=false`. Earlier unified-pool LM cohorts are retained as provisional/historical. Their slot ceiling alone did not prove independent per-row capacity.
- Garnet and vLLM image hashes and prompt IDs must match. LM raw prompt IDs are unavailable; input counts and full prompt-evaluation logs are checked. Unique images and full prompt evaluations evidence actual work, not a verified setting disabling every internal cache.
- Equal-output groups below contain32 sampled tokens per row in every engine. Group completion latency is not aggregate throughput or individual streaming latency. API timing boundaries differ.

Protocol: [inputs/matched512-batches/manifest.json](../../../work/qwen35-cuda/inputs/matched512-batches/manifest.json). Configuration sources: [matched profile](../../test2026/cuda/qwen35_matched_benchmark_profiles.json), [development methods](qwen35-cuda-development-log.md).

## Completed capacity-matched comparisons

| Model | Batch | Equal-output groups | Garnet ms | LM Studio ms | vLLM ms | 2× fastest-baseline target ms |
|---|---:|---:|---:|---:|---:|---:|
| 2B | 1 | 16 | 553.3 | 298.9 | 286.8 | 143.4 |
| 2B | 2 | 14 | 618.0 | 393.7 | 339.5 | 169.8 |
| 2B | 4 | 8 | 698.8 | 611.7 | 349.0 | 174.5 |
| 2B | 8 | 12 | 1014.6 | 1099.9 | 445.2 | 222.6 |
| 4B | 1 | 19 | 978.1 | 672.3 | 535.8 | 267.9 |
| 4B | 2 | 19 | 1224.2 | 776.2 | 626.5 | 313.3 |
| 4B | 4 | 16 | 1531.7 | 1113.6 | 565.1 | 282.5 |
| 4B | 8 | 15 | 2223.7 | 1351.6 | 669.5 | 334.8 |

Evidence: [matched512-2b-b1-partitioned-summary.json](../../../work/qwen35-cuda/matched512-2b-b1-partitioned-summary.json), [matched512-2b-b2-partitioned-summary.json](../../../work/qwen35-cuda/matched512-2b-b2-partitioned-summary.json), [matched512-2b-b4-partitioned-summary.json](../../../work/qwen35-cuda/matched512-2b-b4-partitioned-summary.json), [matched512-2b-b8-partitioned-summary.json](../../../work/qwen35-cuda/matched512-2b-b8-partitioned-summary.json), [matched512-4b-b1-partitioned-summary.json](../../../work/qwen35-cuda/matched512-4b-b1-partitioned-summary.json), [matched512-4b-b2-partitioned-summary.json](../../../work/qwen35-cuda/matched512-4b-b2-partitioned-summary.json), [matched512-4b-b4-partitioned-summary.json](../../../work/qwen35-cuda/matched512-4b-b4-partitioned-summary.json), [matched512-4b-b8-partitioned-summary.json](../../../work/qwen35-cuda/matched512-4b-b8-partitioned-summary.json).

## Aggregate measured sampled throughput

Total sampled tokens divided by total group wall time across20 measured groups, including unequal stop lengths. This metric does not establish a per-request latency speedup.

| Model | Batch | Garnet tok/s | LM Studio tok/s | vLLM tok/s |
|---|---:|---:|---:|---:|
| 2B | 1 | 57.3 | 102.2 | 110.3 |
| 2B | 2 | 101.2 | 161.7 | 183.8 |
| 2B | 4 | 181.6 | 215.2 | 355.8 |
| 2B | 8 | 251.7 | 229.0 | 595.4 |
| 4B | 1 | 32.4 | 47.4 | 59.7 |
| 4B | 2 | 52.2 | 82.5 | 103.0 |
| 4B | 4 | 84.1 | 114.6 | 224.9 |
| 4B | 8 | 114.7 | 187.1 | 380.3 |

## Experimental direct-KV attention candidate

Opt-in `fusedAttention` is not accepted or enabled by default. Baseline files above remain unchanged. These comparisons reuse saved competitor cohorts and select four-way equal32-token groups.

| Model | Batch | Candidate | Groups | Prior Garnet ms | Candidate ms | Own latency reduction | Same sequences /28 |
|---|---:|---|---:|---:|---:|---:|---:|
| 2B | 1 | fused-attention | 16 | 553.3 | 456.7 | 17.4% | 22/28 |
| 2B | 1 | fused-attention-workspace16k | 16 | 553.3 | 442.9 | 20.0% | 22/28 |
| 2B | 1 | fused-attention-chunk16 | 16 | 553.3 | 353.2 | 36.2% | 19/28 |
| 4B | 8 | fused-attention | 14 | 2223.2 | 1357.1 | 39.0% | 13/28 |

The strict batch4 decoder fixture passed; batch8 still fails a padded-step hidden tolerance. Six changed2B cases reproduced, with192 independent teacher-forced decisions:185 deterministic top1 matches, 189 reference-maximum-set choices and192 reconstructed-head matches. Three non-tied differences remain. The oracle consumes native image embeddings; this is not independent vision or broad quality acceptance.

One instrumented2B request has GPU activity span466.237ms, union423.850ms and kernel-duration sum417.377ms. BF16 projection families and the head account for about83.2% of summed kernel time. Actual GPU copy time is6.239ms; blocking API durations must not be interpreted as transfer time. These are profiling diagnostics, not matched latency or a projected speedup. See [profile-fused-attention-2b-b1-summary.json](../../../work/qwen35-cuda/profile-fused-attention-2b-b1-summary.json).

## Batch1 phase diagnostics

These are engine-specific diagnostics, not the primary latency ranking. Garnet TTFT includes reset/JPEG/vision/prefill; vLLM request metrics start after caller image decode; LM reports a server TTFT. Garnet decode uses post-prefill steps/time, vLLM uses first-to-last token timestamps, and LM supplies its own rate.

| Model | Engine / scope | Median TTFT ms | Median decode tok/s |
|---|---|---:|---:|
| 2B | Garnet / Model API | 211.3 | 90.9 |
| 2B | vLLM / scheduler request | 43.3 | 129.7 |
| 2B | LM Studio / server-reported | 60.6 | 142.3 |
| 4B | Garnet / Model API | 390.4 | 52.5 |
| 4B | vLLM / scheduler request | 45.1 | 63.8 |
| 4B | LM Studio / server-reported | 95.4 | 57.7 |

## Memory evidence and scope

The excessive-memory diagnosis led to one BF16 text-weight set per layer serving prefill/decode, one shared BF16 embedding/head, GPU-resident state, and explicit allocator tracking. This applies to the new Qwen3.5 CUDA path; the legacy Qwen3-VL path is not thereby proven memory-optimized.

| Model | Batch | Stable retained ownership GB (decimal) |
|---|---:|---:|
| 2B | 1 | 4.642 |
| 2B | 2 | 4.838 |
| 2B | 4 | 5.231 |
| 2B | 8 | 6.016 |
| 4B | 1 | 9.613 |
| 4B | 2 | 10.125 |
| 4B | 4 | 11.150 |
| 4B | 8 | 13.199 |

Retained ownership excludes some driver/external-library/graph-executable/transient allocations. Engine live counters include weights and runtime buffers, not just workspace. Per-engine peaks are never added as a process peak. These values are not isolated process VRAM or proof of physical residency.

Historical batch1 WDDM process-counter diagnostics were close to LM Studio, but used Garnet384 versus LM512 context; they do not substitute for current-profile isolated peak/residency measurements. WSL host VM counters had ambiguous duplicate instances; vLLM worker allocator counters were reported separately. See [ISOLATED_MEMORY.md](../../../work/qwen35-cuda/ISOLATED_MEMORY.md).

## Correctness and regression evidence

- Component/state/masking/CUDA-graph tests cover selected independent fixtures. The real-model decoder oracle uses native image embeddings, so it bypasses independent vision validation.
- Changed output trajectories after batched full-attention projections reproduced;2B and4B each matched64/64 teacher-forced decoder decisions. This is narrow trajectory evidence.
- The4B batch8 candidate decoder reference had750/768 exact IDs and754/768 maximum-set choices; fourteen non-tied rank2 differences remain unresolved. A synthetic batch8 hidden check fails on both archived and current builds; layer tracing localizes its first error after an exactly matching recurrent layer.
- Legacy Qwen3-VL-2B current-DLL smoke passes two16-token requests with exact text/count replay. Its preprocessing produces308patches/77visual tokens for this input, not Qwen3.5’s60. Raw IDs are not exposed by that legacy API. See [qwen3-vl-matched-grid-regression.json](../../../work/qwen35-cuda/qwen3-vl-matched-grid-regression.json).
- Metal existing-binary pretrained layer0/3 reference tests passed batches1/2 and16 stateful steps. Original/current4B Python factories produced identical8-token outputs twice on one real image. Remote native Metal sources differ from the Windows base; fresh local-source native build preservation and broad quality remain open. See [metal-factory-preservation-summary.json](../../../work/qwen35-cuda/metal-factory-preservation-summary.json).

## Failed experiments retained

- TensorRT conditional branch around a plugin was rejected by the builder.
- 4Bbatch8 attention builds exceeded64MiB and128MiB workspace caps; a256MiB layer probe and subsequent384-context full profile passed. This is not proof every512-context profile fits.
- Original unified LMbatch8 load passed slot-ceiling checks but failed KV allocation on the first group. Partitioned non-unified512 slots completed. Prior shared-pool measurements remain historical/provisional.
- The first legacy regression used an incompatible240-patch compiled profile; the retry derives its308-patch profile from native preprocessing.
- Harness startup/schema failures and native precision discrepancies remain in raw logs; successful retries do not erase them.

## Remaining acceptance work

- All eight baseline profiles completed; preserve these artifacts and compare accepted optimizations on the same workload. Capacity and full prompt work are audited; a setting disabling every internal cache remains unverified.
- Resolve or rigorously bound numerical discrepancies and add independent vision/held-out quality coverage, admission/cache-boundary tests at official-model scope.
- Resolve numerical and broader pretrained-model gates for the opt-in attention candidate. TensorRT/component passing tests and selected speed measurements do not establish full model acceptance.
- Repeat relevant isolated steady/peak/residency memory diagnostics on the accepted implementation and larger profiles; do not use allocator ownership as peak VRAM.
- Verify fresh native Metal source/build preservation beyond same-binary factory compatibility.
- Achieve the requested2× full-request latency advantage over the fastest matched baseline with acceptable memory/correctness, then audit every requirement before marking the goal complete.

## Attention-core component status

Standalone7-case parity, exact KV/input/guard/graph replay checks and memcheck/racecheck/synccheck pass. The original reduction raced despite matching outputs; its failure logs/binary are retained. Opt-in serving integration now has selected measurements above; component checks alone do not establish quality or promotion.

## Attention candidate: isolated4B batch8 memory

Context512/chunk4, same attention binary as the candidate cohort. Timings excluded. PCI/CUDA-LUID/node resolved WDDM process counters; no ambiguous nonzero duplicate instances.

| Phase | Enclosed samples | Dedicated GB median | Shared GB median |
|---|---:|---:|---:|
| steady_before | 4 | 11.704 | 0.082 |
| steady_after | 4 | 11.704 | 0.082 |
| after_model_close | 5 | 0.272 | 0.082 |

Largest observed dedicated counter 11.716GB; sampled maximum is not a true lifetime peak or physical-residency proof. There is no matched large-batch comparator memory cohort yet. See [memory-attention-4b-b8-512-summary.json](../../../work/qwen35-cuda/memory-attention-4b-b8-512-summary.json).

## MTP feasibility — separate from baseline acceptance

Experimental native MTP is now connected behind a default-OFF profile (depth1/3/7). On one traced2B trajectory, independent and native implementations propose the same31 tokens, with24/31 matching teacher next tokens. Target-head reconstruction is32/32. This consumes saved native target rows; it is not end-to-end inference or a latency claim. Three synthetic MTP profiles still fail the unchanged strict numerical bound; input/unwritten-cache preservation, graph replay and borrowed-head lifetime checks completed separately. A reciprocal-normalization hypothesis did not fix the failure and was reverted.

Conditional native drafting from teacher-correct prefixes suggests18/12/11 hypothetical target passes at depth1/3/7 on this single32-token trajectory. This omits actual verification/commit/corrected-draft work and does not establish rollout acceptance or speedup. The subsequent verifier/session implementation passed rollback/prefix/capacity/stop/reset tests: six synthetic profiles with24cases each, plus real2B/4B fresh-image probes with exact32tokens and all final target state bytes equal to native serial controls. This is native target equivalence, not broad independent-model quality acceptance.

The full2B B1 C16 depth3 MTP cohort completed28requests, all28sequences matching the C16 control. Across16four-way equal32groups, latency increased353.2359→384.22405ms (+8.77%); retained ownership increased4,578,699,912→4,745,831,440B (+167,131,528). Median20target forwards include12verification and8prefix-recomputation passes. This speed candidate is rejected and stays default off. Saved vLLM disables speculation; LM Studio runtime drafting is audited above. Algorithm settings differ. See [matched512-2b-b1-verified-mtp-d3-candidate.json](../../../work/qwen35-cuda/matched512-2b-b1-verified-mtp-d3-candidate.json).

Indexed draft heads and recurrent prefix replay are measured separately below. Every emitted proposal remains target-verified; no overall goal or quality acceptance is implied.

## Indexed MTP head — measured experiment

The proposal head reads selected rows directly from the existing tied BF16 table. A GPU stable sort selects8192 candidates from the previous target distribution. Every emitted proposal is checked by the full target head; no second embedding table or persistent gathered weight matrix is stored. Fixed independent FP64 dot bounds, CPU stable sorting, guards/invalid/alias/nonfinite/graph tests and CUDA memory/race/synchronization checks passed. Native API6CPU-oracle cases and seven synthetic session profiles passed, plus fresh-image2B/4B probes with exact serial target outputs and all final target-state bytes. This preserves native target behavior on tested cases; the pre-existing independent-model quality gaps remain.

The28-request2B B1 C16 D3 cohort matched all28 control output sequences. Across16common equal32groups, median latency was337.16ms versus353.24ms for non-speculative C16 Garnet, 298.86ms for LM Studio and286.77ms for vLLM. Retained ownership was4.752GB. Speculation differs from the saved OFF comparator cohorts. The2× target is unmet; this option remains default off. See [matched512-2b-b1-verified-mtp-d3-indexed8192-candidate.json](../../../work/qwen35-cuda/matched512-2b-b1-verified-mtp-d3-indexed8192-candidate.json).

![Completed2B experiments](../../../work/qwen35-cuda/progress-speculative-2b-b1.png)

## Recurrent prefix replay — measured experiment

Partial commits retain projected inputs and rerun only the small convolution/delta recurrence from saved state. They restore unaccepted attention KV suffixes and select the accepted target hidden rows. No per-token copies of all recurrent states or second set of large projection weights is stored. Helper small weights and buffers, parent retained payloads and all session allocations are counted. The original restore/recompute path remains available.

Four transactional verifier profiles and seven24-case session profiles passed exact native state/output/slot and stop/limit/rollback/capacity checks. Fresh-image2B/4B probes each passed two MTP requests plus serial controls, matching all32output IDs and every final target-state byte. These tests do not close independent quality gaps.

| Depth | Equal32groups | Garnet ms | LM Studio ms | vLLM ms | Same sequences /28 | Owned GB |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 16 | 273.46 | 298.86 | 286.77 | 28/28 | 4.828 |
| 7 | 16 | 313.81 | 298.86 | 286.77 | 28/28 | 4.830 |

All replay requests report zero full prefix recomputations; recurrent replays are counted separately. Saved vLLM disables speculation; LM Studio runtime drafting is audited separately; algorithms differ. Default OFF,2× target unmet. No isolated process peak/residency comparison is implied by retained ownership.

## Shared projection profiles — measured experiment

Separate bounded prefill/decode graphs borrow one readonly BF16 projection pool and share the same target state/last-hidden tensors. Header-audited target matrices total2,745,434,112B for2B and7,138,181,120B for4B; real-model retained pool counters match these payloads. Both graph sets, small constants, runtime buffers and the MTP session remain included in ownership totals. Synthetic constant-versus-pooled and independent explicit-copy controls pass; this is not process-residency proof.

| Prefill / decode | Equal32groups | Garnet ms | LM Studio ms | vLLM ms | Same sequences as older C16 /28 | Owned GB |
|---|---:|---:|---:|---:|---:|---:|
| 64 /4 | 16 | 255.03 | 298.86 | 286.77 | 21/28 | 4.979 |

Cross-profile precision remains open: a synthetic narrow-versus-wide comparison differs only in key-cache values (max7.15e-7), while an identical-shape control with independent explicit state copies matches byte-for-byte. Real2B and4B probes match their own serial target outputs and final state bytes;4B differs from the older saved caption after token26. That original assertion failure remains retained. Changed full-cohort2B trajectories also need independent analysis. All options stay default OFF; no quality/default/2× acceptance. Saved vLLM disables speculation; LM Studio runtime drafting is audited separately.

## Packed whole-decoder experiment

An explicit default-off profile compiles each target chunk into one graph with a canonical packed state tensor. Wide prefill and narrow decode share one readonly BF16 projection pool and state. FP32 layer boundaries remain materialized. Partial acceptance uses one graph of small delta-core replays and exact accepted-span KV selection; target matrix projections are not rerun. Eight two-layer packed-program profiles passed byte-exact same-shape native controls. Four transactional verifier profiles and seven24-case speculative profiles passed rollback, partial acceptance, EOS, length and capacity checks. Independent pretrained quality, process memory and2× acceptance remain open.

Completed2B B1 cohort: 250.04ms versus same-profile per-layer control 255.03ms, saved LM Studio 298.86ms and vLLM 286.77ms across16equal32groups. 28/28 output sequences match the same-profile control; retained ownership 5.070GB. Saved vLLM disables speculation; LM Studio runtime drafting is audited separately. No quality/default/2× acceptance is inferred.

## Inline request controls and direct chunk embeddings

The default-off residentControls profile transports small scalars through CUDA launch parameters and gathers the shared BF16 table into caller-owned padded chunks. It changes data transport and temporary allocation behavior while preserving model arithmetic. Sixteen independent IO profiles, four parent controls, four transactional verifier profiles and eight24-case speculative profiles passed exact native comparisons; all three native IO sanitizer checks found zero errors.

Completed2B B1 cohort: 239.19ms versus 255.03ms for the same-profile old IO control, across16equal32groups; all28/28sequences match. Retained ownership 5.010GB. Asynchronous prefill grew the MTP context pool from1to8contexts, adding31.1MB. A separate default-off serial graph reuse experiment addresses this growth; its result appears below when available. The2× target remains unmet, independent quality gaps remain, and saved vLLM disables speculation; LM Studio runtime drafting is audited separately.

## Serial MTP graph reuse — measured memory fix

The same16equal32groups measure 239.44ms versus 239.19ms for resident IO with the old pool policy. All28/28output sequences match. Retained ownership returns to 4.979GB, saving31,098,368B with oneMTPcontext. No latency gain is claimed. Pending replay requires a captured graph, identical bindings, static shapes, same stream/thread semantics and no provider owner. Changed cases retain conservative retirement. Gated real-TRT pool tests and all three sanitizers passed; ten24-case MTP profiles preserve exact states/outputs/stops/limits. Default OFF; independent quality, physical memory and2× acceptance remain open.

## Fixed128 recurrent state kernel — measured experiment

The optional kernel keeps the original sequential FP32 recurrence/reductions and convolution workspace, using fixed128-state loops and volatile shared Q/K. Native parent, prefix-commit and MTP controls pass exact bytes; all49pretrained final target-state hashes match the same-profile old kernel. Standalone independent FP64 and all three sanitizers pass. Native compiler resources differ from the standalone probe, so component speeds do not predict full-model gains.

Completed2B B1 cohort: 218.07ms versus 239.44ms, across16equal32groups, all28/28output sequences matching. Retained ownership remains 4.979GB. Saved LM Studio 298.86ms and vLLM 286.77ms have different or unresolved proposal settings. The2× target, independent quality, physical-memory and all-profile acceptance remain open; option stays default OFF.

## Flat packed-state concatenation

The optional variadic concatenation joins layer payloads once instead of retaining a chain of intermediate concatenations. CPU row-order controls include batch8 and129 parts; native parent, verifier and speculative controls preserve exact bytes. The real2B probe preserves canonical-state and last-hidden hashes.

The flat-only cohort measures 241.24 ms versus 250.04 ms for chained packing, with 28/28 sequences matching. Retained ownership is 4.887 GB, a reduction of 183,819,264 bytes. This is retained allocation accounting, not physical peak VRAM.

## Combined packed, resident, serial-reuse and register profiles

These experiments combine several options. Their differences from the original target can include chunk-size arithmetic changes; same-sequence counts compare the saved control indicated by each artifact. Independent quality remains open. The2B batch1 row uses the completed vLLM MTP3 comparator; other vLLM rows use saved serial cohorts. LM Studio runtime drafting is audited above. Each row uses its own common equal-output subset.

| Model | Batch | Equal32 groups | Garnet ms | LM Studio ms | vLLM ms | 2× target ms | Owned GB | Same sequences /28 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2B | 1 | 17 | 199.96 | 302.21 | 203.27 | 101.64 | 4.749 | 28/28 |
| 2B | 2 | 14 | 250.91 | 393.66 | 339.54 | 169.77 | 5.195–5.298 | 20/28 |
| 2B | 4 | 8 | 327.75 | 611.74 | 348.99 | 174.49 | 6.018 | 7/28 |
| 2B | 8 | 11 | 468.50 | 1086.66 | 467.49 | 233.74 | 7.458 | 6/28 |
| 4B | 1 | 19 | 353.84 | 672.34 | 535.77 | 267.88 | 10.246 | 23/28 |
| 4B | 2 | 19 | 444.97 | 776.23 | 626.51 | 313.25 | 11.136 | 17/28 |
| 4B | 4 | 15 | 587.84 | 1112.48 | 563.89 | 281.95 | 12.917 | 17/28 |
| 4B | 8 | 15 | 727.65 | 1348.27 | 665.65 | 332.82 | 12.706 | 28/28 |

Missing rows have no completed cohort. No projected speed or memory values are included. Ownership ranges retain allocation growth instead of selecting the lowest observation. The2× goal remains unmet; these options remain default OFF.

## New vLLM MTP comparator — completed2B batch1

The same pinned BF16 checkpoint, JPEGs, prompt IDs,512-token capacity and32-token limit now run with vLLM native MTP depth3. Cache budgets96MiB and112MiB per sequence both failed admission; their errors remain preserved. 128MiB is the smallest successful tested budget, not a proven mathematical minimum. The four-request admission smoke is separate from the full28-request cohort.

Across 17 common equal32 groups, Garnet measures 211.04 ms, vLLM MTP 203.27 ms and LM Studio 302.21 ms. The current2× threshold is 101.64 ms. Garnet is slower than this MTP comparator. vLLM MTP matches17/28 saved vLLM serial sequences; independent quality remains open. FP32 convolution/recurrent caches remain requested. Runtime selects the Triton GDN speculative path and logs that the fused CUDA path requires BF16 convolution cache. A separately identified native-cache profile may be faster and is still pending. Exact proposal settings and intermediate arithmetic are not equivalent across engines.

Evidence: [matched512-vllm-2b-b1-mtp3-cache128.json](../../../work/qwen35-cuda/matched512-vllm-2b-b1-mtp3-cache128.json), [matched512-2b-b1-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat-mtp3-cache128-candidate.json](../../../work/qwen35-cuda/matched512-2b-b1-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat-mtp3-cache128-candidate.json).

## CUDA capture lifetime gate — unresolved

All18 target-serial native integration profiles preserve states, tokens, stops, limits, rollback and one-context ownership. However, the full batch8 memcheck reports use-before-allocation errors in a TensorRT Myelin fused kernel. A same-profile target-reuse-OFF control also fails. Target-only isolation with no MTP/replay passes when capture is OFF and fails when capture is ON, with and without target serial reuse. Target serial reuse remains default OFF; no latency or memory improvement has been measured for it.

Eight independent plain-CUDA allocation/capture controls pass memcheck with zero errors. Skipping unchanged context binding setters did not fix the TensorRT failure; that diagnostic is rejected as the fix. The failing10,633,216-byte buffer is absent from the synchronous runtime allocator trace. Overriding the distinct TensorRT temporary-storage allocator also did not fix it. No sanitizer error is suppressed or treated as a pass. The current capture gate prevents acceptance.

## Single-stream capture route — validation in progress

The default wide target graph uses two auxiliary streams; decode and final norm use zero. Its captured prefill graph has52 nodes and no allocation/free/child nodes, so the failing buffer was allocated outside that graph. The optional zero-auxiliary-stream build passes all three target-only memcheck controls: capture OFF, capture ON, and capture ON with serial target reuse. All final state hashes, tokens and slots match the older uncaptured control. Only serial target reuse bounds all three graph context counts to one. This is a measured workaround for the isolated route, not a universal TensorRT bug diagnosis or performance claim.

Full batch-profile and MTP sanitizer gates: full MTP memcheck passed; full integration racecheck was interrupted and remains incomplete. Full integration synccheck was not run.

Exact native-DLL two-layer C4/B8 packed-versus-per-layer component checks pass racecheck and synccheck with zero findings and12 exact records each. The Garnet kernel-name filter excludes vendor kernels. This bounded component coverage does not replace the incomplete full24-case MTP race/sync checks.

## Matched4B batch8 physical memory diagnostic

Each engine ran alone with the same six JPEG groups,8 slots and512-token independent capacity. Garnet uses the combined MTP profile; the saved4B GGUF lacks MTP tensors. Proposal configuration differs. Loading is observed only for Garnet. Sampling and idle holds invalidate latency measurement; the primary cohorts above are separate.

| Engine | Warm dedicated GB | Warm shared GB | Largest observed dedicated GB | Retained owned GB |
|---|---:|---:|---:|---:|
| Garnet combined | 15.806 | 0.252 | 15.815 | 15.534 |
| LM Studio | 10.284 | 0.101 | 10.284 | unavailable |

The short Garnet diagnostic kept one target-prefill context. The longer28-request timing cohort allocated a second context and reached16.480GB retained ownership, with several requests above6seconds. Both observations are retained; the short diagnostic does not describe the longer cohort. Shared usage also remained after model close, so its full amount cannot be attributed to live model spill. A causal paging diagnosis remains unproven. LM Studio unloaded its worker before an enclosed close-phase sample was available. Largest sampled usage is not a true lifetime peak. Memory acceptance remains open.

Evidence: [memory-combined-4b-b8-512-summary.json](../../../work/qwen35-cuda/memory-combined-4b-b8-512-summary.json), [memory-lmstudio-4b-b8-512-partitioned-summary.json](../../../work/qwen35-cuda/memory-lmstudio-4b-b8-512-partitioned-summary.json).

## Bounded target contexts with zero auxiliary streams — measured4B batch8

Across15common equal32groups, the new route measures 1129.04ms versus 1763.29ms for the older combined control; all28/28 output sequences match. The longer timing cohort now retains one wide target context, removing945,818,112B of duplicate context allocation. This combines serial target reuse with zero auxiliary streams; the latency change does not isolate either option. The short old memory diagnostic already had one context, so it cannot establish the same physical-memory improvement.

## Deferred canonical state commit — measured4B batch8

The optional transaction borrows canonical state unchanged while target verification computes proposed outputs. Rollback restores small request buffers; accepted prefixes write canonical state only at the final commit. Pending transactions reject other parent mutation. A final-write failure poisons the decoder until reset. This removes the full canonical backup without changing the CUDA recurrence or precision. Default OFF.

All12 integration profiles exited0, including parent ownership, invalid-use, partial/full acceptance and injected failure recovery. The batch8 verifier and full24-case MTP session both passed unfiltered memcheck with zero errors. These controls do not resolve existing independent pretrained quality gaps.

The28-request real-image cohort measures 805.41ms versus 1129.04ms on 15common equal32groups, with all28/28 sequences unchanged. Retained ownership falls from15,534,163,516B to14,837,909,052B (-696,254,464B). vLLM measures665.65ms, LM Studio1348.27ms; the2× threshold332.82ms remains unmet.

Evidence: [matched512-4b-b8-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-candidate.json](../../../work/qwen35-cuda/matched512-4b-b8-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-candidate.json), [deferred-state-verifier-b8-memcheck-exit.json](../../../work/qwen35-cuda/deferred-state-verifier-b8-memcheck-exit.json), [deferred-state-session-b8-memcheck-exit.json](../../../work/qwen35-cuda/deferred-state-session-b8-memcheck-exit.json).

## Latest isolated4B batch8 memory counters

Sequential runs use the same six JPEG groups. Dedicated/shared counters are sampled WDDM process usage; they do not prove lifetime peaks or physical residency. Loading was not observed for LM Studio. MTP configurations differ.

| Profile | Warm dedicated GB | Warm shared GB | Largest observed dedicated GB | Owned GB |
|---|---:|---:|---:|---:|
| Garnet zero-aux / serial target | 15.976 | 0.082 | 15.984 | 15.534 |
| Garnet plus deferred commit | 15.279 | 0.082 | 15.288 | 14.838 |
| Garnet plus native prefix commit | 13.834 | 0.082 | 13.843 | 13.407 |
| Garnet plus shared output storage | 13.161 | 0.082 | 13.170 | 12.706 |
| LM Studio | 10.284 | 0.101 | 10.284 | unavailable |

Evidence: [memory-combined-4b-b8-512-targetserial-aux0-summary.json](../../../work/qwen35-cuda/memory-combined-4b-b8-512-targetserial-aux0-summary.json), [memory-combined-4b-b8-512-targetserial-aux0-deferred-summary.json](../../../work/qwen35-cuda/memory-combined-4b-b8-512-targetserial-aux0-deferred-summary.json), [memory-combined-4b-b8-512-targetserial-aux0-deferred-nativecommit-summary.json](../../../work/qwen35-cuda/memory-combined-4b-b8-512-targetserial-aux0-deferred-nativecommit-summary.json), [memory-combined-4b-b8-512-targetserial-aux0-deferred-nativecommit-arena-summary.json](../../../work/qwen35-cuda/memory-combined-4b-b8-512-targetserial-aux0-deferred-nativecommit-arena-summary.json), [memory-lmstudio-4b-b8-512-partitioned-summary.json](../../../work/qwen35-cuda/memory-lmstudio-4b-b8-512-partitioned-summary.json). Memory acceptance remains open.

## Native state-only prefix commit — experimental

The default-off nativePackedPrefixCommit route replaces the packed replay TensorRT graph with separate CUDA state-only kernels. It recomputes accepted convolution/recurrent state in place and copies only accepted KV slots from authoritative target outputs. The canonical state stays unchanged until this final commit; failed final writes poison the decoder. It requires the deferred-state and register128 profiles. Small convolution/dt/A_log parameters and convolution workspace are explicitly counted. TensorRT input/output aliasing is not used.

All12 numerical profiles pass exact ordinary-native states/outputs, including B1/2/4/8, C4/C16 and P32/P64. The B8 verifier and full24-case MTP session pass unfiltered memcheck. Racecheck/synccheck pass with zero findings for the three new packedCommit kernels inside the full MTP session; other kernels are excluded from these two checks. Invalid spans/layouts are rejected before mutation, and post-write injected failures require reset. These scopes do not replace missing full integration race/sync coverage or existing independent pretrained quality gaps.

The real4B B8 probe passes: two native-prefix requests followed by a separately loaded non-MTP target. Full finite target-state hashes stream bounded column chunks; timings are invalidated by diagnostic hashing. Evidence: [native-prefix-pretrained-4b-b8.json](../../../work/qwen35-cuda/native-prefix-pretrained-4b-b8.json).

The full28-request4B B8 cohort measures 727.84ms versus 805.41ms on 15common equal32groups, with28/28 unchanged sequences. Retained ownership is13,407,056,828B versus14,837,909,052B. vLLM serial665.65ms and LM Studio1348.27ms; target332.82ms remains unmet.

## Shared packed output storage — experimental

A prospective design records one maximum-sized allocation with stable exact-shape SDK storage views for sequential packed prefill/decode outputs. The generic binding API checks static output shape, dtype, device, contiguity, alignment and storage span before accepting it; input/output overlap and late rebinding are rejected. The decoder counts the shared pool once and excludes borrowed output views from each graph's owned-output bytes. The initial test harness failed before engine execution because it omitted the required fusion graph boundary; its source/archive/log/exit1 are retained. The corrected harness preserves the same exact-output and lifecycle assertions.

The generic corrected test passes two interleaved graph shapes, capture/replay, caller-reference drop, closure of other owners and invalid storage boundaries. Model validation and sanitizer results are listed only when their explicit exit records exist. Default OFF. Evidence: [shared-output-arena-design.json](../../../work/qwen35-cuda/shared-output-arena-design.json), [output-arena-storage-fusion.json](../../../work/qwen35-cuda/output-arena-storage-fusion.json).

All12 packed verifier/MTP profiles pass exact states and outputs. Generic storage, B8 verifier and full24-case B8 MTP session pass unfiltered memcheck with zero errors. This covers memory access/lifetime, not full integration race/synccheck or independent quality acceptance.

The real4B B8 shared-output probe passes: two production MTP requests followed by a separately loaded ordinary target with independent outputs. All full finite canonical-state and last-hidden hashes match. Probe retained ownership is12,705,559,484B; this is not process VRAM or cohort timing. Evidence: [output-arena-pretrained-4b-b8.json](../../../work/qwen35-cuda/output-arena-pretrained-4b-b8.json).

Completed4B B8: 727.65ms versus 727.84ms on15common equal32groups; 28/28 unchanged sequences. Owned12,705,559,484B versus13,407,056,828B, a measured reduction of701,497,344B. Retained allocation accounting is separate from the isolated physical-counter table. The2× goal remains unmet.

## Scoped whole-program CUDA state-kernel compilation

The resource audit isolates relocatable device code as a register-count difference: the same native and volatile-template source bodies use168 registers without RDC and255 with it. Those object counts precede final device linking and do not establish model speed. The initial audit mirrored compiler FLAGS but omitted the hard-coded CMake rule addition; the second run adds only-rdc=true and records the correction.

The default-off GARNET_QWEN35_WHOLE_PROGRAM_CUDA option moves only the self-contained delta and packed-prefix CUDA units to a non-separable object target. Other CUDA sources retain their existing linking mode. No algorithm, precision, fast-math, projection or layout change is requested. The actual DLL reports168 registers/zero stack for target and prefix recurrence; a marker compiled in the CUDA unit identifies the real policy. Benchmark/probe labels must match that marker. All12 original verifier/MTP profiles pass exact state/output/lifecycle controls.

Evidence: [qwen35-whole-program-design.json](../../../work/qwen35-cuda/qwen35-whole-program-design.json), [register-template-audit-comparison.json](../../../work/qwen35-cuda/register-template-audit-comparison.json), [whole-program-dll-resources.txt](../../../work/qwen35-cuda/whole-program-dll-resources.txt).

| Model / batch | Groups | Previous ms | Whole-program ms | Same sequences | Owned GB |
|---|---:|---:|---:|---:|---:|
| 2B / 1 | 17 | 199.96 | 227.03 | 28/28 | 4.749 |
| 4B / 8 | 15 | 727.65 | 716.80 | 28/28 | 12.706 |

Missing rows are pending. Independent quality, other batch profiles and 2× acceptance remain open. The 2B batch-1 result is a latency regression despite the lower register count. The option remains experimental and default off; this table reports measured candidates, not accepted improvements.

2B B1 cross-build pretrained finite-state/output preservation: PASS. Identical real inputs/profile; full ordinary and repeated speculative state hashes/signatures compared. Evidence: [whole-program-pretrained-2b-b1-cross-rdc-state-control.json](../../../work/qwen35-cuda/whole-program-pretrained-2b-b1-cross-rdc-state-control.json).

Generic storage, B8 verifier and full24-case MTP unfiltered memchecks pass with zero errors. Exact-DLL C4/B8 two-layer program/control race/sync checks pass with zero findings. New prefix-kernel race/sync checks also pass in the full MTP session. Their kernel-name filters limit scope; full integration race/sync remains incomplete.

## Bounded finite-state audit transfer

A host diagnostic exports at most 1,048,576 contiguous FP32 words (4 MiB) per call, rejects every NaN/Inf and returns exact little-endian bytes for SHA-256. The original column chunk order and full target-state coverage remain unchanged. The inference path does not call this helper, so this is not an inference speed improvement.

Exact finite-byte, signed-zero, subnormal, maximum-finite, legacy SHA, rejection, logical-view and lifetime checks: 10 passed. Unfiltered memcheck exit: 0. CUDA device-fatbin equality with the prior build: True. Device equality does not by itself establish host correctness.

Evidence: [finite-f32-bytes-validation.json](../../../work/qwen35-cuda/finite-f32-bytes-validation.json), [finite-f32-bytes-memcheck-exit.json](../../../work/qwen35-cuda/finite-f32-bytes-memcheck-exit.json), [finite-f32-bytes-device-preservation.json](../../../work/qwen35-cuda/finite-f32-bytes-device-preservation.json).

The earlier 4B batch-8 scalar-conversion probe was interrupted while CPU-active after over 13 minutes. Its logs and incomplete exit are retained. No success, kernel failure or timing result follows from that interruption. Redirected stdout can be buffered, so the absence of a printed phase marker does not localize the delay.

2B B1 bounded-byte real probe versus saved RDC full-state/output hashes: PASS. Evidence: [whole-program-fastbytes-pretrained-2b-b1-cross-rdc-state-control.json](../../../work/qwen35-cuda/whole-program-fastbytes-pretrained-2b-b1-cross-rdc-state-control.json).

4B B8 bounded-byte real probe versus saved RDC full-state/output hashes: PASS. Evidence: [whole-program-fastbytes-pretrained-4b-b8-cross-rdc-state-control.json](../../../work/qwen35-cuda/whole-program-fastbytes-pretrained-4b-b8-cross-rdc-state-control.json).

2B B1 bounded-byte versus saved same-policy legacy full-state/output hashes: PASS. Evidence: [finite-f32-bytes-2b1-cross-method-control.json](../../../work/qwen35-cuda/finite-f32-bytes-2b1-cross-method-control.json).

## All tested 4B batch-8 comparators on one subset

Every following value uses the same 14 equal-32 groups, including serial and MTP vLLM. This is distinct from the 15-group four-way cohort table.

| Profile | Median group ms |
|---|---:|
| garnet | 731.36 |
| control | 741.09 |
| lmstudio | 1347.87 |
| vllm_serial | 667.58 |
| vllm-mtp3-cache128-convbfloat16 | 850.56 |

The fastest tested baseline is vllm_serial; the half-latency target of 333.79 ms remains unmet. Evidence: [matched512-4b-b8-whole-program-all-baselines.json](../../../work/qwen35-cuda/matched512-4b-b8-whole-program-all-baselines.json). The source manifest identifies the exact candidate and control; the whole-program candidate remains experimental, with a 2B batch-1 regression and an unrepeated small 4B batch-8 difference.

## Additional vLLM4B batch8 MTP baseline

A separate28-request vLLM MTP3 profile uses BF16 convolution cache and FP32 recurrent state, with128MiB cache per sequence and512-token capacity. Its runtime log explicitly selects the CUDA GDN speculative kernel. On14common equal32groups it measures850.56ms versus 823.80ms for the deferred Garnet profile. The saved vLLM serial cohort is faster, so this slower MTP result does not replace it as the goal baseline. Proposal algorithms and intermediate arithmetic differ, and independent quality remains open.

## Isolated projection-layout screen

36 matrix shapes cover decode rows 4/8/16/32/64 and prefill rows 64/128/256/512 for the existing 2B/4B gate/up/down dimensions. Historical quick/full grids remain unchanged. Candidates retain BF16 inputs/output, FP32 accumulation, eligible compute-type reductions and the original 128 FP64 sample bounds per result.

| Screened candidate class | Median component ratio | Maximum component ratio |
|---|---:|---:|
| Current layout, zero workspace | 1.026 | 1.308 |
| Current layout, bounded workspace | 1.040 | 1.383 |
| Allow transposed layout | 1.096 | 1.394 |

Ratios summarize heterogeneous isolated matrices and are not weighted model-speed estimates. The 256 MiB sweep does not establish L2 residency. Candidates were selected from multiple algorithms in one screen, so repeats and full-model numerical/lifecycle validation would be required before integration. Algorithm IDs alone do not describe complete heuristic configurations. No projection change was promoted. Evidence: [cublaslt-profile-grid-summary.json](../../../work/qwen35-cuda/cublaslt-profile-grid-summary.json).

## Wider D8/MTP7 experiment

Four verifier profiles (B4/B8, C8/C16, verification width 8) and eight speculative-session profiles (B1/B2/B4/B8, P32/P64, D8/MTP7) pass exact state/output/lifecycle checks. Saved graph markers confirm the restored RDC build. These synthetic checks do not establish pretrained quality or speed.

2B B1 real probe: ordinary-output match True, full finite-state hashes match True, saved D4 sequences match True. Evidence: [wide8-pretrained-2b-b1.json](../../../work/qwen35-cuda/wide8-pretrained-2b-b1.json).

2B B1, 17 equal-32 groups: D8/MTP7 205.90 ms versus D4/MTP3 199.96 ms; 24/28 same sequences. Independent quality and 2× acceptance remain open. Evidence: [matched512-2b-b1-packed-p64-d8-mtp7-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-nativecommit-arena-mtp3-cache128-candidate.json](../../../work/qwen35-cuda/matched512-2b-b1-packed-p64-d8-mtp7-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-nativecommit-arena-mtp3-cache128-candidate.json).

4B B8 real probe: ordinary-output match True, full finite-state hashes match True, saved D4 sequences match True. Evidence: [wide8-pretrained-4b-b8.json](../../../work/qwen35-cuda/wide8-pretrained-4b-b8.json).

4B B8, 14 equal-32 groups: D8/MTP7 833.61 ms versus D4/MTP3 741.09 ms; 14/28 same sequences. Independent quality and 2× acceptance remain open. Evidence: [matched512-4b-b8-packed-p64-d8-mtp7-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-nativecommit-arena-candidate.json](../../../work/qwen35-cuda/matched512-4b-b8-packed-p64-d8-mtp7-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-nativecommit-arena-candidate.json).

## Conditional GPU-control feasibility probe

An isolated WHILE graph containing a captured child graph passes 84 bounded-counter cases on SM89, including zero/short/32 limits, mixed rows, invalid limits and repeated reset. The source capture is destroyed before parent instantiation and launch. Normal execution and unfiltered memcheck/racecheck/synccheck pass with zero findings. The initial compile failure used an older API signature; its log and source remain retained.

This is a control/lifetime prerequisite, with no model weights, speculative acceptance algorithm or inference-speed claim. A future resident serving loop would need stable child graphs and resource ownership outside the conditional body. NVIDIA documents restricted body-node types in its [CUDA 13.2 graph guide](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/04-special-topics/cuda-graphs.html#conditional-node-body-graph-requirements). Evidence: [conditional-control-api13-identity.json](../../../work/qwen35-cuda/conditional-control-api13-identity.json), [conditional-control-design.json](../../../work/qwen35-cuda/conditional-control-design.json).

## TensorRT binding-reuse diagnostic

2B B1, 17 common equal-32 groups: 198.21 ms versus historical control 199.96 ms and vLLM MTP 203.27 ms. 28/28 same saved sequences; retained ownership unchanged. All 12 C4 numerical profiles, three unfiltered memchecks and a real full finite-state/output probe pass. The small timing difference is unrepeated; the option remains unselected/default off. This does not establish the 2× target. Evidence: [matched512-2b-b1-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-nativecommit-arena-staticbindings-mtp3-cache128-candidate.json](../../../work/qwen35-cuda/matched512-2b-b1-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-nativecommit-arena-staticbindings-mtp3-cache128-candidate.json).

## Copy-only prompt packing — validation in progress

A default-off packed-decoder option assembles one [time,batch,hidden] FP32 request buffer with pitched copies and exposes contiguous token views. Batched vision output is snapshotted before encoding the next row because the vision graph reuses its buffer. No model arithmetic, weights, precision, prompt cache or image cache changes.

The native helper passes 28 exact-byte/legacy-copy/invalid-input/lifetime checks. The production prompt-preparation path passes 8 cases using deliberately reused fake vision output with distinct images and ragged text. Both component suites pass normal execution plus unfiltered memcheck, racecheck and synccheck with zero findings. These are component results; broad pretrained quality and full-model race coverage are not established.

Source review caught a missing vision snapshot in the first serving integration before any pretrained timing cohort. The owned chain was interrupted, partial numerical logs retained, and the corrected chain uses separate evidence names. The CUDA device fatbin is byte-identical to the preceding RDC binary; host correctness is checked separately. Evidence: [prompt-pack-device-preservation.json](../../../work/qwen35-cuda/prompt-pack-device-preservation.json), [prompt-pack-first-chain-interrupted-exit.json](../../../work/qwen35-cuda/prompt-pack-first-chain-interrupted-exit.json), [prompt-pack-native-plain.json](../../../work/qwen35-cuda/prompt-pack-native-plain.json), [prompt-pack-serving-fixed-plain.json](../../../work/qwen35-cuda/prompt-pack-serving-fixed-plain.json).

2B B1, 17 common equal-32 groups: packed inputs 199.03 ms versus saved control 199.96 ms; 28/28 same saved sequences. Requires paired repetition before promotion. Evidence: [matched512-2b-b1-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-nativecommit-arena-promptpack-mtp3-cache128-candidate.json](../../../work/qwen35-cuda/matched512-2b-b1-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-nativecommit-arena-promptpack-mtp3-cache128-candidate.json).

4B B8, 15 common equal-32 groups: packed inputs 698.52 ms versus saved control 727.65 ms; 28/28 same saved sequences. Requires paired repetition before promotion. Evidence: [matched512-4b-b8-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-nativecommit-arena-promptpack-candidate.json](../../../work/qwen35-cuda/matched512-4b-b8-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-nativecommit-arena-promptpack-candidate.json).

## Packed-input isolated Windows memory

| Model | Batch | Warm dedicated GB | Warm shared GB |
|---|---:|---:|---:|
| 2B | 1 | 5.076701 | 0.081789 |
| 4B | 8 | 13.154992 | 0.081789 |

Differences from the preceding memory profile are only 2 MiB for 2B B1 and 6 MiB for 4B B8; no meaningful memory reduction is claimed. Tracked Garnet allocations remain unchanged. All six image-hash groups match the saved reference and LM Studio 4B B8 memory run. These are isolated, sampled process GPU counters, separate from timing and not proof of true peak or physical residency. Evidence: [memory-combined-2b-b1-512-targetserial-aux0-deferred-nativecommit-arena-promptpack-summary.json](../../../work/qwen35-cuda/memory-combined-2b-b1-512-targetserial-aux0-deferred-nativecommit-arena-promptpack-summary.json), [memory-combined-4b-b8-512-targetserial-aux0-deferred-nativecommit-arena-promptpack-summary.json](../../../work/qwen35-cuda/memory-combined-4b-b8-512-targetserial-aux0-deferred-nativecommit-arena-promptpack-summary.json), [prompt-pack-isolated-memory-input-control.json](../../../work/qwen35-cuda/prompt-pack-isolated-memory-input-control.json).

## Current packed decoder whole-graph trace

One instrumented 2B B1 request records 88 graph launches, 680 copies and 259 memsets. Whole-graph plus uncaptured GPU activity union is 150.804 ms within a 244.092 ms span; between-activity gaps total 93.288 ms. Whole-graph tracing hides internal gaps and node costs. API/idle overlap is not causation or recoverable time. The same saved input/output and request counters had 176.22 ms historical cohort latency; the instrumented Model request is 244.35 ms. This is not an interleaved profiler-overhead control or a speed forecast. Evidence: [profile-packed-current-2b-b1-p64-d4-graph-summary.json](../../../work/qwen35-cuda/profile-packed-current-2b-b1-p64-d4-graph-summary.json).

## Logical SDK-view I/O correction

The original native implementation fails the new byte-boundary oracle: zeroing a middle token view also zeroes the following token. The original failure is retained. Zero, clone and copy now use validated contiguous logical element bytes, while preserving backing-storage owners, grouped access leases and partial-overlap rejection. All 8 boundary/lifetime cases pass normal execution and unfiltered memcheck, racecheck and synccheck. CUDA device fatbin bytes are identical to the preceding DLL; host behavior changed and is checked separately. The packing inference path reads its token views and had already passed full-state preservation; this correction protects general writable-view operations. Evidence: [logical-views-before-fix-exit.json](../../../work/qwen35-cuda/logical-views-before-fix-exit.json), [logical-views-fixed-plain.json](../../../work/qwen35-cuda/logical-views-fixed-plain.json), [logical-views-device-preservation.json](../../../work/qwen35-cuda/logical-views-device-preservation.json).

2B B1 complete finite target-state hashes and outputs match the preceding packed-input build: [logical-views-pretrained-2b-b1-promptpack-cross-prior-state-control.json](../../../work/qwen35-cuda/logical-views-pretrained-2b-b1-promptpack-cross-prior-state-control.json).

4B B8 complete finite target-state hashes and outputs match the preceding packed-input build: [logical-views-pretrained-4b-b8-promptpack-cross-prior-state-control.json](../../../work/qwen35-cuda/logical-views-pretrained-4b-b8-promptpack-cross-prior-state-control.json).

## Reusing verified target-head work

Current source inspection corrects an earlier working-note assumption: partial packed-prefix commit calls `_read()` and repeats full target projection/argmax. The traced case has one initial head, eleven verification heads and six repeated partial-commit heads; seventeen M4 CUTLASS launches plus one M1 GEMV agree with that count. The default-off change preserves masked normalization for MTP correction and reuses authoritative accepted-token predictions. Zero-prefix fallback, accepted-row candidate equality, rollback/fault behavior, state preservation and paired timing remain required. The option has no performance acceptance. Design: [reuse-verified-head-design.json](../../../work/qwen35-cuda/reuse-verified-head-design.json).

Default-off implementation: 12/12 numerical profile exits currently pass. Verifier tests compare accepted candidate IDs with a fresh full head using the same packed shape, check evaluation/reuse counters, preserve zero-prefix fallback and retain original rollback/fault assertions. Session tests retain all 24 MTP/ordinary state and stop cases per profile. These preservation tests do not establish independent pretrained quality.

2B B1 complete finite target-state hashes and output signatures match the preceding head-off build; the probe also compares a separately ordinary target. Evidence: [reuse-head-pretrained-2b-b1-promptpack-cross-prior-state-control.json](../../../work/qwen35-cuda/reuse-head-pretrained-2b-b1-promptpack-cross-prior-state-control.json).

2B B1, 17 common equal-32 groups: head reuse 186.14 ms versus saved packing-only 199.03 ms; 28/28 same saved sequences. Kept GPU allocations unchanged. This single trial requires same-build repetition; it is not a 2× or independent-quality acceptance. Evidence: [matched512-2b-b1-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-nativecommit-arena-promptpack-headreuse-mtp3-cache128-candidate.json](../../../work/qwen35-cuda/matched512-2b-b1-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-nativecommit-arena-promptpack-headreuse-mtp3-cache128-candidate.json).

4B B8 complete finite target-state hashes and output signatures match the preceding head-off build; the probe also compares a separately ordinary target. Evidence: [reuse-head-pretrained-4b-b8-promptpack-cross-prior-state-control.json](../../../work/qwen35-cuda/reuse-head-pretrained-4b-b8-promptpack-cross-prior-state-control.json).

4B B8, 15 common equal-32 groups: head reuse 678.38 ms versus saved packing-only 698.52 ms; 28/28 same saved sequences. Kept GPU allocations unchanged. This single trial requires same-build repetition; it is not a 2× or independent-quality acceptance. Evidence: [matched512-4b-b8-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-nativecommit-arena-promptpack-headreuse-candidate.json](../../../work/qwen35-cuda/matched512-4b-b8-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-nativecommit-arena-promptpack-headreuse-candidate.json).

## Current-DLL Qwen3-VL regression smoke

Qwen3-VL-2B-Instruct starts on the current native DLL and completes two matching 16-token requests with 77 visual tokens each. The existing regression cache folder is reused. This is a legacy serving/reset/vision smoke test, not broad independent quality, historical input-equivalence proof or matched speed acceptance. The first harness attempt passed a program file where the API requires an xModel directory; its failure is retained. The corrected argument and full startup diagnostics preserve the readiness/output assertions. Evidence: [qwen3-vl-promptpack-dll-regression-fixed-root.json](../../../work/qwen35-cuda/qwen3-vl-promptpack-dll-regression-fixed-root.json), [qwen3-vl-startup-harness-correction.json](../../../work/qwen35-cuda/qwen3-vl-startup-harness-correction.json).
