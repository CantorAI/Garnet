# WS-M5-64 — Direct interleaved BF16 prefill: rejected

Correctness and the original resource gate passed. The separate actual-WS10 speed screen failed: companion text derived decode fell8.518706%, beyond the unchanged2% preservation limit. Both models' TTFT/total and whole-Skill wall time improved, but those results do not override rejection. No unchanged retry, later stage or promotion; WS10 remains selected and the optimization goal remains active.

## Prospective method and selection

[Design recorded before source](native-design-before-implementation.txt), [read-only tile/load audit](read-only-prefill-reader-audit.json). WS63 preserved arithmetic but materialized a complete row-major matrix before each converted prefill projection. Its whole-profile rejection did not establish restoration's isolated causal cost. This distinct candidate reads the sole interleaved representation directly in the original FP32 tiles. Earlier broad interleaving retained a second row-major master; block-packed studies changed storage/tiling. Those results are not an M5/4B baseline or a predicted gain here.

Immutable63 seed1720files; current1721files/seven delta versus63: Metal CMake, context header/implementation, native entry, serving counters, extended existing sole test and one new shader header. Every original embedded shader,63decode/read shader, graph/session/xModel/image/CUDA and original seven native tests remain byte-identical. Source transformations inverse exactly. Three prefill bodies are generated from actual constructor source: bf16_gemm,gemm_16_64_64,gemm_32_64_32. Only function names and one global BF16 load expression change. The shared ushort helper indexes(column/4)*inner*4+k*4+column%4. All FP32 threadgroup arrays, input loads, padding, barriers, K/offset order, simdgroup loads/multiply-accumulate and stores restore byte-exactly under inverse normalization. Source equivalence does not prove ISA/register behavior.

DefaultOFF SOLE_INTERLEAVED_DIRECT_PREFILL requires63sole storage and its policy exclusions. Raw exact(N,K)=(6144,2048),(2048,6144), original finite/bounds checks and existing primary admission first. All three direct PSOs and tiled-reader capability checks must pass before conversion; unavailable capacity keeps row-major before conversion. Runtime uses the unchanged original wide predicate and corresponding base/16wide/32wide tile. Converted direct weights never silently fall back to restoration or a different tile. The new fragment appends only to the separate sole Metal library; actual original MAIN/packed-prefill strings remain exact54 in all five builds.

Direct mode adds no restore encoder, retained restore scratch, queue, flush or wait. Original threadgroup tile scratch remains. Existing sole decode, single final BF16 storage and original output ownership rules are retained. Native counters add direct-prefill dispatches; actual text decode/prefill must be positive when exercised, restore counts and restore scratch zero. VLM new dispatches zero; global text owners may remain loaded for laterVLM calls. Disabled rebuilt roles report zero native counters; legacy WS10 lacks these fields and memory is unavailable/null, never zero.

Costs declared before implementation: extra address arithmetic, load/coalescing/register/instruction effects, added Metal/PSO setup and code, with unproven speed benefit from eliminating restoration. No checkpoint/dtype/precision/fast-math/context/image/output/prefix budget change, CPU inference/image helper, camera, engine/SDK change or CUDA removal. C++/Metal/XLang3 perform inference/images; CPython only orchestrates/audits.

## Native, learned and functional proof

Initial design declared four roles; before source/build/model execution an explicit preserved amendment added correctness-only RESTORE compatibility. Five builds31145 pass: RAWoff; minimalPRIMARY; packedON/soleON/directON; separate fault; packedRESTORE/soleON/directOFF. RESTORE is never timed or used to reconsider63's failure. Native64607 passes40tests(8perrole), original186complete projection records and28complete MLP projection records, bitwise equal across all five builds with originalFP64 bounds. Rows1,2,3,16,32,64,128/two inputs, dense cache comparisons, logical bits, three tile geometries, ranges/uninitialized/active batches, separate caches, distinct weights, partialFlush/Abort/source release/peer context, unavailable preconversion, upload faults and predispatch fault/reset/RAII pass. Direct restore scratch remains zero. Default-OFF sole shader source is byte-identical63; no GPU machine-code claim.

New bounded native GPU tiled readers use the SAME address helper and original right-tile N/K assignment/thread counts for32x32,64x64,64x32, then reconstruct logical FP32 words. They validate closed/drained/same-device/geometry/range admission and do not increment serving dispatch counters. Each geometry checks every value in all84 learned textMLP matrices:1,056,964,608FP32words per reader, in addition to the scalar reader and245raw logicalHOST checks. These overlapping categories are separate reader coverage, not unique-word totals or speed samples.

Full71517 initially stopped when a RESTORE resident fixture assumed cached metadata only for a directory ending-on. It had completed all learned word readers; the fixture failed before its numerical execution. Preserved initial scripts, identities, logs and results. Corrected only the explicit role capability selector to-on/-restore; no model source, math, output oracle or threshold changed. Revised closure refrozen BEFORE dispatch. Full25905 terminal0 and independent44508 audit passed. Repeated deterministic correctness is not independent performance confirmation.

Five packed readers each verify4,200,857,600FP32words/201matrices;245raw logicalHOST1,724,383,232words; direct546weights/334,609,152words; scalar+three directtile readers each1,056,964,608words/84matrices. Retained packed+raw9,850,930,176B, with2,113,929,216B in the84 sole matrices. Full32VLM28text states/tokens, generic/resident/private/shared/rebuild/vision/multigrid/prefix/capacity/error recovery pass original oracles across four actualAPI modes including RESTORE. Direct text restore/scratch exactlyzero; compatibility restore route remains positive where exercised. Guard capacities512VLM/256text; actual pipeline calls separately retain2048capacity. All source/runtime/libs/19model assets/math+outer/verifier identities frozen before learned calls and unchanged afterward.

Twenty-one configuration exclusions and10,400malformed copied-call native-route cases pass. Copies are offline fixtures, not new inference samples. All five original analyzer golden/numeric/metric/threshold ASTs remain unchanged outside native route admission. ActualWorldSense functional79808: two recorded car Skill requests/four model calls exact original semantics, prompt IDs, complete tokens/text/configuration/input counts/frame/Skill hashes and native routes. Fresh image preparation remains native-cpp. Functional timings ineligible for speed.

## Original resource gate

Resource7276 terminal0, eight actualWorldSense->Galaxy->Garnet calls, fresh rebuilt RAW control then candidate, one first+one followup per worker. All original semantic/native checks and independent maxima/threshold recomputation pass. Maximum post-decode Metal21,903,376,384 ->16,870,899,712B(-22.975803%); task footprint18,054,936,904 ->16,077,846,328B(-10.950415%). Original10%/5% gates pass.

Snapshots are not peaks/free/unique/residency RAM. This is the whole packed/lazy/direct profile versus fresh RAW, not an isolated direct-prefill memory gain.1,875compiler/SDK/host dependencies and aliases, source/libs/drivers/assets unchanged. Collector actually enumerates all five build directories; its inherited scope sentence still says four. The raw dependency-file list and recorded collector establish actual coverage, not that stale description. No source edits/builds/other investigationGPU work/archive copying during timing.

## Actual deployed-WS10 speed screen

Separate predeclared25270 terminal0,20exact/native calls. Two fresh sequential workers, five requests each; first retained separately, four warm samples per model/worker. Original2048capacity, VLM64/text16chunks,65,536image pixels, shared vision and bounded64token/2entry/64MiB text prefix policy retained. Fresh image encoding; no image/answer reuse or speculation. Warm medians:

| Metric | Deployed WS10 | Candidate |
|---|---:|---:|
|4B TTFT(s)|19.645849|4.960724|
|4B total(s)|29.586240|13.701859|
|4B derived decode(tokens/s)|9.579802|10.613446|
|Text TTFT(s)|7.209787|1.929867|
|Text total(s)|8.506251|3.454886|
|Text derived decode(tokens/s)|25.960016|23.748558|
|Complete Skill wall(s)|38.365900|17.280940|

Text TTFT improves73.232682%, total59.384156%, but deriveddecode regresses8.518706% and fails unchanged2% preservation.4B TTFT improves74.749249%, total53.688408%, deriveddecode10.789822%; wholeSkillwall improves54.957553%. Valid REJECT despite favorable other values. No subsequent matchedPRIMARY/RAW/mirrored/canonical stage, unchanged retry or promotion. Old speed-control memory unavailable/null, never borrowed from resource control. Whole profiles/current pair only; no isolated direct-versus63 comparison, cross-cohort causal ratio, significance or confirmation claim.

Retained text decode-loop median1.309356 ->1.433432s and GPU span1229.759750 ->1400.212812ms. Per-call postdecode residual medians0.000292396 ->0.001011271s. [Individual phase data and counts](posthoc-phase-diagnostics.json). These suggest investigating inner decode work, but do not establish clock/thermal/VM/residency/ISA/load-layout cause. Differences are formed per call before medians; spans overlap and medians are nonadditive.

Deriveddecode=(generated-1)/(total-TTFT) includes host/stop/postloop/possibleEOS/snapshots, not pureGPU throughput. Hardware caches not flushed; first is not hardware-cold. Old readiness cannot separate startup/model-load costs; retain those raw boundaries without inventing a split. Fixedframe/currentSkill coverage excludes live video decode/scheduling/event persistence/sharedstates/retrieval/ranking/navigation/independentcamera fusion. M5/24GiB/4B remains separate from unmetM4mini/smallmodel/vLLM handover.

## Continuation

Preserve64's valid rejection. Next review existing decoder-layout/head/fusion evidence and audit the remaining raw text readers/weight inventory before choosing a DISTINCT decode design. No new successor design/source or gain is established here. All original numerical/ownership/resource/speed/confirmation/canonical gates remain. Normal WS10 restored, four ordinary files online, no camera, valid benchmarkPOST404. Full optimization active; no global-optimum/completion claim.


Archive audit fixture correction: the first independent archive verifier used inherited WS63-to-WS54 path normalization for two reconstructed guard/metric scripts; WS64 is seeded from WS63. It failed before completing verification. The original verifier, failure log, manifest and reason are preserved in first-archive-verifier-path-normalization/. Corrected only the two path normalizations to WS64-to-WS63; the revised independent audit passes. No source, scientific data, oracle, threshold or inference/timing attempt changed.
