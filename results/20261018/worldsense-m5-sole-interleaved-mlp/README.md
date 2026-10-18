# WS-M5-63 — Sole interleaved text MLP weights: rejected

The candidate passed native, learned-model and real WorldSense correctness, and the original resource gate. It failed the separate actual-WS10 speed screen: companion text TTFT increased40.478894%, total increased34.213622%, and derived decode decreased2.146325%, beyond unchanged2% preservation limits. Favorable4B and complete-Skill values do not override that failure. No unchanged retry, later stage or promotion. WS10 remains selected; full optimization remains active.

## Prospective question and implementation

[Design recorded before source](native-design-before-implementation.txt); [all-reader audit](read-only-consumer-audit.json). WS62 localized substantial sampled custom-pass intervals to text MLP projections, but did not establish end-to-end shares or predict a layout benefit. Earlier lookahead, worker isolation, QoS and encoder-reuse failures were not repeated unchanged.

Immutable untraced WS54 seed1718 files; isolated1720-file source/seven changed or added: Metal CMake, context header/implementation, native entry, serving telemetry, new standalone shader and new native test. All original embedded shaders, graph/session/xModel/image/CUDA and original tests remain byte-identical. Source transformations have exact inverse records. Actual original MAIN and packed-prefill strings match54 in every build; new shader compiles in a separate Metal library. Source equivalence is not ISA/register/performance equivalence.

DefaultOFF SOLE_INTERLEAVED_MLP. Only remaining raw weights after existing primary admission, exact(N,K)=(6144,2048)or(2048,6144), non-float/non-vision, with device pipeline admission before conversion. Exactly one final shared BF16 buffer per converted weight, filled directly from checkpoint bytes with an original-bit roundtrip. No second full CPU packing vector or retained row-major master. Shared weight owner carries the format; memory inventory labels it once. Existing packed4B weights, raw attention/vocabulary weights and unsupported shapes retain their original routes. Separate bounded shape-PSO cache prevents raw/interleaved cache collisions.

Decode preserves16outputs/128threads, laneK*4/stride128, j0..3 FP32 accumulation and SIMD reduction order; only interleaved uint4 loads change. Generic interleaved fallback never dispatches a row-major reader on converted bytes. For rows>2, an integer GPU deinterleave writes one bounded25,165,824-byte private scratch per context, then the exact original selected prefill kernel consumes it. Original serial encoders/queue and tracked buffers order restore, consume and next overwrite, including Flush/Abort. No new flush/wait or restoration cache. Actual serving text model shared one25MiB scratch total in these calls; this observed total is not a universal model/context bound.

The84 admitted textMLP matrices contain1,056,964,608 BF16 values (each expands to one exact FP32 word) and2,113,929,216 retained bytes. HOST logical readback translates indices explicitly; a new bounded independent GPU logical reader reconstructs every admitted learned word. Existing245raw logical HOST checks remain, rather than falsely calling converted physical storage row-major. Native atomics report live weights/bytes, uploads, decode/restore dispatches and scratch bytes with RAII decrements. New stats are returned at native request boundaries for all rebuilt roles; disabled roles explicitly report zeros. Legacy WS10 has no such telemetry and its missing memory remains null.

Costs considered prospectively: CPU packing/roundtrip and Metal/PSO setup, address/register effects, private scratch and full restore traffic for every prefill projection. Inference and image preparation stay native C++/Metal/XLang3. CPython only orchestrates/audits. No precision/checkpoint/context/image/output/prefix budget change, external CPython inference/image helper, camera, engine/SDK change or CUDA removal.

## Correctness, setup failures and provenance

Four roles: RAW with nine prior flags/new flagOFF; minimalPRIMARY with three base flagsON/newOFF; packedON with all nine prior flags/newON; fault with packed/newON plus correctness-only hooks. Original source/flags remain separately identifiable; a rebuilt OFF role is not the old deployed WS10 binary.

First new unit had an invalid repository-relative header include. Build81238 exited1; preserved scripts/manifests/logs, corrected it to the exported header path, then all four builds62705 exited0 before native execution. A separate mixed-auto declaration was corrected before first build and recorded. Native18800 passed32tests(8perrole), original186 full projection records and new28 complete MLP projection records across seven row counts/two changed inputs, all bitwise equal. Original FP64 bounds retained. Tests cover every logical bit, sparse full-width inputs plus dense cache/queue comparisons, distinct weights through one scratch, partial submissions, source release, peer context, Abort/unreadability/reset, unsupported shapes/nonfinite/ranges, unavailable preconversion fallback, weight/scratch allocation failure and RAII accounting. No timing claim from these tests.

Full controller62149 exited0 and independent20743 audit passed. Five packed readers each4,200,857,600FP32words over201matrices;245raw logicalHOST1,724,383,232words; retained packed+raw9,850,930,176B. Direct546weights/334,609,152words exact. Additional GPU reader verifies all84 sole matrices/1,056,964,608originalwords. Full32VLM28text states/tokens, generic/resident/private/shared/rebuild/vision/multigrid/prefix/capacity/error recovery pass original oracles. Actual new text route positive, VLMzero. Entire source, libraries/runtime,19learned assets, mathematical guards and outer controller/verifier identities frozen BEFORE first learned launch and unchanged afterward.

Configuration19966 initially encountered an earlier joined-prefill dependency guard because that invalid fixture omitted exact-half. Preserved failure and script; supplied the dependency and expected the explicit exclusion. Revised79177 passed six original/fourteen new rejected configurations; no source/numerical gate changed. All row-major-only auxiliary policies fail before execution, including combinations without primary storage. Offline admission checks reject8,920 malformed copies of retained calls; these are fixtures, not additional inference samples. Five original analyzer golden/numeric/metric/threshold ASTs are unchanged outside native-route admission.

Four actualWorldSense functional calls41869(two recorded car Skill requests) match original semantics, complete tokens/text/prompts/config/input/frame/Skill hashes and native routes. Not speed evidence. Candidate/source/driver/runtime/model/build/SDK/host closure froze before resource and speed,1874dependency files/aliases unchanged after workers. Every first/warm returned record retained; no posthoc exclusion or causal ranking across earlier cohorts.

## Real WorldSense resource and speed results

Resource9600 terminal0, eight actualWorldSense->Galaxy->Garnet calls over two fresh workers(one first+one followup each). Original native/golden checks and independent snapshot recomputation passed. Fresh rebuilt RAW control versus candidate: maximum post-decode Metal21,903,376,384 ->16,896,065,536B(-22.860909%); task footprint18,053,822,744 ->16,101,716,856B(-10.812701%). These pass original10%/5% gates. They are snapshots, not peak/free/unique/resident RAM, and describe the entire packed/lazy/layout profile rather than an isolated layout memory gain.

Separate predeclared20-call actualWS10screen14265 terminal0. Two fresh workers, five requests each, first retained separately and four warm samples per model/worker. All20complete outputs and actual route/source/control identities pass. Warm medians:

| Metric | Deployed WS10 | Candidate |
|---|---:|---:|
|4B TTFT(s)|14.284340|4.691133|
|4B total(s)|22.760063|12.723547|
|4B derived decode(tokens/s)|10.974320|11.522185|
|Text TTFT(s)|4.668330|6.558018|
|Text total(s)|5.925971|7.953460|
|Text derived decode(tokens/s)|26.277303|25.713306|
|Complete Skill wall(s)|27.884260|20.792476|

Valid REJECT. Text TTFT/total/deriveddecode all fail unchanged preservation gates.4B TTFT improves67.158911%, total44.097051%, deriveddecode4.992248%; wholeSkillwall improves25.432929%. This is a joint profile screen against WS10, not an isolated interleaving comparison to54, significance, independent confirmation or deployable acceptance. No subsequent matchedPRIMARY/RAW/mirrored/canonical timing stage launched. No favorable retry. Speed-control memory unavailable/null, never reused from resource control or treated as zero.

Deriveddecode=(generated-1)/(total-TTFT), including host/stop/postloop/possibleEOS/snapshot overhead, not pureGPU. [Posthoc retained phase diagnostics](posthoc-phase-diagnostics.json) retain per-call decode/prefill spans, residuals and exact restoration counts. Differences formed before aggregation; GPU/wait/encode spans overlap and medians are nonadditive. Extra prefill restores are an explicit cost to investigate, not an isolated causal explanation of this whole-profile contrast. Hardware caches not flushed; first is not hardware-cold. Old readiness does not separate startup and model-load costs.

## Scope and continuation

Fixed recorded JPEG/currentSkill calls test actual WorldSense/Galaxy/native inference. They exclude live video decoding/scheduling, event persistence, shared world-state writes, retrieval/ranking, navigation and independent-camera fusion. Four ordinary video files remain online, camera unmounted; original portal/tenant preserved. This M5/24GiB/4B investigation remains separate from the unmet M4mini/small-model/vLLM handover.

Next possible DISTINCT design: audit the original prefill tile load/index arithmetic for direct interleaved BF16 readers, removing full-matrix restoration while preserving the original FP32 tile/reduction arithmetic. No design/source/performance result for that successor exists here. Keep every original numerical/ownership/resource/performance/confirmation/canonical gate. Full optimization remains active, no global-optimum or completion claim.
