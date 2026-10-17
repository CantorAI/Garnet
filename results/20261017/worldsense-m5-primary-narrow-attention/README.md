# WS-M5-42 raw narrow attention with packed MLP: rejected

The design preceded implementation and proposed returning semantic narrow attention projections to original raw BF16 storage while retaining packed text MLP and all4Bweights. Exactly112text Q/K/V/O matrices (704643072B) became raw in place of537615360B payload+metadata; +167027712B overWS41. The already-raw text vocabulary stayed raw. No4Bweight qualified. This is a new storage policy, not a causal comparison against another cohort.

## Native scope and correctness

Immutable WS41private seed1716files, five-filedelta in semantic policy, loader, CMake, capability and native tests. Combined finaldelta against immutable canonicalWS40f21c1a258 is sevenfiles. The new defaultOFF Metal-only flag recognizes *.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight with0<inner<=2048; MLP/ordinary weights of identical dimensions and wider attention retain their prior eligibility. Native upload/compute and entire primaryshader are byte-identical toWS41. Model/wrapper/profiles/budgets/FP32math/Korder/threads/payloadcodec/u32addressing/prefix/image-answer policy and CUDA unchanged.

OFF/ON/fault builds exited0; all7native tests each passed. All186original complete projection records remain byte-identical under original FP64bounds. BothGPUreaders checked5257822208originalFP32words across285packed matrices each.161rawfallback buffers comprise48originalsmall,1vocabulary and112attention owners; these113semantic exclusions are explicitly not primary-reader GPUbit coverage. Totalretained9349231104B matches the prospective byte budget. Full32VLM/28text finite states/tokens,trained-layer/vision/grid/owner/fault/prefix/capacity checks passed. Actual text84MLP/VLM201owner tuples matched before timing.

All three quantitative/golden analyzers retain originalASTs except the route predicate. Ten real native calls and negative capability/counter cases passed. Source1716files, model/checkpoints/library/buildflags, video-derived image/Skill cases and1865compiler/SDK/hostfiles were frozen beforedispatch and remained unchanged after both stages. No competing experiment,build,source mutation or archive copy during timing.

## Fresh real WorldSense gates

Resource8calls passed: Metal21903376384→19405996032B (-11.401805%); sampled processfootprint18060474840→15571957560B (-13.778803%). Postdecode maxima are not peaks/freeRAM. Native owner/encoded counters do not establish physical residency/completedGPUoperations. Resource timings cannot establish speed.

NEW20call speed screen used fresh control/candidate workers, five single-car repetitions each, four warm calls/model. All28combined benchmark calls preserved complete tokens/text,input-prompt/effectiveconfig/image/Skill IDs and interpreted outputs. Warm medians:

| Model | TTFT seconds control / candidate | Total seconds control / candidate | Derived tok/s control / candidate | Decode change |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | 8.723618 / 2.151039 | 9.965900 / 3.682279 | 27.341420 / 22.217445 | -18.740707% |
| Qwen3.5-4B | 25.313396 / 6.501306 | 37.266638 / 15.749718 | 7.781177 / 10.000370 | +28.520017% |

Qwen3-1.7B: TTFT -75.342357%, total -63.051213%.
Qwen3.5-4B: TTFT -74.316735%, total -57.737755%.

NativeSkill wall median 47.238101834→19.508816563s (-58.701100%).

## Decision and evidence limits

**REJECT**: text deriveddecode-18.740707% exceeds unchanged2%preservation bound. Favorable4B/latency/memory/Skillwall results do not rescue failure. Additional actualWS10speed,fullconfirmation and canonicalintegration were not dispatched. No unchangedretry or promotion. WS10normal/fourfile sources/noCamera/validbenchmarkPOST404 restored.

Do not divide WS41/WS42orWS40timings across cohorts to claim a causal attention/head effect. Profile spans cannot isolate those kernels. Deriveddecode=(generated-1)/(total-TTFT)includeshost/stop/postloop/possibleextraEOS,notisolatedGPUthroughput. GPU/wait overlap; packing/init/transient peaks not isolated. First and warm results retained; parent Skill wall includes demand initialization and is counted once perframe.

Fixed video-derived frames traverse actualWorldSense currentSkill/Galaxy/nativeGarnet, excluding live decode/scheduling/eventcommits/retrieval/ranking. NativeC++/Metal/XLang3 image/inference; CPython orchestrates/audits only. No camera. M5MacBook24GiB/4Bcohort remains separate from original unmetM4mini/vLLMtarget. AllF32fixtures, complete original native words, allrawfirst/warmcalls anduniqueparentrequests, prospective methods and source/model/build/input/runtime/SDK IDs retained; fixtureJSON absolute paths may need remapping.

A reporting-only status writer briefly used the speed-check field name on a resource result; it was corrected to native_admission. It did not change or repeat any benchmark. No hardware-before-speed capture occurred, and none is claimed.

Next investigate a distinct lossless vector-helper reconstruction using bounded16-bit value intermediates, starting from full-primaryWS40 rather than this rejected mixed policy. Write design and new exact proof/gates first. No implementation or speed effect is established here.

Packaging recovery: the first packaging attempt reused the WS41 archive destination. The committed7c605abe1 WS41 snapshot was restored exactly (all268SHA files plus originalmanifest), and only two newly generated helper files were removed. All raw benchmark/source files were unaffected. The destination is now asserted against the work-directory name, protocolID and report title; the invalid first packaging-verifier output is retained and not used as acceptance evidence.

Posthoc within-WS42 profile observation: warmtext GPUdecode-span median1193.513542→1484.746000ms; encode12.017001→21.487229ms; prepare3.143560→5.762936ms; decode_seconds1.242626→1.531200s. The GPUspan includes inter-command gaps and does not isolate attention/MLP/ALU costs; wait overlaps GPU. This motivates a GPU-helper hypothesis, not a causal conclusion. All first/warm/model profile slices and source artifact hashes are retained in posthoc-nested-profile-summary.json.
