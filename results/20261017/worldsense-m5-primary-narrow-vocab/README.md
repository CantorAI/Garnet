# WS-M5-41 semantic raw narrow vocabulary: rejected

Before implementation, the design proposed returning semantic narrow vocabulary owners to the original raw BF16 route. For this workload it restores exactly the tied Qwen3-1.7B vocabulary matrix (151936 x 2048,622329856 bytes), leaving all4B primary weights and other matrices unchanged. This increases retained bytes148525312 versusWS40. It is a distinct storage tradeoff, not a quantization or arithmetic change.

## Native implementation and correctness

Seven-file isolated delta over canonicalWS40 source f21c1a258, copied while repositoryHEAD50260dee3 had identical consumed source. DefaultOFF Metal-only policy uses checkpoint role names and bounded hidden width<=2048/rows>65536; ordinary matrices of the same shape remain eligible. UploadBF16 gains an explicit allowPrimary flag, returning false to the exact original raw upload. No forced floatPrefill, dual raw/packed owner, Model/wrapper/primaryshader/FP32/Korder/thread/profile/budget/prefix/image-answer-cache/CUDA change. A shared pure native predicate governs loader and corpus expectations.

Builds OFF/ON/fault exited0; all7native tests each passed, including new role/boundary and complete raw/packed FP32 projection/ownership checks. The original186complete projection records remain byte-identical under original FP64bounds. Both GPU readers reconstructed5610143744 originalFP32words across397packed matrices each;49raw fallback buffers include the one semantic exclusion. The raw vocabulary is explicitly not counted as primary-reader GPU bit coverage. Retained payload+metadata+raw total9182203392B matches the prospective plan. Full32-layerVLM/28-layertext finite states/tokens, vision/grid, prefix/cache/capacity, trained layers, owner isolation and failure/recovery checks passed. Actual native counts196text/201VLM/397combined and metadata tuples matched before any benchmark.

Three quantitative/golden/identity analyzer ASTs matchWS40 except the native route predicate. Ten actual native calls and negative capabilities/metadata/head-count cases passed. Frozen1716source files, model/checkpoint assets, library/build flags, video-derived image/Skill inputs and runtime/SDK/host dependencies (1865files) were captured before timing and unchanged after both stages. No concurrent experiment/build/source mutation/archive transfer during timing.

## Fresh WorldSense gates

Resource8-call gate passed. Metal21903376384→19239862272B (-12.160290%); sampled process footprint18071895448→15410329256B (-14.727654%). Post-decode sampled maxima, not peaks or freeRAM. Native accounting and encoded counters do not establish physical residency or completedGPUcounts. Resource timings do not select speed.

NEW20-call single-car screen used two fresh workers, five repetitions each, first/warm separate. Complete tokens/text, prompts, effective configs, input/image/Skill identities and interpreted outputs matched the original goldens in all28combined benchmark calls. Warm results:

| Model | TTFT change | Total change | Derived decode change |
| --- | ---: | ---: | ---: |
| Qwen3-1.7B | -72.295214% | -58.953578% | -6.116880% |
| Qwen3.5-4B | -71.110831% | -54.426493% | +23.787389% |

Native Skill wall median 45.038090209 → 20.578387354s (-54.308925%).

## Decision and limits

**REJECT**: text decode -6.116880%, beyond the unchanged2%bound. Favorable4B/memory/latency results do not rescue failure. The additional actualWS10 speed screen, full mirrored confirmation and canonical integration were gated and never dispatched. No promotion or unchanged retry. NormalWS10 was restored, four video files online, camera unmounted, valid benchmarkPOST404.

The result does not isolate the vocabulary cost and must not be divided against WS40 timings from another cohort to claim a causal head effect. Decode is `(generated-1)/(total-TTFT)`, including host/stop/post-loop/possibleextraEOS work, not isolated GPU throughput. Descriptive first/warm costs remain in raw data and posthoc summary; GPU/wait spans overlap and cannot be added. Native Skill wall counts the whole parent request once, including demand initialization for the first request. Packing time/transient peaks are not isolated.

Fixed archived video-derived frames traverse actual WorldSense currentSkill→Galaxy→nativeGarnet. They exclude live decode/scheduling/event commits/retrieval/ranking. CPython orchestrates/audits only; image/inference is nativeC++/Metal/XLang3. M5MacBook24GiB cohort is separate from the original unmetM4mini/vLLM objective.

Archive reconstruction preserves all1716consumed source hashes against immutablebase, all original native output words, fullF32fixtures, raw first/warm calls and unique parent requests, model/build/runtime/SDK identities, and prospective/rejected/unexecuted methods. Fixture JSON absolute paths may need remapping to the included byte-identical .f32files. Next consider a distinct semantic raw-narrow-attention policy while retaining narrow MLP primary storage, only after an actual tensor byte budget and new design; no speed effect is established.
