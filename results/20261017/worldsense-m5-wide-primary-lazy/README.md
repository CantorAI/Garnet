# WS-M5-46 — Wide primary weights with global default retirement rejected

The candidate passed correctness and reduced allocation, but the companion text model decoded 26.23% slower. It failed the unchanged 2% preservation gate. No later validation or promotion. WS10 was restored with four file sources online, camera unmounted and the valid benchmark request returning HTTP 404.

## Method declared before source changes

The new joint policy packs only consistent wide checkpoints and retires safe unused default buffers for both models. Qwen3.5-4B uses lossless primary weights; Qwen3-1.7B retains its original raw representation and compute path. Unlike WS44, text default retirement is enabled; unlike WS45, text weights remain raw. This is a distinct whole-candidate comparison, not an isolated causal test of packing or retirement.

An eight-file overlay over canonical `f21c1a2584031ee9d1c129d1e58818f9d1c826d2` reuses the frozen WS44 classifier/admission/shared-scope protection, removes its retirement early return and names the flag/getter for primary packing specifically. The default-OFF `GARNET_METAL_WIDE_PRIMARY_STORAGE` requires primary weights and lazy defaults. Canonical embedding/head metadata must be positive 2D, consistent and width>2048 to qualify for packing. Missing/malformed/conflicting metadata fails closed for packing; retirement is governed by the original safe graph-semantic plan for all checkpoints. Eligibility is cached once, observed through `wide_primary_eligible` and reset on Clear. Cross-primary-scope sharing rejects before GPU allocation; same-scope ownership remains intact.

Whole primary shader, CPU codec, FP32 math/reduction/K-order, kernel geometry, native compute, Model, serving wrapper, budgets/cache policy and CUDA unchanged. No private defaults, explicit residency sets, cached matrix metadata or value16/mixed-attention/vocabulary variant. Native C++/Metal/XLang3 inference and images; CPython orchestrates/audits only. Both OFF/ON use the same 1716-file source; exactly three flags differ, with primary test faults only in the separate correctness build.

The structural allocation projection was 16,867,738,624 bytes; actual allocation was 16,870,899,712 bytes. The projection was not used as acceptance evidence. Owner accounting is distinct from physical memory and allocation overhead.

## Correctness before timing

OFF/ON/fault builds exited zero and all seven native tests per build passed. All 186 complete original FP32 projection records remain byte-identical at original FP64 bounds. Both primary GPU readers checked 4,200,857,600 original FP32 words each across all 201 packed VLM matrices. All 245 raw buffers (197 large text plus 48 original small) remain in the inventory; they are not claimed as primary-reader GPU bit coverage. Retained weight payload plus raw buffers totals 9,850,930,176 bytes.

Full 32 VLM/28 text finite states/tokens, vision/grid, trained layers, host/private/session isolation, shared-after-close, cross-primary-scope rejection and parent isolation, invalid/rebuild, prefix and capacity recovery passed. Both ON models retire defaults and never rehydrate during normal infer. Generic/resident alternation rehydrates and preserves complete outputs/states and host buffers. VLM standalone owners=201; text standalone owners=0 and its per-call primary dispatches remain zero. Joint calls use the 201-owner tuple. OFF primary/default counters remain zero. The old disabled lazy-fault API branch remains unexecuted/unclaimed; original C++ primary fault checks execute.

The three original quantitative/golden/source analyzer ASTs remain unchanged except the new native predicate. Ten actual native correctness calls and negative capability/metadata/retirement/dispatch cases pass. Source/checkpoint/model/library/build/runtime/frame/Skill identities and 1865 build/SDK/host files were frozen before timing and unchanged after both stages. Read-only hardware capture before native tests records Mac17,4, Apple M5, 24 GiB, ten physical/logical cores.

## Fresh WorldSense results

Eight actual WorldSense Skill/Galaxy calls over fresh workers passed complete output/native policy checks. Post-decode sampled maxima:

| Metric | Control bytes | Candidate bytes | Reduction |
| --- | ---: | ---: | ---: |
| Metal allocation | 21,903,376,384 | 16,870,899,712 | 22.975803% |
| Process footprint | 18,061,146,488 | 16,077,125,432 | 10.985023% |

These are snapshots, not peaks or free RAM. Resource timings do not establish speed.

The separate speed stage used twenty calls over two fresh workers, five single-car requests each. First results are retained; warm medians use four samples per model. Complete generated tokens/text, input/effective-config/image/Skill identities and interpreted Skill semantics match the frozen oracle. Warm medians:

| Model | TTFT seconds control / candidate | Total seconds control / candidate | Derived tokens/s control / candidate | Decode change |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | 8.037616 / 2.095901 | 9.355110 / 3.838149 | 26.254647 / 19.367919 | -26.230511% |
| Qwen3.5-4B | 20.205573 / 7.336487 | 31.784764 / 16.699239 | 8.132870 / 9.802632 | +20.531032% |

Text TTFT changed -73.923854%, total -58.972703%; VLM TTFT changed -63.690773%, total -47.461497%. Complete native Skill wall median fell from 41.144560021 to 20.510170479 seconds (50.150954% reduction).

## Decision and limitations

**REJECT:** text derived decoding -26.230511% violates the original 2% limit. VLM +20.531032%, lower latency/allocation/footprint and complete Skill wall do not rescue failure. No favorable unchanged retry, extra actual-WS10 screen, full mirrored confirmation, canonical integration or promotion. All 28 model calls and 14 unique parent requests remain. Parent Skill wall is counted once per frame, not twice for model phases.

The text representation and compute path match the original raw path, but the global allocation/lifetime configuration differs. These results do not prove the cause of the slowdown. Do not divide across other cohorts to claim causal effects. First/warm control latency drift remains visible; no favorable subset is selected. Derived decoding `(generated-1)/(total-TTFT)` includes host/stop/post-loop/possible extra EOS work. GPU/wait spans overlap and native counters do not prove physical residency/completed GPU work. Packing initialization costs and transient peaks are not isolated.

Fixed recorded frames traverse actual WorldSense current Skill → Galaxy → native Garnet. They exclude live decoding/scheduling, event persistence/world-state commits and search retrieval/ranking. Four file sources are restored normally; the second car file is correlated, not independent camera fusion. The M5/24 GiB/4B cohort remains separate from the unmet M4 mini/vLLM target. No consumed source/driver edit, build, competing investigation GPU job or archive copy occurred during timing. All full F32 fixtures, native word binaries, raw first/warm calls, source/model/build/input/runtime/SDK identities and failure decisions are retained; fixture paths may need remapping.

A setup-only UnicodeDecodeError occurred while renaming every C++ file, including unrelated legacy encodings. No build/test existed or launched. Recovery restored the eight known UTF8 overlay files from the frozen seed, applied the declared change and verified all other bytes unchanged before building. The partial source manifest and failed absent-driver log are retained. No numerical/performance run or gate was changed or repeated.

A subsequent read-only source/profile review identified vector packed-weight tile loading as a possible distinct prefill bottleneck experiment. It is not implemented or proven here, and no isolated kernel benefit follows from the existing profiles.
