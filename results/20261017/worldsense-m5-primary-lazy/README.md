# WS-M5-45 — Joint primary weights and resident-default retirement rejected

The candidate saved memory and improved VLM decoding, but the companion text model decoded 7.91% slower. It failed the unchanged 2% preservation gate. No promotion or later validation stages were dispatched. WS10 was restored, with four file sources online, no camera and the valid benchmark request returning HTTP 404.

## Method and source

The design predates setup and execution. The full 1715-file source closure exactly matches canonical `f21c1a2584031ee9d1c129d1e58818f9d1c826d2`, with **zero source overlay**. Only two build flags differ between OFF and ON: full-primary lossless BF16 storage and safe resident-default retirement. Both apply to both models. WS40 used primary with retirement OFF; WS15 used retirement with primary OFF; WS44 applied both only to wide VLM graphs. This is a distinct joint configuration, not a repeat of those rejected candidates or an isolated component comparison.

No shader, codec, allocation/retirement algorithm, FP32 math/K-order/reduction, Model, image preparation, serving wrapper, profile/budget, prefix policy or CUDA change. Wide-checkpoint scope, cached matrix metadata, private defaults, explicit residency sets, value16 helpers and other rejected variants are absent or OFF. Inference and image preparation use native C++/Metal/XLang3; CPython orchestrates and audits only.

The prospective weight budget was 9,033,678,080 bytes, plus existing unused-default retirement. The structural allocation projection was 16,050,486,528 bytes; actual measured Metal allocation was 16,055,271,424 bytes. The projection was not used as acceptance evidence. Native owner accounting is distinct from physical memory and driver allocation overhead.

## Correctness before timing

New OFF, ON and separate primary-fault builds exited zero. All seven native tests per build passed; all 186 complete projection records remained byte-identical at the original FP64 bounds. Both GPU primary readers checked all 398 actual packed matrices, 5,921,308,672 original FP32 words each. All 48 raw fallbacks remain; retained primary plus raw bytes match the declared budget.

Full 32-layer VLM and 28-layer text finite states/tokens, vision/grid, trained layer, host/private/session isolation, shared ownership after source close, invalid-input/rebuild, prefix and capacity recovery checks passed. Generic/resident alternation rehydrated defaults while preserving complete outputs and host/session states. Actual normal infer calls retire defaults for both ON models and never rehydrate them; OFF counters remain zero. Standalone API owners were 201 VLM and 197 text; normal jointly loaded requests used the expected combined 398-owner tuple. The old disabled lazy-fault API branch remains unexecuted and unclaimed; the original C++ primary fault checks executed.

The three original metric/golden/source analyzer ASTs remain unchanged except the declared native route predicate. Ten real correctness calls and negative capability/owner/metadata/retirement/dispatch cases passed. Complete source, checkpoint/model, library/build, frame/Skill, runtime and 1864 build/SDK/host identities were frozen before dispatch and unchanged after each stage. Hardware was additionally observed through read-only sysctl during resource execution: Mac17,4, Apple M5, 24 GiB, ten physical/logical cores. That observation is explicitly labeled after dispatch.

## Fresh WorldSense resource and speed results

Eight actual WorldSense Skill/Galaxy calls over fresh OFF/ON workers preserved complete outputs and native policy checks. Post-decode sampled maxima:

| Metric | Control bytes | Candidate bytes | Reduction |
| --- | ---: | ---: | ---: |
| Metal allocation | 21,903,376,384 | 16,055,271,424 | 26.699559% |
| Process footprint | 18,058,574,200 | 15,262,905,568 | 15.481115% |

These snapshots are not peaks or free RAM; resource timings cannot establish speed.

The separate twenty-call speed stage used five single-car requests per fresh worker. First calls remain in raw records; warm medians use four samples per model. Complete generated tokens/text, prompt/effective-config/image/Skill identities and interpreted Skill results matched the frozen oracle. Warm medians:

| Model | TTFT seconds control / candidate | Total seconds control / candidate | Derived tokens/s control / candidate | Decode change |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | 8.535188 / 2.227769 | 9.799432 / 3.578353 | 26.707979 / 24.594626 | -7.912814% |
| Qwen3.5-4B | 24.151986 / 6.725575 | 35.362541 / 15.996376 | 8.302791 / 9.973898 | +20.127057% |

Text TTFT changed -73.899008%, total -63.484078%; VLM TTFT changed -72.153119%, total -54.764631%. Complete native Skill wall median fell from 44.725950229 to 19.684644687 seconds (55.988314% reduction).

## Rejection and limits

**REJECT:** text derived decoding regressed 7.912814%, exceeding the original 2% limit. VLM decoding +20.127057%, lower latency, memory and complete Skill wall do not rescue that failure. No unchanged retry, extra actual-WS10 speed screen, full mirrored confirmation, canonical integration or normal-production promotion. All 28 model calls and 14 unique parent requests are retained. Count parent Skill wall once per frame, not once per model phase.

Derived decoding is `(generated_tokens - 1)/(total - TTFT)`, including host/stop/post-loop/possible extra EOS work. GPU/wait spans overlap; these measurements do not isolate ALU, bandwidth, compression or a particular kernel. Do not divide across WS44/WS40 cohorts to claim a causal component effect. Control first/warm latency drift remains visible; no favorable subset is selected. Packing initialization costs and transient peaks are not isolated.

Recorded video frames traverse actual WorldSense current Skill → Galaxy → native Garnet. The benchmark excludes live decoding/scheduling, event persistence, world-state commits and search retrieval/ranking. Four source files are used for normal restoration; the second car file is a correlated control, not independent multi-camera fusion. This M5/24 GiB/4B cohort remains separate from the unmet M4 Mac mini/vLLM goal. No consumed source/driver mutation, build, competing investigation GPU job or archive copy occurred during timing.

All source/build/model/input/runtime/SDK identities, full F32 fixtures, original native word binaries, raw first/warm calls and decisions are retained. Fixture paths may need remapping. Reporting-only failures before timing are retained: an inline verifier SyntaxError and premature restoration attempts missing the per-experiment baseline backup and request JSON. These were corrected before resource execution; no numerical or performance cohort was repeated or gate relaxed.

A distinct next hypothesis keeps text weights on their original raw compute path while allowing safe unused-default retirement for text as well as VLM. It differs from WS44, where text retirement was disabled, and WS45, where text weights were packed. It requires a new design, source scope proof and every fresh gate; no implementation or benefit is established here.
