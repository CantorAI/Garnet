# WS-M5-56 — Whole-request native raw-weight residency rejected by text decode gate

REJECT; no promotion. All28 actual WorldSense/Galaxy model calls/14 parent Skill requests matched original generated tokens, prompts, images, Skills, effective configuration and interpreted semantics. Correctness and resource gates passed, but text derived decoding regressed28.498098%, exceeding the unchanged2%limit. Best validated deployment remains WS10; four video files/no camera/normal benchmark POST404 restored after terminal session37967 exit0.

## Method declared before implementation

WS55's mirrored no-code diagnostic was mixed/order-dependent, with heavy system-wide swapping and large exact-control drift; it did not prove an intrinsic new-runtime or packed-kernel cause. Read the prior WS17 allocation and WS32 per-batch bounded residency failures. This distinct method amortizes one bounded native residency request across a complete serving inference, rather than acquiring/releasing per submitted batch. No unchanged favorable retry or historical-ratio causal comparison.

Seed is immutable WS54,9changed files,1720total files/15overlays from canonical f21c1a2584031ee9d1c129d1e58818f9d1c826d2. Default-OFF REQUEST_SCOPE_WEIGHT_RESIDENCY adds Context/Session begin/end/stats, graph/buffer coverage/generation validation and an XLang3 Model.infer try/finally lifecycle. Original BF16 bits, FP32 math/order, shader programs, xModels, geometry, model/checkpoint/config/token budgets and CUDA source remain unchanged. CUDA execution is not tested on this Mac. CPython orchestrates/audits only; native C++/Metal/XLang3 handle inference and image preparation. No camera or external CPython image/inference bridge.

Lease deduplicates actual MTLBuffer identities, holds strong weight owners and enforces4GiB global/per-device and2048unique-allocation limits. Same queue/set spans prefill and decode; normal Submit preserves it. Drain/end, Abort, Reset, Close and failure paths release lease/budget. Graph generation and actual supplied weight coverage are checked before state mutation; covered shared prefill graphs are cached with strong owners, bounded256graphs. Packed primary weights explicitly fall back(reason7); no packed payload or transient activations are leased. Real text model admits197masters/3,440,902,144B. Empty/unsupported/oversize/busy conditions retain fallback. Residency API success is not proof of physical residency or guaranteed speed. Test-only default-OFF hooks use64KiB and two injected failure stages; measured builds retain4GiB and no hooks.

Controls: raw all10optionsOFF; storage PRIMARY/WIDE/LAZY ON, other7OFF; candidate all10ON; separate fault candidate plus original/new fault hooks. This joint candidate differs from raw by10flags and storage by7; it does not isolate request-residency causality. VLM-only201packed/245raw partition, global lazy retirement, cached metadata, K32 vector prefill, full primary secondary-library isolation and native direct-upload remain seed policies.

## Correctness and evidence before timing

All8native tests x4builds passed(32runs);186original complete projection records were exact. New native test covers lease admission/dedup/owner lifetime, Submit/Abort/partial flush, nesting, count/byte limits, packed fallback, injected reserve/attach failure, contention and destructor release. Initial new128x128negative fixture failed to meet existing packed admission; changed ONLYnewfixture to512x128. Initial source/test/logs/binary identities and failure are retained before rebuild. No inference math, bounds, budgets or performance gates changed and no performance attempt preceded the correction.

All201packed learned matrices passed five GPU readers, EACH4,200,857,600original FP32words. HOSToriginal raw-storage reader checked245raw matrices/1,724,383,232words. Together446unique matrices/5,925,240,832originalwords, retained9,850,930,176B; reader repetitions are not independent outputs. Direct native upload checked546selected weights/334,609,152originalwords. Original32VLM/28text state bits, generated tokens, vision grids, trained layers, prefixes/capacity/error recovery, private/host/session isolation, close/rebuild/rehydration passed three API modes. Short native API capacities512/256 differ from actual WorldSense2048, as declared.

Additional trained native request guards passed actual alternate/uncovered/shared/rebuilt/closed graph handling, invalid active flags without private/host mutation, nesting, Reset/Close and real197-master4GiB global contention/readmission. Actual API text admission/cache hits/no coverage fallback/final active+global0 and VLM packed fallback/final0 were asserted. Seven invalid option configurations rejected. Exact inverse audits recover original Context/Session and Model.infer expressions; all other serving AST and four actual generated shader programs match WS54. Inference source1720files remained fixed. Original numerical/golden/input/model/metric AST gates retained; new request capability and corrupted telemetry guards exercised. Checkpoint/assets/source/build/input/frame/Skill/runtime/SDK plus1874compiler/host dependencies pinned and verified unchanged after each cohort.

## Actual WorldSense measurements

Resource: two fresh workers, one first and one followup car Skill each,8calls. Sampled post-decode Metal21,903,376,384 ->16,870,899,712B(-22.975803%); process footprint18,054,838,552 ->16,080,320,336B(-10.936227%). Original>=10%Metal/>=5%footprint gate passed. These are snapshots, not peaks/freeRAM/physical residency. Resource timings ineligible for speed evidence.

Actual WS10 screen: two fresh workers, one first-after-initialization and four warm identical car Skills/model/worker,20calls. Every returned call passed original exact/native checks. Four warm samples/model/worker; medians below:

| Model | TTFT seconds control / candidate | Total seconds control / candidate | Derived tokens/s control / candidate |
|---|---:|---:|---:|
| Qwen3.5-4B | 24.428803 / 6.005540 | 36.457134 / 15.524295 | 7.731752 / 9.770187 |
| Qwen3-1.7B | 9.062938 / 2.130616 | 10.413626 / 3.975125 | 25.808511 / 18.453577 |

4B TTFT-75.416149%, total-57.417677%, derived decode+26.364459%. Text TTFT-76.490891%,total-61.827662%,derived decode-28.498098%. Complete parent Skillwall45.214559 ->19.504299s(-56.862790%). These are unconfirmed single-screen joint candidate observations. The old deployed WS10 memory telemetry is unavailable/null; do not substitute the separate raw resource denominator. All first outputs/phase profiles/request counters are preserved separately, no favorable selection or exclusions.

Observed system-wide worker-window swap counter equivalents:

| Worker | Global swap-ins GiB | Global swap-outs GiB |
|---|---:|---:|
| control-1 | 41.152 | 44.103 |
| candidate-1 | 4.206 | 4.142 |

vm_stat page size16384B; deltas cover initialization and requests, are system-wide/compressor page-counter equivalents, not per-process attribution or direct disk I/O, and do not identify a latency cause. All endpoints' AC/thermal tool outputs retained, not proof of GPU clocks or invariant thermal conditions. Task footprint/compression/resident fields and Metal allocation/recommended-working-set estimates are distinct, nonadditive and not peak/freeRAM measures.

## Decision, tradeoffs and coverage limits

REJECT this candidate. Favorable4B/Skill latency and memory do not override text decode failure. Request-scoped residency did not establish a practical fix under the complete original gates. It is not valid to compare this regression against WS54's8.54%as an isolated effect: cohorts/controls differ. Do not retry unchanged or loosen gates. Matched storage/raw screens, fresh mirrored full48, canonical rebuild/reproof and actual-production mirrored48 remain undispatched because this stage failed.

Export every first/warm/failure call once, parent Skillwall once per frame. Derived decode=(generated-1)/(total-TTFT), including host/stop/post-loop/possible extraEOS; not isolated GPU throughput. Profiles and per-call residuals are descriptive, component medians nonadditive. Fixed video frames traverse actual WorldSense Skill ->Galaxy ->Garnet; this screen excludes live decode/scheduling, event persistence/shared-state writes and retrieval/ranking. Four normal files are restored; second car view is correlated, not independent-camera fusion. M5/24GiB/4B remains separate from the unmet M4mini/small-model/vLLM handover. No consumed source/driver mutation, builds, competing investigation GPU jobs or archive copying during timing. Full optimization remains active.
