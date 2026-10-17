# WS-M5-48 — Isolated vector prefill library rejected by matched timing

The candidate passed complete original-bit/model correctness and the raw-control resource gate, but failed its predeclared matched primary-only timing and memory-preservation screen. It was rejected. No later speed stages, canonical integration or promotion. The terminal stage log verifies normal WS10 restoration, four file sources online, camera unmounted and valid benchmark POST returning HTTP404.

## Prospective method and controls

Design and protocol preceded implementation. Six source overlays over canonical f21c1a2584031ee9d1c129d1e58818f9d1c826d2 form a1716-file closure, seeded from frozen WS47. Joint changes: a second Metal library for prefill, compile-time tile geometry, and aligned float4 stores of original FP32 bit expansions. The main library is constructed from the original unextended source. The separate prefill library contains that entire main source plus the new vector-prefill fragment; only three new prefill PSOs and the bounded tile diagnostic use it. The extra library/source/init costs are included. Both models use full-primary u32 BF16 blocks and eager defaults. No quantization, arithmetic reorder, cache expansion, image/answer reuse, smaller budgets or CUDA change.

Generic C32/K32 uses128threads, C64/K32 and C64/K64 use256. Same original right-tile byte counts, multiply order, barriers and outputs; template geometry and aligned float4 bit stores replace dynamic scalar stores. The matched primary-only/candidate contrast changes TWO flags (vector prefill and isolated library), not one. The raw/candidate contrast changes all three including primary storage. New options default OFF.

## Correctness and provenance

Four builds raw/prior-primary/candidate/fault passed seven native tests each and all186complete original projection records exactly. Both original GPU readers and all THREE new tile-placement readers each checked5,921,308,672 original FP32 words over398packed learned matrices;48small matrices remain raw. Retained primary+small raw storage=9,033,678,080B. Finite/mixed compact/raw/zero/sign patterns, all3tile geometries, nonzero row starts, tail260, invalid geometry, open-batch, 8Miword cap and Abort recovery passed. Reader counts are separate coverage, not independent inference outputs.

Full32VLM/28text states/tokens/vision/grid/prefix/capacity/ownership/isolation tests passed across raw/prior-primary/candidate actual APIs (five full records each), with original numerical bounds and goldens unchanged. Original primary/body/embedded shaders, CPU codec, Model, wrapper, CUDA and projection test remain byte-identical. Source inverse and independent C++ fragment materialization recover original GEMM bodies. ACTUAL constructor-produced main programs are byte-identical prior-primary/candidate/fault (SHA2562b29178d6b2f3ec30db757766dd1274845711eccf573e78497ee1f8828afdb81); candidate prefill program is exactly main+fragment. All eight actual program files retained, including empty raw/prior-primary prefill strings. These are source identities, not GPU ISA/register or performance proof.

Three original analyzer ASTs retain quantitative/golden checks, fifteen actual API calls and negative route cases passed; matched metrics preserve the common AST and add prospective >=5% VLM warm TTFT gain and <=2% regressions for both models, complete Skill wall, sampled Metal and footprint. Source/build/model/runtime/frame/Skill identities and1867 compiler/SDK/host files including both immutable external CMake profiles were pinned and unchanged. Native C++/Metal/XLang3 inference and images; CPython orchestration/audits only.

## Fresh resource and matched WorldSense results

Eight resource calls were exact. Raw-control/candidate post-decode sampled Metal maxima21,903,376,384→19,091,357,696B (-12.838289%); process footprint18,055,608,768→15,628,302,064B (-13.443505%). These include the storage contrast, are snapshots rather than peaks and are ineligible for speed claims.

The NEW matched screen used two fresh workers, five single-car requests each,20model calls, retaining first requests and four warm samples/model/worker. All outputs/tokens/prompts/effective configs/images/Skills and interpreted semantics match the frozen oracle.

| Model | Warm median TTFT s control / candidate | Total s control / candidate | Derived tokens/s control / candidate |
| --- | ---: | ---: | ---: |
| Qwen3-1.7B |1.898073 /1.635913|3.019821 /3.128556|30.402692 /22.811124|
| Qwen3.5-4B |5.036732 /5.569731|12.295953 /15.247389|12.815986 /9.640914|

4B TTFT+10.582237%, total+24.003310%, decode-24.774311%; text total+3.600733%, decode-24.970050% despite improved TTFT. Complete native Skill wall15.460178→18.376658s (+18.864470%). Matched sampled Metal stayed19,091,357,696B; footprint15,266,100,760→15,639,492,432B (+2.445887%), exceeding the2% limit. No qualifying5%VLM TTFT improvement. See posthoc-observations.json for within-cohort nonadditive phase medians and compressed/internal/resident memory snapshots. Substantial compression is observed; this is not an exclusion or a proven cause.

Both workers recorded AC100% before/after. Before/after pmset thermal metadata was declared before dispatch and collected outside inference: no OS thermal/performance warning was recorded. This is not direct GPU clock or temperature telemetry and does not establish absence of throttling. No verified exclusion justifies discarding the valid failure.

## Decision and limits

REJECT; no unchanged retry, raw20/actualWS10-20/full48 stages, canonical integration or promotion. All28returned calls and14parent requests retained. Derived decode=(generated-1)/(total-TTFT), including host/stop/post-loop/possible extraEOS work, not pure GPU throughput. Parent Skill wall is counted once/frame. Snapshots are not peaks; native owner accounting and encoded deltas are not physical residency or completed GPU counts. Never combine old/new cohorts or select favorable requests.

Fixed frames traverse WorldSense Skill→Galaxy→Garnet, excluding live decode/scheduling, event persistence/world-state commits and retrieval/ranking. Four files restored; second car is correlated, not independent camera fusion. M5/24GiB/4B is separate from the unmet M4mini/vLLM target. No consumed source/driver edits, builds, other investigation GPU jobs or archive copies during timing. All failures, costs, first/warm calls and source profiles are retained. Canonical production acceptance remains separately required.

A future distinct hypothesis may reduce duplicate prefill-library source and change the2K/4row load organization to use more loader threads. Current evidence proves neither cause nor benefit. Any such method requires a new prospective design, all original-bit/model/source/program checks and every fresh WorldSense gate.
