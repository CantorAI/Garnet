# M5 Qwen3.5-4B low-copy native kernel profile

Two fresh native control/candidate cohorts completed successfully on Apple M5 / 24 GiB. This is a small 4B serving regression experiment, separate from the handover's M4 / six-profile vLLM objective. No vLLM speed ratio is claimed.

## Handoff assessment and hypothesis

The reviewed handoff is `docs/research/MACBOOK_CODEX_HANDOFF.md`. The current WorldSense native build had all optional optimization flags disabled. The handover's ordered vector vision admission explicitly requires 768 hidden / 12 heads / 216 vision tokens, so its selected d52 route does not apply directly to 4B's 1024 / 16-head encoder. Its extended prefill explicitly limits hidden size to 1024; it is not a route to a wider 4B prefill chunk. Extra interleaved/coalesced/exact-half weight copies also have a different memory tradeoff at 4B. Rejected reduced-bit rejection indexes and reordered vision math were not retried.

Hypothesis: existing chunk recurrence and cache-update kernels improve 4B prompt processing without new body-weight layouts or precision changes. The predeclared candidate uses parallel delta split, chunk convolution, chunk delta, direct chunk writes, attention prefix and resident KV in-place. The graph profile remains chunk16, parallel linear attention, 65,536 image pixels and the same 1,536+256 admission for these calls. No prefix/image/answer reuse or speculation. Every image is freshly prepared and encoded.

## Method and gates

Frozen source is based on `50890a5ca`; `source-manifest.json` records consumed source including instrumentation. The instrumented serving return adds generated token IDs and input/prefill/decode phase clocks to both runtimes. It changes no model math or prompt. Existing native binary is the control. Candidate is built separately from the same tracked source with the six flags. Library hashes, actual backend strings, complete outputs and source-manifest digest are in measurements.json. Official checkpoint is Qwen/Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, both shards plus index, as in the prior serving report.

WorldSense and its managed Cantor service were paused for each cohort and restored in a finally block. Runs were sequential: control1, candidate1, candidate2, control2. Each fresh process ran bunny, car, traffic, bunny; the first request is recorded separately and excluded from the three-request warm median. There are two complete control/candidate cohorts, eight full outputs per runtime, six warm calls per runtime. All generated token IDs match pairwise. Companion Qwen3-1.7B had a separate three-prompt exact-token guard.

The predeclared retention screen requires both cohort warm total/TTFT improvements, complete-token equality, and median decode rate no worse than 2%. This is a smoke screen, not the older 28-stage trained numerical validation or proof over arbitrary input. The native Metal matmul reference passed. No resource exclusions or favorable extra cohort were added. Hardware caches were not flushed.

## Results

| Cohort | Warm median total control → candidate | Warm median TTFT control → candidate | Median decoder prefill control → candidate |
|---|---|---|---|
| 1 | 11.210 → 8.659 s | 4.816 → 2.470 s | 2.439 → 1.320 s |
| 2 | 11.587 → 8.664 s | 4.649 → 2.410 s | 2.573 → 1.497 s |

Total improves 22.8% and 25.2%; TTFT improves 48.7% and 48.1%. The isolated decoder-prefill medians improve about 46% and 42%. Input-phase timing includes CPU embedding/preprocessing and vision build/encode; it is not pure vision GPU timing. The bundle does not isolate which flag causes the gain.

First requests retain setup costs: complete individual records are saved. Peak process footprint is about 16.73/16.92 GB control versus 15.81/15.82 GB candidate. Maximum RSS is about 6.68/9.58 GB versus 10.04/10.04 GB, respectively. These different process measures do not justify a general memory-saving claim. Initialization is recorded separately. These are not camera FPS or production queue latency.

## Decision and remaining work

Retain as an explicit, opt-in Mac WorldSense kernel profile (`cmake/MetalWorldSense4B.cmake`), preserving default options and CUDA builds. The 4B and text guards passed, but broader trained layer/state parity, image grids, grouped-head masking, long contexts, failure recovery and labeled event/search quality still need dedicated 4B coverage. Do not claim general numerical equivalence from greedy-output parity. The handover's full M4/vLLM goal remains unmet and was not adopted as this experiment's target.

The isolated source, builds, stderr/time logs and controllers remain in the local chat's work/4b-handover directory. Native kernel code is unchanged; the new source deliverable is the optional build profile and evidence. The main WorldSense model size, precision, serving resolution and portal/tenant configuration are preserved.

## Documentation audit limit

The index builder includes this report. The legacy all-research verifier stops
at its assertion that today's src/entry/metal_main.cpp equals the older vector
integration hash. The hash still matches that file at handover commit 703da10ff;
subsequent serving work intentionally changed the current entry point before
this experiment. That historical check is not weakened or reported as passing.
This does not invalidate the separate actual token comparisons, but the global
documentation verifier is not green for the current serving branch.

## Local deployment

The same source/flags were rebuilt at out/build/garnet-metal-4b-opt. Its native
matmul reference passed, and all four fixture outputs matched the selected
candidate again before installation. The library was copied into the running
WorldSense bin directory with the original retained as
libgarnet.dylib.before-4b-kernels. Cantor/WorldSense were restored. Deployment
hash is in measurements.json. Future staging must use this optional build
profile; rebuilding the old all-flags-OFF parent target can replace it.
