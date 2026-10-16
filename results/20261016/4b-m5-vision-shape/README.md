# Further 4B optimization: shared vision weights and fixed body shapes

Two candidates were actually executed on the M5/24 GiB MacBook against the already deployed six-kernel optimized 4B library. The first was rejected; the second passed its two fresh predeclared timing cohorts, complete output checks and additional state/embedding checks. This is a small Garnet-versus-Garnet serving experiment, not an M4/vLLM comparison or a production accuracy benchmark.

## Bottleneck and implementation

Switching camera aspect ratios rebuilt the entire fixed-grid vision graph, reloaded the checkpoint and uploaded unchanged vision weights. New Metal vision graphs can now use the existing native `build_shared` mechanism: checkpoint path/shape/dtype/offset/size/mtime checks remain in force, and graph-owned buffers retain their lifetime after the old graph closes. Graph replacement becomes transactional; a failed construction does not discard the current encoder. Only one completed grid graph remains active. Images, patches and outputs are not cached; every frame is freshly encoded.

The 4B Metal serving default enables `shareVisionWeights`; explicit false remains available. CPU vision's default builder is unchanged through a method hook, and CUDA paths are unchanged. The optional Mac cache profile additionally enables existing body-only fixed-shape decode kernels. It does not specialize the vocabulary head, change checkpoint precision, reduce pixels, shrink contexts, add weight layouts or use another inference engine. Decoder specialization has bounded pipeline caching and an original-route fallback.

## Methods and gates

Frozen control/candidate Python source manifests and actual library hashes are retained. The selected checkpoint is official Qwen3.5-4B revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, with both shards and index. Profile: 1,536 input + 256 output admission, prefill16 parallel linear, 65,536 image pixels; output cap180, same compact JSON prompt, three fixture images. Each fresh process evaluates bunny, car, traffic, bunny. First-after-init is separate; warm medians use the other three calls. No hardware cache flush is claimed.

Each attempt uses the predeclared control/candidate/candidate/control order: four separate processes, two contemporary control/candidate cohorts, eight complete outputs per policy (six warm). WorldSense/Cantor are paused during timing and restored. All first/init measurements, individual successful calls and process memory remain in measurements.json. No timing attempt is excluded or repeated favorably. CPU/input phase clocks include graph building, embedding assembly, preparation and vision execution, not pure GPU time.

## Attempt 1 — vision sharing only: rejected

Token outputs matched; warm total improved 9.4% and 8.2%, TTFT31.8% and30.2%. But cohort2 median decode throughput fell from11.5320 to11.2561 tokens/s (2.39%), exceeding the predeclared2% preservation limit. This candidate was not deployed or rescued by an extra unchanged cohort. The experiment does not isolate whether that small change reflects memory/cache effects or runtime variation.

## Attempt 2 — sharing plus body specialization: retained

The second candidate adds existing BODY_SHAPE kernels, selected to address the prior decode-preservation failure. Both fresh cohorts must improve warm median total by at least5%, improve TTFT, retain all complete generated tokens, and keep median decode within2% of control. Actual last-decode counters must show248 specialized body projections and zero misses per request, with no specialized dispatches in control. All those checks passed.

| Cohort | Median total control → candidate | Median TTFT control → candidate | Median decode tokens/s control → candidate |
|---|---|---|---|
| 1 | 11.340 → 9.087s | 4.279 → 2.455s | 11.389 → 12.154 |
| 2 | 10.781 → 8.915s | 3.605 → 2.233s | 11.287 → 12.124 |

This is another17–20% total-latency improvement versus the previously optimized build, with38–43% lower TTFT and7–8% higher median decode rate in these fixtures. Do not multiply this by earlier component or live-workload gains. Vision graph construction with sharing was about15–19ms median in the full serving runs, versus649–1,475ms control. The separate vision-only test measured shared builds about5–7ms. First initialization still loads weights and retains its cost.

## Correctness and recovery checks

- All642,560 FP32 vision embedding values are bitwise identical across three actual grids (12×20,22×14,12×18) and a return to the first grid.
- Shared replacement survives closing its source; a closed source is rejected. A failing replacement (missing extra vision layer) leaves the original graph usable with identical output.
- Modified patch bytes produce changed embeddings, establishing fresh encoding for that test.
- Sixteen complete generated requests per attempt (eight per policy) match token-for-token.
- After a separate car-image request with16 generated tokens and512-token capacity, all state parts across all32 decoder layers match bit-for-bit between control and retained candidate. This is one sampled request, not exhaustive equivalence.
- Three independent Qwen3-1.7B text prompts match complete tokens between libraries.
- Native Metal matmul reference and the 4B config contract pass.

The exact state/value counts and hashes, drivers and source manifests are included. Broader long-context, grid, batch, concurrency, numerical and labeled semantic-quality coverage remain necessary. No numerical tolerances were loosened. The old global research verifier still encounters the documented historical entry-point hash assertion; it is not represented as passing.

## Memory and deployment

Separate maximum RSS and process-footprint values are in measurements.json. Peak footprint is about12.1GB for the sharing candidate versus about15.8GB in the first attempt's control; maximum RSS is about10.0GB for both. These process observations are not unique GPU allocations or a general memory-saving guarantee. Replacement temporarily retains both graph structures while sharing immutable weights.

The BODY_SHAPE library and serving changes are installed locally with the previous optimized library/source retained for rollback. A launchd bootstrap raced asynchronous removal during deployment; it was recovered by loading the service after removal, stopping the temporary package-owned Cantor, and restarting WorldSense against the single launchd Cantor. Final runtime verification is recorded separately. Portal/default tenant and video-only sources are preserved.

The optional profile is cmake/MetalWorldSense4B.cmake. For an existing CMake cache, explicitly set BODY_SHAPE=ON (cache defaults do not override existing OFF entries), then rebuild/stage Garnet. Fresh builds can use the cache initializer. Serving code changes require a Cantor restart. This retained change includes checkpoint-immutable sharing and original-arithmetic specialization; general model correctness is not claimed from this sample.
