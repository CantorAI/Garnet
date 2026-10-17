# WS-M5-47 — Exact vector prefill loading rejected by matched timing

The new tile loader passed complete GPU/model correctness, but failed the predeclared matched primary-only timing screen. VLM first-token latency rose22.20%; decoding fell37.68% for VLM and33.16% for text. No promotion or later benchmark stages. WS10 was restored with four file sources online, camera unmounted and the valid benchmark POST returning HTTP404.

## Method declared before implementation

Read-only WS45 warm-profile motivation placed VLM prefill at5.934975s versus vision0.740909s; those nonadditive medians are motivation from a rejected cohort, not this experiment's baseline or an isolated kernel measure. The actual new contrast uses identical full-primary u32 weight representation and eager default lifetimes, changing only `GARNET_METAL_PRIMARY_VECTOR_PREFILL` in a common source.

The default-OFF Metal-only flag requires primary BF16 blocks. A new appended shader fragment copies the three original prefill GEMM bodies with only names and right-tile loading replaced. Existing `primary_quad` restores4K positions across4weight rows; the filler stores identical FP32 bit expansions into the original threadgroup right tile. Shapes, thread counts, left/result buffers, barriers, SIMD multiply/reduction order and writes stay fixed. Generic K32/C32 uses128threads; K32/C64 and K64/C64 use256. Admission proves inner%128 and weightrows%4, so quad boundaries and whole-quad tails are valid. No new weight cache/master, quantization or floating-point tile arithmetic.

Six changed/new files over immutable canonicalf21c1a2584031ee9d1c129d1e58818f9d1c826d2 form a1716-file closure. The entire original primary/body/embedded shaders, CPU codec, original projection tests, Model, serving wrapper, profiles/budgets/cache policy and CUDA source remain byte-identical. Lazy defaults, wide classifiers, private defaults, residency sets and cached matrix metadata are OFF/absent. The main Metal library contains the appended vector fragment in ON; original source-level decode bodies are unchanged, but this does not prove identical GPU machine code for them.

A bounded C++-only GPU reader uses the SAME new filler and matching thread counts for all three geometries, reading original FP32 words after tile placement. It rejects invalid geometry/range/alignment, zero or>8Miword count and open batches before encoding. It does not change the XLang inference/image API or wrapper.

## Compiler and complete correctness

C++ materialization proves OFF's primary fragment byte-identical to the immutable base and ON equal to original+new fragment. Independent inverse substitution recovers each original kernel body except trailing fragment blank separators. This is fragment/source evidence, not whole-program, GPU machine-code/register or performance identity.

Four builds (rawOFF, prior-primary, vectorON, fault) passed. Seven native tests per build and all186complete original projection records match exactly across all four. New GPU checks cover finite BF16 patterns, mixed compact/raw blocks, signs/zeros, all3tile geometries, nonzero row starts, tail260, malformed/open-batch/cap rejection and Abort recovery; actual success markers are preserved for ON and fault.

Both original GPU readers plus THREE new tile-placement readers each checked5,921,308,672 original FP32 words over all398actual packed learned matrices. All48raw fallbacks remain; retained primary+raw=9,033,678,080B. These five reader totals are separately reported, not summed as independent inference outputs. Full32VLM/28text finite states/tokens, trained layers, vision/grid, host/private/shared/rebuild/invalid/prefix/capacity tests passed across actual raw/prior-primary/vector API modes (five full records each). Original bounds/goldens are unchanged. Existing positive `prefill_encoded`, frozen source/library and ON host selection of all3alternate prefill PSOs support the serving-route check; a compiled marker alone is not a completed-GPU counter.

Three original quantitative/golden/source analyzer ASTs are unchanged except native-route checks. Fifteen actual native API calls and negative capability/lazy/owner/delta cases pass. The extra matched screen preserves the complete shared input/golden/model-metric AST prefix and adds prospective5%VLM TTFT and2%memory-preservation gates. Source/model/checkpoint/library/build/runtime/frame/Skill identities and1867 compiler/SDK/host files, including two immutable external CMake profiles, were pinned before timing and unchanged after each stage. Native C++/Metal/XLang3 inference/images; CPython orchestrates/audits only.

## Fresh WorldSense resource gate

Eight raw-control/vector-candidate calls preserved complete outputs and native checks. Post-decode sampled maxima:

| Metric | Raw control bytes | Vector candidate bytes | Reduction |
| --- | ---: | ---: | ---: |
| Metal allocation |21,903,376,384|19,091,357,696|12.838289%|
| Process footprint |18,051,922,296|15,761,700,664|12.686857%|

Snapshots are not peaks/freeRAM. This resource contrast includes primary storage and cannot attribute memory or speed to the vector loader alone. Resource timings are ineligible as speed evidence.

## Fresh matched primary-only/vector speed screen

Two fresh workers, five single-car requests each, twenty model calls. The control uses the new common source with primaryON/vectorOFF; candidate primaryON/vectorON. Exactly one build flag differs. All complete outputs/tokens/prompts/effective configs/images/Skills and interpreted semantics match the frozen oracle. First requests remain; warm medians use four samples/model:

| Model | TTFT seconds control / candidate | Total seconds control / candidate | Derived tokens/s control / candidate | Decode change |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | 1.742743 / 1.779819 | 2.762322 / 3.301154 | 33.446725 / 22.354293 | -33.164478% |
| Qwen3.5-4B | 4.561234 / 5.573706 | 10.885098 / 15.833600 | 14.749632 / 9.192408 | -37.677034% |

Text TTFT changed+2.127477%, total+19.506510%; VLM TTFT changed+22.197323%, total+45.461249%. Complete native Skill wall rose40.139940% (13.649538959→19.128455750s).

Matched Metal allocation stayed19,091,357,696B both sides; sampled footprint15,262,578,320→15,262,396,800B. Memory preservation passed; speed gates failed. Both workers recorded AC power/100% before and after. No power-source difference or other verified exclusion justifies discarding this valid failure.

Within-cohort VLM prefill median4.115255→4.656186s and decode6.323154→10.116177s; GPUdecode span6237.885458→9970.747812ms. Text prefill1.520510→1.572261s, decode1.018593→1.513082s; GPUdecode span986.836792→1473.128146ms. These descriptive spans include inter-command gaps and overlap waits. They do not identify clock/thermal/bandwidth/ALU or a particular kernel cause. Unchanged source-level decode bodies do not establish identical whole-program code generation.

## Decision, failures and limits

**REJECT.** VLM TTFT worsened instead of improving>=5%; both decode regressions, text TTFT+2.127477%, totals and native Skill wall also fail prospective preservation limits. No unchanged retry, later raw-speed20, actual-WS10-speed20, full48 confirmation, canonical integration or promotion. All28 model calls/14unique parentrequests remain. Parent Skill wall is counted once/frame. Derived decoding `(generated-1)/(total-TTFT)` includes host/stop/post-loop/possibleextraEOS; it is not pure GPU rate. Do not combine across cohorts or select favorable requests.

Fixed frames traverse actual WorldSense current Skill→Galaxy→nativeGarnet, excluding live decode/scheduling, event persistence/world-state commits and search retrieval/ranking. Four normal source files are restored; secondcar is correlated, not independent camera fusion. M5/24GiB/4B remains separate from unmet M4mini/vLLM. No consumed source/driver edits, builds, competing investigation GPU jobs or archive copies during timing.

The initial four configure attempts failed before compilation because the frozen native closure omitted the external initial-cache profiles. Primary builds also correctly rejected absent BODY_SHAPE defaults. Both profiles were copied byte-exact from immutablef21Git together; failed logs/CMakeCaches retained and generated folders renamed after terminal failure to prevent stale OFF defaults. Fresh builds then passed. No native or performance test was repeated or gate changed. Full source overlays, external profiles, F32fixtures, FOUR original projection-word binaries, five-reader corpus, shader/proof artifacts, raw first/warm calls, identities, costs and decisions are retained; paths may need remapping.

Next investigate a distinct isolated shader-library and compile-time/vector-store tile implementation, keeping actual legacy main-program source invariant. This is a hypothesis about avoiding cross-program effects and improving load/store organization; the current result does not prove their cause. It needs a new design, full original-bit/source/program proof and every fresh WorldSense gate. No implementation or benefit is established by this report.
