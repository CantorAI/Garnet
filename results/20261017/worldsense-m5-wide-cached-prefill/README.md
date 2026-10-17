# WS-M5-53 — Wide 4B storage with cached metadata rejected on text decoding

The prospective WS53 joint method passed complete correctness and resource feasibility, then failed its first speed gate against deployed WS10. Text derived decode regressed14.974627%, beyond the unchanged2%limit. The substantial 4B/Skill improvements below remain single-screen observations. REJECT, no unchanged retry, no later stages and no promotion. Normal WS10 and four video sources were restored, with camera unmounted and a valid benchmark POST returning404.

## Method and controls declared before implementation

Design/protocol precede source creation. Merge the previously tested WS46 wide-checkpoint admission into frozen WS52, preserving original raw BF16 storage for the2048-wide text model and packing only the consistent2560-wide 4B checkpoint. Eligibility uses canonical embedding/head metadata: present widths must agree and exceed2048; missing, malformed or conflicting evidence fails closed. Graph reset clears eligibility; cross-scope shared builds reject before GPU allocation without changing the parent. Existing raw direct-upload callers retain their default admission parameter. Retirement remains global for both models; a compiled primary marker does not imply that text weights are packed.

The candidate combines lossless primary storage, lazy default retirement, cached immutable matrix metadata, isolated minimal duo prefill, K32-only host selection and hoisted native direct float upload. Raw control has all8 optionsOFF. The planned storage control has PRIMARY/WIDE/LAZY ON, the other5 OFF; candidate adds vector/isolation/direct-upload/K32/CACHED. That matched-storage stage was NOT dispatched after the actual-WS10 failure. No isolated component gain is claimed.

Source1717 files,9 overlays over canonical f21c1a2584031ee9d1c129d1e58818f9d1c826d2. Reconstructed six wide changes from immutable WS52 plus WS46 delta; the additional context change is a bounded, read-only host raw-storage diagnostic. Entry adds only a capability marker; native tests retain WS52 plus WS46 admission and new raw-reader cases. Shader programs, codecs, projection test, wrapper, learned bits, numerical math and CUDA sources remain unchanged. All options retain defaultOFF behavior. CUDA execution is not tested on this Mac.

The C++-only raw reader holds its owning storage, requires same-device shared original BF16 storage and a closed/drained batch, rejects packed/empty/invalid ranges, caps reads at8Miwords and safely returns original bits via memcpy and <<16. It neither changes inference routing nor represents GPU execution. Finite-pattern, unaligned/source-alias, offset, cap/overflow/empty/packed rejection, open-batch/Abort and owner restoration tests passed. Same-device/private rejection branches are source-guarded, not claimed as exercised on additional hardware.

## Correctness and provenance before timing

All7 native tests in each4 builds passed, including cached/live-MPS property parity and all65536 direct-upload patterns. All186 complete original projection records matched across4 builds. The446 learned matrices partition as201 packed VLM and245 raw (197 large text plus48 small). Five packed GPU readers EACH checked4,200,857,600 original FP32 words; the separate HOST original raw reader checked1,724,383,232 words. These cover5,925,240,832 unique original words, not a sum of independent model outputs. Retained packed/raw storage9,850,930,176B. Direct native float upload checked another546 selected vision/norm/depthwise/patch candidates totaling334,609,152 words; that selection is a superset of this prompt's accesses.

Full32VLM/28text state bits, generated tokens, trained linear layers, vision grids, prefix cache, capacity/error recovery, private/host isolation, shared-source close/rebuild, ordinary/resident alternation and late-session ownership passed. All3 actual API modes yielded5 matching records each. Text primary encoded deltas remain zero, with own primary ownership zero; global VLM ownership may remain201 when both models are loaded. Normal retirement is positive for storage control/candidate, normal rehydration zero; required ordinary-call rehydration remains exact. Full API capacities512/256 differ from normal WorldSense capacity2048, as declared.

Actual four constructor-produced MAIN/prefill programs are byte-exact WS52. Original F32 fallback body is intact; the complete merged loader is mechanically reconstructed from immutable52+46, not incorrectly claimed to differ only by the WS50 materialization block. The built CPU object has one rowBytes getter outside all checked backedge families and23 static vector-shift instructions; that is code-generation evidence, not measured component speed. Four invalid build-option combinations rejected as expected. Original numerical/golden/source/metric analyzer ASTs remain unchanged outside declared native admission. Fifteen real calls and adversarial capability/lifecycle/metadata/counter cases passed before timing. Actual-WS10 admission was checked on12 prior calls solely for schema compatibility.

Source/build/library/model/input/Skill/runtime/frame identities and1871 compiler/SDK/host files were pinned before timing and unchanged afterward. Actual WS10 source1706 files was independently pinned. Native C++/Metal/XLang3 perform inference and image handling; CPython is orchestration/audit only. No consumed source/driver edit, build, competing investigation GPU job or archive copying occurred during timing.

## Fresh actual WorldSense results

Eight exact/native-valid resource calls over two fresh workers: post-decode sampled Metal21,903,376,384 ->16,870,899,712B (-22.975803%); process footprint18,071,715,248 ->16,074,684,240B (-11.050589%). These are snapshots, not peaks/free RAM or physical-residency proof. Resource timings are excluded from speed claims.

The next predeclared stage was the actual deployed-WS10 screen: two fresh workers, each one first-after-initialization request plus four warm requests on the same fixed car frame and Skill,20 model calls total. Every token/text/prompt/effective configuration/image/Skill/interpreted semantic result matched the frozen oracle. First samples are retained separately; the table shows warm medians, four per model/worker.

| Model | TTFT seconds control / candidate | Total seconds control / candidate | Derived tokens/s control / candidate |
|---|---:|---:|---:|
| Qwen3-1.7B | 7.419618 / 2.461594 | 8.645467 / 4.013196 | 27.657614 / 23.515990 |
| Qwen3.5-4B | 20.751181 / 5.832252 | 32.056833 / 15.207581 | 8.178446 / 9.820600 |

4B TTFT-71.894362%, total-52.560562%, derived decode+20.079049%. Text TTFT-66.823176%, total-53.580341%, derived decode-14.974627%. Complete native Skill wall40.707014625 ->19.089722271s (-53.104588%). Old WS10 memory telemetry is unavailable/null, so this stage makes no old-control memory ratio claim. Its source, capabilities and telemetry differ as prospectively declared; no historical cohort is used as this comparison's denominator.

Both workers reported AC power/100% before/after. OS thermal fields are unavailable for this stage; no claim about GPU clocks or lack of throttling. There is no verified exclusion. The valid failure is retained without an unchanged retry. No phase-summing, isolated-cause or statistical-confirmation claim follows from this two-worker screen.

## Decision and limits

REJECT due to text decoding. The raw text storage policy alone did not remove the regression; that observation does not identify the cause. Storage-control20, raw-control20, full48, canonical rebuild/reproof and actual-production mirrored48 confirmation were NOT dispatched. No candidate binary was promoted. The next method requires a distinct, evidence-based design; neither gates nor numerical bounds may be relaxed to admit these attractive VLM numbers.

All28 returned model calls and14 parent requests are exported once. Derived decode=(generated-1)/(total-TTFT), including host/stop/post-loop and possible extra EOS work, not pure GPU throughput. Parent Skill wall is counted once per frame. Fixed video-derived frames traverse actual WorldSense Skill -> Galaxy -> Garnet; these measurements exclude live decoding, scheduling, event persistence/world-state writes, retrieval and ranking. Four normal video files are mounted; the second car is correlated material, not independent-camera fusion. This M5/24GiB/4B result is separate from the unmet M4 mini/small-model/vLLM handover objective.
