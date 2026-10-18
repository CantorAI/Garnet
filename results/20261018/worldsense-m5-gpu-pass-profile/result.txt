# WS-M5-62 — Explicit WorldSense decode GPU-pass diagnostic

Complete diagnostic, not an accepted optimization. Four actual WorldSense → Galaxy → native Garnet calls match the original outputs; all 12 predeclared decode windows join exactly to native model/request/step IDs. The capture contains 7,716 native compute-pass intervals. Selected WS10 remains deployed; prior WS61 remains a valid rejection and full optimization remains active.

## Question and prior evidence

WS61's exact old library in a native child still had longer inner decode/GPU spans; gateway overhead alone did not explain its loss. Existing WS26 per-token timing located the loss across tokens but not individual native operations. Historical sparse counter traces targeted M4/small models and lacked explicit phases; their times are not an M5/4B baseline. Encoder reuse, concurrent projections and disjoint-grid fused projections already have rejected/mixed evidence and were not repeated unchanged.

This new diagnostic asks which custom compute-pass shapes dominate prespecified decode windows of the packed 4B/raw-text profile in real WorldSense. [Design before implementation](native-design-before-implementation.txt). It does not determine GPU clocks, bandwidth, occupancy, residency or a root cause; it does not test a new inference optimization or qualify a rejected profile for promotion.

## Native instrumentation

Immutable WS54 seed has 1,718 files. Isolated source has 1,720 files/seven changed or added: Metal CMake, context header/implementation, native entry, serving metadata, new counter helper and native test. All original shader literals, graph/session/xModel/image/CUDA code and original tests remain byte-identical; reversible metadata transformations are recorded. Actual constructor MAIN and prefill shader texts match the seed in all five builds. Source equality is not compiler ISA identity.

Default-OFF GPU_PASS_COUNTER_PROFILE. Native InferJson owns a thread-local, nest-restoring RAII request scope with unique positive ID and explicit model argument. The existing XLang3 serving loop marks index+1 immediately before the unchanged decoder.step and returns the request ID. Requests restore on return/error; no scope reaches other threads, prefill/vision/status or the next request. Direct serving calls without a native request keep their original inference and remain untraced.

Only decode steps 4,16,28 are sampled. All original custom compute encoders in those commands use timestamp attachments; no new empty encoders, forced flushes, extra GPU waits or reordered dispatches. One 4,096-entry/32 KiB timestamp buffer per active context bounds each selected command to 2,048 encoder intervals. Unsupported API, capacity overflow, missing/wrong-length/zero/error/reversed timestamps and partial-command sampling fail the diagnostic instead of fabricating zero or truncating. Resolve and append/flush records after the original completed submission. FILE/descriptor/JSON/sampling work perturbs execution; this is not a serving performance build.

The native M5 capability probe found timestamp set/stage-boundary support, with dispatch/blit sampling unsupported. Two dependent passes produced all 4,096 expected values exactly and four valid raw timestamps. No checkpoint, image or model inference in that probe. See Apple's [GPU counter overview](https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers) and [compute-pass start index](https://developer.apple.com/documentation/metal/mtlcomputepasssamplebufferattachmentdescriptor/startofencodersampleindex); installed SDK headers and hashes identify the compiled API.

RAW has all nine prior method flags OFF and tracing ON; PRIMARY has PRIMARY/WIDE/LAZY ON and tracing ON; packed has all nine ON and tracing ON. Quiet has the packed flags with tracing OFF. Fault has the packed/tracing flags plus test hooks. RAW/PRIMARY/packed/quiet are numerical compatibility profiles; the actual pipeline capture is candidate-only packed, as declared. There is no RAW timing comparison or speed-selection gate in this diagnostic. Existing optimization resource/speed/confirmation/canonical gates remain unchanged for a later distinct candidate.

## Correctness and retained setup history

Five builds/40 native tests passed. Review then found that the first default-OFF wrapper still evaluated label arguments; initial source/manifests/logs were preserved. An OFF macro now discards those arguments. All five builds and all 40 native tests were repeated before the first learned-model call. Preprocessor proof confirms original encoder calls without diagnostic label construction in the OFF projection path. A separate prebuild include-path correction for the existing JSON header was preserved before any first build. These are deterministic numerical repetitions, not independent speed trials.

Final native tests cover original math/186 complete projection records, exact dependency values, valid/error timestamps, nested request restoration, thread isolation, unscoped rejection, selected/unselected steps, Abort/readability/retry, capacity overflow, simulated unsupported API and post-completion resolution failure. Actual MAIN/prefill sources and every projection word match the seed in all five roles. Fault hooks are absent from measured/captured libraries.

Full learned proof passed: five packed GPU readers each check 4,200,857,600 FP32 words over 201 matrices; 245 raw buffers check 1,724,383,232 HOST words, retained packed+raw bytes 9,850,930,176. Direct corpus checks 546 weights/334,609,152 words. Full 32 VLM/28 text state hashes and tokens, trained generic/resident/private sessions, shared owners/close/rebuild, inactive/invalid controls, vision grids/embedding hashes, prefix/capacity/reset/recovery match original oracles. Three traced API roles each return five exact state/token calls and five explicit sampled commands; quiet matches outputs/states with no counter metadata/file. A real learned text request failed at counter resolution, cleared scope, then recovered all 28 states/tokens exactly after reset.

Source/model/runtime/library/mathematical guard/oracle identities were hashed before learned calls and checked afterward. The outer service/test sequencing wrapper's first explicit SHA was observed after launch; its creation was recorded before launch and it stayed unchanged. That timing limitation is explicit; no prelaunch SHA record for the outer wrapper is claimed. The later actual pipeline capture froze the full outer/capture/verification driver closure before dispatch, plus all 19 assets and 1,877 compiler/build/SDK/host/backend dependencies. Actual capture aliases/file bytes remained unchanged. No source edits, builds, competing investigation GPU work or archive copying during capture.

## Real pipeline capture and findings

One fresh native Cantor process PID40391; two identical recorded car Skill requests, first and followup, four model calls. The original 2,048 capacity, 64-token VLM/16-token text chunks, 65,536 image-pixel budget, shared vision, bounded 64-token text prefix policy, JPEG/Skill/prompt/output caps/stops and full generated outputs remain. Every image is freshly prepared natively; no image/answer reuse or speculation. GPU trace file is scoped to that owned process through a temporary launchd plist; original plist is untouched, and trace bytes are copied only after the process stops.

VLM requests have 94 decode steps and text requests 35; each reaches 4,16,28. Exactly 749 custom passes per sampled VLM command and 537 per sampled text command. All four original goldens/native capability/storage/prefill/count/source identities match. No additional model call or phase inferred from sparse missing shapes. [Raw records](capture/candidate-1.counters.jsonl), [encoder CSV](capture/encoder-samples.csv), [per-window/aggregate analysis](capture/counter-analysis.json), [independent capture verification](capture/verification.json).

The table ranks sums of sampled interval durations in raw timestamp units. It is NOT a complete GPU time fraction, an end-to-end contribution or a comparison to an untraced run. First and followup remain separate.

| Model | Request | Native operation/shape/representation | Passes across steps4/16/28 | Share of sampled interval-duration sum |
|---|---|---|---:|---:|
| Qwen3.5-4B | first | `bf16:1:2560:9216:primary` | 192 | 31.9892% |
| Qwen3.5-4B | first | `bf16:1:9216:2560:primary` | 96 | 16.1271% |
| Qwen3.5-4B | first | `bf16:1:2560:8192:primary` | 96 | 14.2509% |
| Qwen3.5-4B | first | `bf16:1:2560:248320:primary` | 3 | 12.8448% |
| Qwen3.5-4B | first | `bf16:1:4096:2560:primary` | 96 | 7.6175% |
| Qwen3.5-4B | followup | `bf16:1:2560:9216:primary` | 192 | 31.8228% |
| Qwen3.5-4B | followup | `bf16:1:9216:2560:primary` | 96 | 15.9659% |
| Qwen3.5-4B | followup | `bf16:1:2560:8192:primary` | 96 | 14.3120% |
| Qwen3.5-4B | followup | `bf16:1:2560:248320:primary` | 3 | 13.4466% |
| Qwen3.5-4B | followup | `bf16:1:4096:2560:primary` | 96 | 7.5587% |
| Qwen3-1.7B | first | `bf16:1:2048:6144:raw` | 168 | 34.3464% |
| Qwen3-1.7B | first | `bf16:1:6144:2048:raw` | 84 | 17.6735% |
| Qwen3-1.7B | first | `bf16:1:2048:151936:raw` | 3 | 14.5105% |
| Qwen3-1.7B | first | `bf16:1:2048:2048:raw` | 168 | 12.1142% |
| Qwen3-1.7B | first | `bf16:1:2048:1024:raw` | 168 | 6.0801% |
| Qwen3-1.7B | followup | `bf16:1:2048:6144:raw` | 168 | 34.1400% |
| Qwen3-1.7B | followup | `bf16:1:6144:2048:raw` | 84 | 17.6314% |
| Qwen3-1.7B | followup | `bf16:1:2048:151936:raw` | 3 | 14.6342% |
| Qwen3-1.7B | followup | `bf16:1:2048:2048:raw` | 168 | 12.1204% |
| Qwen3-1.7B | followup | `bf16:1:2048:1024:raw` | 168 | 6.0765% |

For the text followup, gate/up and down projections together are 51.7714% of the sampled interval-duration sum; vocabulary head 14.6342%. For the VLM followup, those MLP projections are 47.7887%, vocabulary head 13.4466%. Native projection/data-movement work is a better next investigation target than repeating host QoS or encoder reuse unchanged. This is a workload-specific prioritization inference, not proof of a new layout's benefit.

Timestamp intervals overlap even though original command ordering is retained: all 12 windows have positive sum-minus-union overlap. Analysis stores the sum, merged union, overlap and span/gaps separately. Raw timestamp units remain raw; no guessed conversion to milliseconds. Original command GPU span in ms is separate and is not mixed with raw units. MPS-owned encoders may be absent; uncovered work/gaps are not assigned to a kernel or called pure CPU time. Ranks describe these three selected steps, not all 94/35 decode steps or an unbiased throughput distribution. One first and one followup request provide no significance or independent performance confirmation.

## Limits and next work

Capture timings remain raw JSON but are ineligible for optimization acceptance. Default-OFF compatibility and numerical equivalence do not establish speed. Fixed-frame currentSkill coverage excludes live video decode/scheduling, persisted events, shared-state writes, retrieval/ranking, navigation and independent-camera fusion. CPython orchestrates/audits only; native C++/Metal/XLang3 perform inference and image preparation. No camera, external CPython image/model helper, CUDA source removal or engine/SDK change. This M5/24 GiB/4B work is separate from the unmet M4 mini/small-model/vLLM handover.

Next read-only direction: audit every raw weight reader before designing sole interleaved storage for the two text MLP shapes. Historical interleaving retains a second buffer plus row-major master; no gain or memory saving is imported from those cohorts. A sole representation plus bounded native GPU restoration for unchanged prefill arithmetic is a prospective possibility, not implemented or accepted here. Any later candidate needs its own design, all original learned/numerical/ownership proofs and fresh unchanged performance gates.

Normal selected WS10 verified PID40565, four ordinary video files online, camera unmounted, valid benchmark POST404. Goal remains active; no optimization promotion or global completion claim.
