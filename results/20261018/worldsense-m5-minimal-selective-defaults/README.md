WS-M5-74: minimal FULL profile with selective graph default retirement
Decision: VALID REJECTION. Keep WS10; full optimization remains active.

The candidate passed all correctness, native pipeline and resource gates. It failed the original 2% generation-throughput preservation gate for BOTH models. Favorable latency and whole-Skill measurements do not override that failure. No unchanged timing retry, matched PRIMARY/RAW comparison, mirrored confirmation, integration, or promotion followed the failure. Canonical evidence prepared for independent source/data/metric reconstruction before publication.

Method and prospective evidence
The design and prior audit were saved before source changes or execution. Source seed: immutable WS71, commit 4c811ec4e305317efac622177d4623b1cf8c83a4. Policy: exact graph-local PRIMARY_ONLY_LAZY_DEFAULTS implementation from WS57, commit 91b1eaded0ce26ad4d03210615626f175f095a1c. WS57 already tested and rejected that policy with direct upload, K32-only prefill and cached metadata enabled. This experiment tests a distinct minimal FULL/vector composition with those extras disabled. The audit found three archived selective role caches and no exact minimal-profile match within its declared scope; it is not a universal history search or a claim that the policy is new. Prior phase spans were descriptive motivation, not proof of a retirement-related cause.

Implementation: 1,721 source files; only CMake, compiled_graph.cpp and metal_main.cpp changed. The three edits import the exact old option/dependencies, graph-local eligibility guard and capability marker. Their inverse recovers all seed bytes. All other 1,718 files are exact, including shaders, math, geometry, codec, weight payload, metadata, loader, Model/session/runtime/xModel/image code, CUDA, engine and original tests. Actual MAIN/prefill/empty CORE programs equal the corresponding seed programs. No grouping2/8 implementation was imported.

Inside the existing locked, once-per-graph retirement planner, after its original validity checks, the policy inspects immutable slot PrimaryReady() values. A graph without packed weights caches an empty plan. Packed graphs execute the complete original alias/weight/layout/liveness safety plan. No model-name, width, checkpoint-name or global-owner heuristic is added; existing Clear/rebuild resets the plan. Retaining raw-text defaults has a memory cost relative to global retirement and can affect the joint profile. No isolated policy speed/memory or physical-residency claim is made.

Six fresh roles: OFF; PRIMARY with primary/wide/lazy/selective; ON adds vector prefill, prefill isolation and FULL isolation; FAULT adds the original primary faults; COMPAT56 and COMPAT56FAULT preserve the old direct/K32/cached/raw-request/global-retirement/fault behavior with selective OFF. ON direct/K32/cached/raw-request/CORE are OFF. RAW-to-candidate is a seven-flag joint contrast. Planned matched PRIMARY-to-candidate differs by three vector/isolation flags, with identical selective retirement, but was not reached. Fourteen-field PrimaryStats and the original five encoded deltas are unchanged; CORE counters are zero. Counters indicate encoded routes, not completed GPU duration.

Correctness before timing
Six builds passed (57027). Native 54 tests passed (35484), independently audited on first execution. All 186 original complete FP32 projection streams and all 20 complete records/432,804 words equal the original references across six roles under unchanged FP64 bounds. Actual GPU program strings and source/import/flag identities passed independent checks.

Preparation 64574 froze explicit source, six libraries, 19 model assets, fixtures, oracles, numerical guards and outer verifiers before model execution; outputs were excluded. Full proof 84026 and independent audit 96431 passed on first execution: 16 subprocess checks, five original GPU readers EACH checking 4,200,857,600 original FP32 words across 201 packed matrices in the actual candidate; 245 raw matrices/1,724,383,232 HOST words; 546 direct-upload matrices/334,609,152 words in compatibility. These coverage categories overlap. Packed original bytes=8,401,715,200; payload+metadata=6,402,163,712; metadata=32,819,200; retained packed+small raw=9,850,930,176. All 64 private resident steps, trained linear/vision/grid/prefix/capacity/error/shared-session/close/rebuild/invalid-input/old 4 GiB raw-request/fault checks passed. Four API modes/20 calls matched original finite states for all 32 VLM and 28 text layers, tokens, outputs and configurations. Only strict storage-policy admission was adapted; recorded inverse operations recover all original mathematical guards and control flow. Raw 1.7B fixture scope is asserted explicitly before expecting zero selective retirement.

Eighteen invalid CMake combinations were rejected (91125): original seven, eight inherited CORE exclusions and three selective dependencies. Offline audit 26592 rejected 22,792 malformed copied results; 65 positive applications reuse 15 actual API calls across five analyzers, plus 10 legacy calls, not new inference samples. Exact selective capability, raw-text zero/VLM-positive retirement, 14-field integer schema, original five deltas, CORE zero, forbidden grouping and legacy admission are checked. All five original quantitative/golden/metric/threshold ASTs are unchanged outside native_route.

Actual WorldSense/Galaxy results
Functional worker 6700 and independent audit passed four calls/two identical fixed-car Skill requests against complete original semantics, generated tokens, text, prompt, configuration, frame/Skill identities, routes, native image backend and memory/source/library identities. Text calls had zero retired/planned/rehydrated defaults while 201 VLM weight owners remained loaded. VLM calls retired 1,152,909,312 bytes over 288 planned slots, with zero rehydration. This proves graph-local policy admission in the joint worker, not a latency cause. Functional timing is ineligible for speed comparison.

Resource preparation 68634 pinned 1,875 compiler/SDK/host dependencies and 17 existing outer helpers before timing. Resource worker 49076 and independent audit passed eight calls/two fresh workers, one first and one follow-up request each. Original gates require at least 10% sampled Metal allocation reduction and 5% sampled process-footprint reduction versus RAW. These are maxima of post-decode snapshots, not peaks/free/unique/physical-residency RAM. Whole-profile reductions:
{
  "metal_allocated_bytes": 0.1437814726272294,
  "task_physical_footprint_bytes": 0.07849696114845583
}

Speed preparation 43419, actual deployed WS10 stage 24471 and independent audit 63491 passed all identity/output/admission checks; the speed decision is a valid rejection. Twenty calls/two fresh sequential workers/five identical fixed-car Skill requests each. The first is retained separately; the following four per model/worker form these medians:

Model | TTFT seconds, WS10 -> candidate | total seconds | derived tokens/s
Qwen3.5-4B | 12.641953457991 -> 4.419677750004 | 19.912589437503 -> 11.830159604491 | 12.591864478096 -> 12.245852433079
Qwen3-1.7B | 2.092476436999 -> 1.833302582992 | 3.352938104013 -> 3.150623228998 | 27.391596816502 -> 25.816543293434

Within-current-pair changes:
[
  {
    "model": "Qwen3-1.7B",
    "ttft_change": -0.12385986739161214,
    "total_change": -0.0603395794195255,
    "decode_gain": -0.057501340050341576
  },
  {
    "model": "Qwen3.5-4B",
    "ttft_change": -0.6503959799654137,
    "total_change": -0.40589546921453845,
    "decode_gain": -0.027479015964548714
  }
]
Whole Skill wall medians: {"control-1": 22.91303204151336, "candidate-1": 14.986481312502292} seconds; gain=0.3459407168221956.

Both derived-throughput losses exceed the unchanged 2% gate: 4B -2.747902%, text -5.750134%. TTFT/total and whole-Skill were favorable, but no subset, memory, correctness or pooled rescue is allowed. Legacy WS10 memory/counters are unavailable/null, never invented zero or replaced by a resource-cohort denominator. All 32 actual pipeline calls/16 Skill requests are retained across functional/resource/speed stages, including all first/follow-up measurements, with 22 available memory snapshots. The independent screen audit rechecked all 1,721 source files, six libraries, 19 assets, runtime/drivers/dependencies and frozen inputs. There were no failed builds, native tests, model calls or timing retries in this experiment.

Normal profiles remain VLM64/text16, capacity2048, image65536, shared vision and text prefix64tokens/2entries/64MiB, with original models, budgets, limits, stops and Skill thresholds. Short numerical proofs use declared 512 VLM/256 text capacities. Each request freshly prepares/encodes its image in native C++/Metal/XLang3. No external CPython inference/image processing, image/answer cache, speculation or camera. CPython is used only for orchestration/audits. During timing only bounded observation ran; no builds, competing investigation GPU, hash sweeps, archive copies or consumed edits. Every controller restored WS10/four video files/no camera/valid normal benchmark POST404 in finally; last speed restoration PID57541.

Interpretation and remaining work
Derived throughput=(generated-1)/(total-TTFT), including host/stop/post-loop/possible EOS/snapshots, not pure GPU throughput. GPU/wait/encode spans overlap and separate medians do not add. First requests are not hardware-cold. No significance, clock, thermal, VM/cache, ISA, register, occupancy, physical-residency or isolated-causality claim is supported. Fixed-frame Skill benchmarking excludes live video decode/scheduling, persisted events/shared states/dedup, retrieval/ranking, navigation and independent-camera fusion; the second car source is correlated. This M5/24 GiB/4B run does not satisfy the separate M4mini/small-model/vLLM handover.

Keep WS10 and the full goal active. Archive/reconstruct/audit/publish this valid rejection. Then inspect retained per-call decoding phases and exact earlier native decoding methods before a distinct design; do not repeat this unchanged profile or infer a root cause from historical ratios. No successor source/result is established here.
