WS-M5-66 result — shared-activation sole gate/up decode with separate outputs: VALID REJECT

Keep validated WS10; full optimization remains active. Native/full learned/realWorldSense correctness and resource gates pass. In this actualWS10 paired cohort, text deriveddecode is preserved(+1.069349%), but textTTFT+29.449562% andtotal+12.754560% fail unchanged2%preservation. No unchangedretry/laterstages/promotion. Different cohorts have different absolute timings; no historical control ratio or cross-cohort causal claim. M5/24GiB/4B plus companion1.7B remains separate from unmetM4mini/smallmodel/vLLM handover.

## Prospective method and exact scope

[Design before source/build/model](native-design-before-implementation.txt), [prior-method/consumer audit](read-only-dual-consumer-audit.json), [protocol](protocol.json). Immutable64 seed1721files from publisheda6924be03a4b36a3472677d17f0d16a4f8bdfb5a; candidate1723files/eightdelta. Use original64MLP-only sole layout/directprefill; do not import65attention/head expansion or reconsider its rejection.

Earlier20261013 already tested shared-activation dual-accumulator gate/up+SiLU on smaller models and rejected broad promotion. Later raw/compact grouped projections used disjoint grids and had mixed/rejected outcomes. This distinct candidate uses exactM5/1.7B sole-layout/function-constant pairing, preserves BOTH complete intermediate output matrices and leaves original SiLU/down operations separate. No claim that dual accumulators are previously untried or that component timings select it.

Default-OFF GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE requires soleMLP+directprefill and existing explicit auxiliary exclusions. Pair adjacent same-input transposeBF16 operations only when both resident or packed-output, !profileOps, distinct outputs. Context admits same-device soleN6144/K2048/B1or2/defaultdense strides, checks representations/lengths/inputinitialization/pendingownership/outputaliases before mutation. Unsupported shape/layout/capability returnsfalse with no mutation; eligible invalid batch/ownership/alias errors throw. Graph skips ONLYsecondprojection, retains both outputs and original downstream SiLU. Ordinary/profilediagnostic routes remain. No new queue/flush/wait/scratch/weightcopy/concatenation/speculation/image-answercache/model precision/budget change.

New separate library combines exact original sole program with generated dual fragment. B1/B2 function-constant pipelines use original32lane/128thread geometry and capability limits. The two original FP32 dot/reduction/store orders are retained independently while sharing x/y activation loads. Integer helper uses original two uint4 BF16 loads, reused by new correctness-only paired GPUreader. Source inverse proof is not ISA/register/rounding/performance equivalence. Original MAIN/primaryprefill/sole strings and all eight original tests are byte-exact64. Graph/context have explicit new guarded/inverse-audited blocks; inherited all-graph-unchanged field is explicitlyFALSE, not relied upon. Session/xModel/image/CUDA/engine/SDK unchanged.

Native stats encoded_pairs count physicalpaired dispatches; represented_projections=2*pairs. Original sole.decodeEncoded remains ordinary single-projection dispatches. Physicaltotal=ordinarysingles+pairs; representedtotal=ordinarysingles+2pairs. Separate nativebefore/after/delta/marker proves ONtextpositive, VLMdelta0 and disabled/compat counterszero. No new weight/scratch charge. Unavailablepair andpredispatchfault hooks only separate correctnessbuild.

Costs declared before source: extra accumulators/register pressure/occupancy risk, two weightstreams/vectorhelper address work, graph/context CPUchecks, new library/PSO/init work. Potential fewer encoder/activationloads may not outweigh them. Original shared84textMLP buffers2,113,929,216B and retained packed+raw9,850,930,176B unchanged. Native C++/Metal/XLang3 inference/images; CPython orchestration/audit only. No camera/externalCPython image or model execution.

## Build, native and learned proof

Firstbuild85334 terminal1: OFF/PRIMARY passed, ON stopped on new constructor local-read name colliding with existing sole reader. Preserved logs/first source/manifest/generator/proof; renamed only new local todualReaderPipeline AFTER terminal, inversechecked and rebuilt allsix73705terminal0. No GPU/native/model attempt before repair. Six correctnessroles RAWoff/minimalPRIMARY/expandedON/separateFAULT/compatRESTORE(soleONdirectOFFdualOFF)/compatDIRECT64(soleONdirectONdualOFF); compatibility roles never timed.

Native42888terminal0:54tests/nineperrole. Original186projection+28MLP complete output records unchanged and bitwise equal across six. New36complete separate gate/up records: B1/B2,two distinctweightpairs,four signed/scaled sparse inputs plusdense reference cases. All original per-output FP64 bounds and complete control/candidate bytes pass. Newstream1,328,832bytes/SHA256 63c2515a1d5a24f115d41030f51bd785fe4a5f74d6cbfd0b56f92ed5a57b4e13. Aliases/malformed/uninitialized/foreignpending/unsupported/ranges/activebatchreadback/unavailable/partialFlush/Abort/recovery/source-release/peer/fault/RAII pass. Actual oldshaderstrings exact; new dual program equals declared separate concatenation.

Learnedpreparation84766terminal0 freezes source/runtime/libs/19assets/math+outer/verifier BEFORE first learned launch. Full64144terminal0 andindependent91504terminal0 pass all original fivepackedGPUreaders each4,200,857,600FP32words/201matrices,245rawlogicalHOST1,724,383,232words,546directweights/334,609,152words,84sole scalar+eachthree tile GPUreaders1,056,964,608words. New SAME dualvector helper GPUreader checks all28actualgate/up pairs/56matrices/704,643,072originalFP32words; compares independently mapped original checkpoint bytes and verifies owner/counter restoration. Coverage overlaps existing readers; not unique-word sums.

Original real1.7B checkpoint resident fixtures exercise B1/B2/layer0, not toy weights. Twelve original/compat/profile fixture runs prove408actualgeneric/resident routes, exact explicit/private/shared state, late secondsession/sourceclosure/rebuild/input/error behavior. ONnormal pairspositive once/layercall; diagnosticprofile mode retains original separate ops/pairzero. All32VLM/28text layer state/token/vision/shared/private/prefix/capacity/lifecycle oracles pass; five actualnative API modes/25calls preserve complete original states/tokens/configurations and required paircounter minimums. Guardcapacities512VLM/256text, realWorldSense2048. Full learned proof supplements synthetic math tests.

Twentythree expected config exclusions and14,330malformed offline copiedcall fixtures rejected; copies are not newinferences. First offline pipelinepreparer failed inherited two-branch count after adding a third disableddual-counter branch. Preserve failed generator/log/partialanalyzers; fix count only and retain strict legacy exclusion in original firstbranch. No scientific source/oracle/threshold relaxation or benchmark retry. All five original golden/numeric/metric/threshold ASTs inverse exact outside native admission. Four actualWorldSense/Galaxy functional11039 calls exact, ineligible for speed.

## Original resource gate

Resource75810terminal0/independent audit:8actualcalls, freshRAW thenON, onefirst+onefollowup perworker. Maximum postdecode Metal21,903,376,384 ->16,870,899,712B(-22.975803%); taskfootprint18,073,255,392 ->16,559,831,080B(-8.373833%). Original10%/5%gates pass. Whole packed/lazy/direct/dual profile versus freshRAW, not isolated pairing memory gain. Snapshots not peaks/free/unique/residencyRAM.1,880compiler/SDK/hostdependencies and aliases across sixbuilds, paired proof preprocessor dependencies, source/libs/drivers/19assets unchanged. Supplemental cohort build-artifact copies cover four roles; complete six-role native manifests/logs are also retained. Fifteen existing method/outer helpers pinned before resource, unchanged afterward and after speed. No builds/competingGPU/consumed edits/archivecopies during timing; bounded observations only.

## Actual deployed-WS10 speed screen

Separate predeclared9149terminal0:20exact/native calls. Two fresh sequentialworkers, five requests each, first retained separately and four warm samples/model/worker. Original2048capacity,VLM64/text16chunks,image65536,sharedvision,bounded64token/2entry/64MiBtextprefix policy. Fresh native image preparation/encoding; no image or answer reuse. Warm medians:

|Metric|DeployedWS10|Candidate|
|---|---:|---:|
|Qwen3.5-4B TTFT(s)|4.255353|3.827225|
|Qwen3.5-4B total(s)|11.399027|9.934489|
|Qwen3.5-4B deriveddecode(tokens/s)|13.018517|15.158625|
|Qwen3-1.7B TTFT(s)|1.438878|1.862622|
|Qwen3-1.7B total(s)|2.807990|3.166137|
|Qwen3-1.7B deriveddecode(tokens/s)|26.580051|26.864284|
|CompleteSkillwall(s)|13.996643|12.915799|

TextTTFT+29.449562%,total+12.754560% fail unchanged2%preservation although deriveddecode+1.069349% is preserved.4BTTFT improves10.060918%,total12.847918%,deriveddecode16.438957%; wholeSkillwall7.722169%. Valid REJECT; favorable4B/decode/Skillmetrics do not override textlatency failure. No unchangedretry/later matchedPRIMARY20/RAW20/48mirrored/canonicalintegration48/promotion. Canonical paper archive is not implementation promotion. Legacy speed-control memory unavailable/null, never borrowed from separate resource control.

Text control-1: prefill-loop1.295097917s,prefillGPUspan1034.723937511ms,decode-loop1.253965292s,decodeGPUspan1224.523520810ms,per-call postdecode residual median0.008430833s.
Text candidate-1: prefill-loop1.610339687s,prefillGPUspan1189.104562523ms,decode-loop1.263799459s,decodeGPUspan1229.909208298ms,per-call postdecode residual median0.000760917s.

[All per-call phase/count data](posthoc-phase-diagnostics.json). These motivate a read-only prefill/encode investigation; they do not establish isolated pairing, layout, clock, thermal, VM, residency, ISA or cross-cohort causes. Preserve current cohort rather than importing earlier absolute controls. Deriveddecode=(generated-1)/(total-TTFT) includes host/stop/postloop/possibleEOS/snapshots, not pureGPU. Per-call differences precede medians; separate medians nonadditive, spans overlap/include gaps. First nothardwarecold; old readiness cannot separate init/model-load costs. No significance/bandwidth/root-cause claim.

Fixedframe/currentSkill excludes live video decode/scheduling/event persistence/sharedstates/retrieval/ranking/navigation/independentcamera fusion. All original method/golden/state/token/ownership/resource/speed/confirmation/canonical gates remain. Complete first/warm/failures/identities retained. Independent1748terminal0 recomputes32realWorldSense calls/16Skillrequests/resource/speed/rejection and rehashes source/libs/runtime/models/1,880dependencies/15outerhelpers. NormalWS10 restoredPID78808/fourfilesonline/noCam/validbenchmarkPOST404. Nojobsactive.

## Continuation

Preserve66 valid rejection and prior64/65 rejections. Archive/audit/publish, then read-only audit of the original cooperative prefill-load order against sole interleaved addresses and retained prefill/encode boundaries before a DISTINCT prospective method. No new successor source/design or gain is established here. Full optimization remains active, no global optimum/completion claim.
