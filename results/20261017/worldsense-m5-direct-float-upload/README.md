# WS-M5-50 — Direct FP32 weight upload rejected on text decode preservation

Complete correctness and the raw resource screen passed. The matched primary-only screen preserved memory and improved4B TTFT18.905837%, but text derived decoding fell17.236447%, exceeding the unchanged2%limit. Overall REJECT. No later speed/confirmation/canonical stages, unchanged retry or promotion. Terminal logs restore WS10/fourfilesonline/noCam/validbenchmarkPOST404.

## Prospective method

Design preceded setup/build. FrozenWS49seed1716files verified; sixfiles change from49 and sevenoverlays reconstruct canonicalf21c1a2584031ee9d1c129d1e58818f9d1c826d2. DefaultOFF Metal-only DIRECT_FLOAT_WEIGHT_UPLOAD adds native C++ UploadBF16AsFloat. The BF16 branch of MetalCompiledGraph::Weight synchronously expands originalBF16bits directly into the final shared FP32 MPS buffer instead of first allocating a full std::vector<float> and copying it. Original F32 fallback and all metadata/shape/filebounds/filemodification/shareguards retained exactly. Same Allocate descriptor/rowstride/maxbuffer/device checks and ready semantics. Nonnull/exactlength/Elements overflow/closedbatch checks precede allocation. Perword unalignedmemcpy→integer<<16→rowstride-awarememcpy, no floating arithmetic/finitefilter. Native owner survives mapping/source close and aliases. No new autoreleasepool, eviction, wait, lazyretirement, cache growth or decoded master.

The49duo loader and minimal secondlibrary are unchanged, including all actual shader program bytes. Both models full-primary/eager; LAZY/CACHEDOFF. Raw/candidate differFOURflags(primary/vector/isolation/directupload), matched priorprimary/candidateTHREE(vector/isolation/directupload), not a component-only causal experiment. New direct_float_weight_upload marker plus frozen source/PSO/asset/native tests supports route identity; no new completedGPUcounter is claimed. Model/wrapper/imagepreprocessing/CUDA/primarycodec/originalprojectiontests/profiles/budgets/boundedprefixpolicy unchanged. C++/Metal/XLang3 inference/images; CPython orchestration/audits only.

Motivation was49's transient firstimage-input footprint, not demonstrated retained-library cost. Removing staging is source-proven; a particular memory or speed gain cannot be attributed to it from this joint contrast.

## Correctness before timing

Four fresh raw/priorprimary/candidate/fault builds passed seven native tests each and186complete original projection records exactly. New directmethod checks ALL65536BF16patterns including signedzero/subnormals/nonfinitepayload bits without GPU arithmetic over those values;257x257paddedrows/unalignedsource/source mutation/alias lifetime/disabledOFF/invalidnull-byte-size-zero-overflow/openbatch/Abort recovery passed. Existing primary allocation-fault tests remain; no new injected direct-allocation failure coverage claimed.

The native direct-upload/download corpus checked334,609,152originalFP32words over546selected learned BF16 vision,1Dnorm/bias,depthwise/patch raw-weight candidates from both checkpoints. This is a metadata-selected superset of actual accesses, not a claim everycandidate is used by these prompts. Fullnames/shapes/BF16hashes/wordcounts retained. These host-shared-storage checks are distinct from the FIVE GPU readers, each checking5,921,308,672originalFP32words over398packed matrices;48rawfall backs, retainedprimary+smallraw9,033,678,080B.

Full32VLM/28text states/tokens/trainedlayers/vision/grids/shared-after-close/host-private/session/rebuild/invalid/prefix/capacity/isolation passed in raw/priorprimary/candidate actualAPI modes, five fullrecords each. Original numerical/golden limits unchanged. Loader inverse recovers ENTIREimmutable original Weight function except materialization block; originalF32fallback body exact. Actual fourbuild main/prefill programs byteexact49 (main priorprimary/candidate/fault SHA2562b29178d6b2f3ec30db757766dd1274845711eccf573e78497ee1f8828afdb81; candidate minimalprefill8746bytes). Source/program identity does not prove GPUISA/performance.

Three quantitative/golden/source analyzerASTs unchanged except nativepredicate,15actualAPIrecords/negative capabilities/owners/lazy/dispatch guards pass includingdirectmarker. Matched commonmetricprefix unchanged; prospective>=5%VLM TTFT and<=2%model-wall-memory gates retained. Complete source/build/library/checkpoint/runtime/cases/frame/Skill and1870build/SDK/host/correctness files including bothimmutableCMakeprofiles pinned and unchanged afterbothstages.

## Fresh WorldSense resource gate

Eight exact/nativevalid raw-control/candidate calls. Postdecode sampled maxima: Metal21,903,376,384→19,091,357,696B (-12.838289%); footprint18,057,705,848→15,260,087,808B (-15.492655%). Snapshots notpeaks/freeRAM; storage is part of this contrast, resource timings ineligible for speed evidence.

## Fresh matched primary-only speed screen

Two freshworkers/fivesingle-carrequests each,20modelcalls. Firstrequests retained; warmmedians four samples/model/worker. Complete outputtokens/text/prompt/effectiveconfig/image/Skill/interpretedsemantics match frozenoracle.

|Model|TTFT seconds control /candidate|Total seconds control /candidate|Derived tokens/s control /candidate|
|---|---:|---:|---:|
|Qwen3-1.7B|2.302535 /1.779959|3.662067 /3.424180|25.008867 /20.698227|
|Qwen3.5-4B|6.671285 /5.410023|16.050599 /15.055490|9.915570 /9.812571|

4B TTFT improved18.905837%, total6.199828%, deriveddecode-1.038759%. Text TTFT improved22.695701%, total6.495997%, deriveddecode-17.236447%. NativeSkillwall19.869675062→18.374688896s (-7.523959%). The textdecode failure rejects the wholemethod; other benefits do not override it.

Matched sampledMetal stayed19,091,357,696B; footprint15,752,197,968→15,262,316,104B (-3.109927%), satisfying memorypreservation. Do not compare these ratios with49's differentcontrol or classify its rejection differently. Bothworkers AC100% before/after; noOSthermal/performancewarning recorded. This is not directGPUclock/temperature or proof ofno throttling; no verifiedexclusion.

First afterinitialization VLM TTFT12.233939→7.404222s; visionphase1.853153→2.597005s; prefill10.165682→4.752912s. These different phases and warmresults are nonadditive/descriptive, not isolated uploadtime or hardware-cold measures. Allfirst/warm/raw data retained in CSV/JSON. See posthoc-observations.json for fullphase/memory/chunkplans. Warm textplans all16chunks; VLM six64chunks+one16tail forcarcase.

## Decision, development repairs and next hypothesis

REJECT/no unchangedretry/laterstages/promotion. All28modelcalls/14parentrequests retained; parentSkillwall once/frame. Deriveddecode=(generated-1)/(total-TTFT) includeshost/stop/postloop/possibleextraEOS, not pureGPUthroughput. Snapshots notpeaks; owner/encodedcounters are not physicalresidency/completedGPUcounts. Fixedframes traverse WorldSense currentSkill→Galaxy→Garnet, excluding livedecode/scheduling/eventpersistence/worldstatewrites/retrieval/ranking. Four normalfiles restored, secondcarcorrelated notindependentfusion. M5/24GiB/4B separatefromunmetM4mini/vLLM. No consumededits/build/competingGPUinvestigation/archivecopies duringtiming. Canonical/reproof/actualWS10pairedconfirmation+liveappsmoke stillrequired beforepromotion.

Initial fourbuilds succeeded; source review before FIRSTnative found the newtest expected fp32 instead of existingAPI label matrix_fp32. Corrected onlytestlabel, retained initialsource/logs, rebuiltallfour and ran allproofs. No numerical/performance test repeated or gate changed. Post-terminal selective objdump symbol lookup missed the MachO spelling; warning/command retained, successful full-disassembly extraction plus relocation then proved the measured-object innerloop5f1c..5f3c contains rowBytes ObjectiveCcall at5f24. Read-only objectSHA/commands/175linefunctionextract/relocation retained. This proves a perword host call, not its cost or the cause of warm textdecode regression.

A distinct future method may hoist rowstride and use contiguous CPUconversion loops, and restrict serving duo-prefill to K32geometry while restoring the original K64 16-token kernel used by text. Current native dispatch selects wideK64 atrows16 and wideK32 above16; actualchunkplans corroborate. This targets avoidable loading work and preservation of the textprefill implementation while retaining most4B64chunk work. No benefit or causaldecodefix is established. Newprospectivedesign/fullproof/freshunchangedgates required; not implemented here.
