# WS-M5-51 — Hoisted upload and K32-only prefill rejected on text preservation

All correctness and the raw resource screen passed. The matched primary-only screen improved4Bfirst-token latency12.684251% and preservedmemory, but textTTFT+27.320514%, total+13.121473%, decode-3.371765% failed unchanged2%limits. OverallREJECT/no laterstages/unchangedretry/promotion. NormalWS10/fourfilesonline/noCam/validbenchmarkPOST404 restored.

## Prospective joint method and source

Design preceded source/build/execution. Frozen50seed1716files verified; onlycontext,entry,CMake differfrom50. Sevenoverlays reconstruct canonicalf21c1a2584031ee9d1c129d1e58818f9d1c826d2. No MSL/primaryCPUcodec/Model/wrapper/images/CUDA/originalprojectiontest/profile/budget/boundedprefixpolicy change. Fullprimary/eager bothmodels; LAZY/CACHEDOFF. RawcandidateFIVEflags, priorprimary/candidateFOUR(vector/isolation/directupload/K32ONLY), not isolated componentcausality.

The direct-upload method now reads immutable rowBytes once before conversion, computes rowinput/output pointers, then performs identical unsigned BF16word<<16 bitstores. Same Allocate/rowpadding/guards/ready/owner, no newpool/cache/eviction/floating arithmetic. The measuredONCPUobject has one rowBytes getter outside allchecked backward branches (b,b.cond,cbz,cbnz,tbz,tbnz);23static vectorshiftinstructions appear in the wholefunction. ObjectSHA/commands/functiondisassembly/relocation/backedges retained. This proves codegeneration/staticinstructionpresence, not executed SIMDcount, componenttime or speed. Initial b-only parser retained; analysis broadened toall6families beforetiming, no source/native/performance rerun.

New defaultOFF PRIMARY_VECTOR_PREFILL_K32_ONLY requires vectorprefill. It creates primaryGemm16Wide from originalMAIN primary_gemm_16_64_64, retaining generic/wideK32duo from minimalprefilllibrary. No extraPSOs. Same actual MAIN/prefillprogrambytes as50; the unused K64duo remainscompiled for the originalthreegeometry diagnostic. Host dispatch atrows16 usesoriginalK64, above16 usesduoK32, alladmission/threads/tiles/math unchanged. Actual carplans six64+16 forVLM, nine16 fortext. VLMtail16 alsofallsback. The newmarker/flags/frozenconstructor+chunkplans+existingprefillcounter support route identity, not completedPSOcounters or a causaldecodeexplanation.

## Full proof beforeperformance

Fourfresh raw/priorprimary/candidate/fault builds passed all7native tests each and186complete originalprojectionrecords exactly. All65536BF16pattern/unaligned/padded/alias/source-lifetime/null/byte/zero/overflow/openbatch/Abort/OFFdisabled checks passed; existingprimaryfaulttests remain, no newdirectallocationinjectionclaimed. Native direct-upload/download checked334609152FP32words over546selectedlearnedBF16vision/norm/depthwise/patchrawcandidates (metadata-superset, not everycandidateusedbythisprompt). FiveGPUreaders EACHchecked5921308672originalFP32words over398packed matrices;48smallraw, retained9033678080B. Hostsharedwordchecks are distinct fromGPUreaders.

Full32VLM28text states/tokens/trainedlayers/vision/grids/sharing/private-host/session/rebuild/invalid/prefix/capacity/isolation passed raw/priorprimary/candidate actualAPI modes, fivefullrecords each. Originalbounds/goldens unchanged. EntireWeightloader inverse/F32fallback retained; source/actualallfourbuildMSLprograms exact50; main priorprimary/candidate/faultSHA2562b29178d6b2f3ec30db757766dd1274845711eccf573e78497ee1f8828afdb81, separateprefill8746bytes. Programsourceidentity is not GPUISA/timing identity. NativeC++/Metal/XLang3 images/inference; CPythonorchestration/audits only.

Original3metric/golden/source analyzerASTs unchanged exceptnativepredicate,15actualAPIrecords andnegativecapabilities/owners/lazy/delta cases pass includingK32marker. Matched sharedmetricprefix andprospective5%4BTTFT/2%model-wall-memory gates unchanged. Source/build/library/models/checkpoints/runtime/cases/frame/Skill and1870compiler/SDK/host/correctnessinputs includingtwoimmutableexternalCMakeprofiles pinnedbeforetiming and unchangedafterbothstages.

## WorldSense resource and matched screens

Eight freshraw/candidate resourcecalls exact/nativevalid. Sampledpostdecode Metal21,903,376,384→19,091,357,696B (-12.838289%); footprint18,060,736,936→15,603,316,416B (-13.606424%). Includes storagecontrast; snapshotsnotpeaks/freeRAM; resourcetimings ineligibleasspeedevidence.

Matchedscreen twofreshworkers/fivesingle-carrequests each,20modelcalls. Firstretained, warmmedians foursamples/model/worker. Allcomplete outputtokens/text/prompts/effectiveconfigs/images/Skills/interpretedsemantics matchfrozenoracle.

|Model|TTFT seconds control /candidate|Total seconds control /candidate|Derived tokens/s control /candidate|
|---|---:|---:|---:|
|Qwen3-1.7B|2.128719 /2.710296|3.629183 /4.105385|23.614402 /22.818180|
|Qwen3.5-4B|6.705428 /5.854894|16.222181 /15.357762|9.890265 /9.720440|

4BTTFT-12.684251%, total-5.328624%, deriveddecode-1.717095%. TextTTFT+27.320514%, total+13.121473%, deriveddecode-3.371765%. NativeSkillwall19.854067604→19.511166979s (-1.727105%). MatchedMetalunchanged19,091,357,696B; footprint15,725,229,880→15,614,670,504B (-0.703070%). Memorypreservationpassed, textpreservationfailed; no othergainrescuesit.

Firstafterinit VLMvisionphase2.259141→2.407209s; firstVLM TTFT11.807733→9.195123s. These descriptivefirstresults andwarmphases are not isolateduploadcosts, hardware-cold measures or confirmation. Bothworkers AC100%before/after; noOSthermal/performancewarning recorded, notGPUclock/temperature proof. No verifiedexclusion. Posthocphase/memory/chunkplans andallfirst/warmrawdata retained.

## Decision, repairs and limits

REJECT/no unchangedretry/raw20/actualWS10-20/full48/canonical/reproof/production48/promotion. All28modelcalls14parentrequests retained; Skillwall once/frame. Deriveddecode=(generated-1)/(total-TTFT) includeshost/stop/postloop/possibleextraEOS, notpureGPUthroughput. Memorysnapshotsnotpeaks, owner/encodedcountersnotphysicalresidency/completedGPUcounts. Fixedframes actualWorldSenseSkill→Galaxy→Garnet exclude live decode/scheduling/eventpersistence/worldstatewrites/retrieval/ranking. Fourfilesrestored; secondcarcorrelated, notindependentfusion. M5/24GiB/4B separatefromunmetM4mini/vLLM. No consumedsource/driveredit/build/competingGPUinvestigation/archivecopy duringtiming. Canonicalacceptance andnormalappsmoke remainrequired.

Setupscript else1 parseerror occurred BEFORE execution/sourcecreation; scripttext/logretained, corrected andsetupthenran once. InvalidK32ON/vectorOFF configure correctlyrejectedbeforebenchmark. An optionalextra prereqinsert then asserted exactlyone preexistingcorrectness-terminal check, but template hadtwo; it failedbefore ANYdriveredit afterflag-proofwaswritten. Shellcontinuedresource withunchangedcompleteoriginalproofprereqs. Allnative/model/CPU/flag/gateproofs werealreadyverified beforedispatch, no source/driver/buildoverlap ormissingnumericalproof; validcohort retainedwithoutrestart/reclassification/gaterelaxation. Terminalreport writtenafterresource, OSmetadata planbeforematched, alreadydeclaredin51design. Detailedwrappererror retained.

Next read-onlyreview found WS34 cachedmetadata+lazydefault retirement was rejected atresourcebecause footprintfailed, so no speedtest; WS45primary+lazy had4Bbenefit but textdecodefailure. A distinct interaction could combine losslessprimarymemory saving, lazydefault retirement andcachedmetadatatoavoidgetterpools, retaining51prefill/directupload. This is untested; oldrejectionsremain. Newprospectivedesign/fullcombinedproof/unchangedfreshgates required. No physicalsaving or speed from adding effects is assumed.
