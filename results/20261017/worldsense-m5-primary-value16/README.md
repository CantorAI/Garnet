# WS-M5-43 value16 sign/mantissa fold: rejected

Design and integerbounds were written before implementation. Starting from exact full-primaryWS40canonicalf21c1a258 source (1715files), a four-filedelta adds a defaultOFF Metal-only compact-reader variant. It reconstructs originalBF16bits withushort4intermediates andexact (sm*257)&0x807f sign/mantissa placement, then ORs exponentbits. This is integer decoding, not quantization or floating16 arithmetic. Original payload/u32metadata/addressing,scalarreference/rawbranch,FP32kernelbodies/Korder/reductions/threadgeometry,CPUcodec/nativeupload,Model/wrapper/profiles/budgets/prefix/image-answer policy andCUDA unchanged.

## Compiler and correctness evidence

ActualhostC++preprocessor materialization provesOFFshader exactlymatchesWS40 andON→OFFonehelper replacement recovers the entirefragment. The standaloneMetalCLI dynamichelper probe emitted differentIR: originaluint4operations versusushort4fold; exactcommands/versions/IR/hashes retained. This is not whole-runtime-kernel/GPUmachinecode identity, registerusage, timing or causal speed evidence. Runtimeprofile usesnilMTLCompileOptions; CLIprobeO3defaults are recorded separately. No-oppre-performance exclusion was declared; its IR was distinct, so correctness proceeded.

OFF/ON/fault builds andall7native tests eachpassed. Original186fullFP32projectionrecords remain byte-identical under originalFP64bounds. New8160x128compactpattern exercises1044480originalfiniteBF16words across255exponentbases/all256sign-mantissabytes/all16tagchoices across8blocks; bothGPUreaders exactinONandfault,allblockscompact,ownersrestored. Bothreaders checked5921308672originalFP32words across398actualpacked learnedmatrices each;48originalrawfallbackbuffers unchanged. Retainedprimary+raw9033678080B and197/201/398nativeownertuples remainWS40. Full32VLM/28text finite states/tokens/vision/grid/trained/prefix/capacity/owner/fault recoverypassed.

Threequantitative/golden ASTs unchanged exceptnative-route capabilitycheck;10actualnativecalls andnegativecapability/countercasespassed. Complete1715sourcefiles, model/checkpoint/assets,library/buildflags,frame/Skill/config inputs and1864compiler/SDK/hostfiles were frozenbeforetiming andverifiedunchangedafterbothstages. No consumedsource/driverchanges,builds,otherexperimentGPUjobs orarchivecopiesduringtiming.

## Fresh WorldSense results

Resource8calls passed. Sampledpostdecode maxima,notpeaks/freeRAM:
metal_allocated_bytes: 21903376384→19091357696B (-12.838289%).
task_physical_footprint_bytes: 18051709256→15631365824B (-13.407835%).

NEW20call speed screen: freshcontrol/candidateworkers,five single-car repetitions each;first andfourwarmcalls/model keptseparate. All28combinedbenchmarkcalls preserve completeoutputs/prompts/effectiveconfigs/input/image/Skill IDs and interpretedsemantics. Warmmedians:

| Model | TTFT seconds control / candidate | Total seconds control / candidate | Derived tok/s control / candidate | Decode change |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | 7.275444 / 2.137204 | 8.538238 / 3.548431 | 26.445377 / 24.906913 | -5.817514% |
| Qwen3.5-4B | 21.321267 / 6.605275 | 31.672219 / 15.853043 | 9.031300 / 10.061184 | +11.403486% |

Qwen3-1.7B: TTFT -70.624425%, total -58.440713%.
Qwen3.5-4B: TTFT -69.020249%, total -49.946535%.

NativeSkill wallmedian 40.214017584→19.415021875s (-51.720760%).

## Decision and limits

**REJECT**: text deriveddecode-5.817514% violates unchanged2%preservation gate. Favorable4B/memory/total/wall results cannotrescuefailure. ExtraactualWS10speed/fullmirroredconfirmation/canonicalintegration were notdispatched;no promotion orunchangedretry. WS10normal restored,fourvideofilesonline,no camera,validbenchmarkPOST404.

This whole-candidate-versus-rawcohort doesnotisolate a fold-versus-uint32kernel improvement. Do not divide it against othercohort timings toclaimcausality. Opcodecounts are a standaloneIRdescription,not aGPUthroughput/register/occupancyproxy. Deriveddecode=(generated-1)/(total-TTFT)includeshost/stop/postloop/possibleextraEOS,notpureGPU. ProfileGPU/waitoverlap;nativeowner/encodedcounters arenotphysical/completedGPUcounts;packing/init/transientpeaks notisolated. ParentnativeSkillwallcountedonce/frame;first costsretained.

Fixedvideo-derivedframes traverseactualWorldSensecurrentSkill/Galaxy/nativeGarnet,excludinglivedecode/scheduling/eventcommits/retrieval/ranking. NativeC++/Metal/XLang3 images/inference;CPythonorchestration/auditsonly. M5MacBook24GiB/4Bcohort separatefromunmetM4mini/vLLMgoal. Allsource/build/model/input/runtime/SDK identities,fullF32fixtures,originalprojectionwordbinaries,materializedshader/IR/newcompactcaseproofs andrawfirst/warmcalls/parentrequests retained. Fixtureabsolute paths mayneedremapping.

Next inspect a distinct model-scoped storage strategy: preserve the companion model raw withoriginaldefault-storage lifetimes; apply primaryweights and existing safeunused-default retirement only towide4Bgraphs. This requires anactualper-modelbytebudget and design beforecode; it is notimplemented orspeed-proven here.
