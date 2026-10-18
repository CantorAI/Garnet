"""Record terminal valid rejection, full method and unchanged selected runtime."""
from pathlib import Path
import json,datetime,hashlib
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=root/'work/worldsense-dense-prefetch';d=Path('/Users/shawn/CantorAI/Garnet/results/20261018/worldsense-m5-dense-prefetch')
load=lambda p:json.loads(p.read_text());res=load(w/'resource-analysis.json');speed=load(w/'production-speed/analysis.json');normal=load(root/'work/worldsense-opt/ws60-normal-final.json');post=load(w/'posthoc-text-decode-boundaries.json')
assert res['passed']and len(res['checks'])==8 and all(c['exact']and c['native_admission']for c in res['checks'])
assert not speed['passed']and len(speed['checks'])==20 and all(c['exact']and c['native_lazy_defaults_valid']for c in speed['checks'])
assert load(w/'model-assets-after-screen.json')['all_recorded_model_assets_rehashed_exact']
assert load(w/'dependencies-after-production-speed.json')['all_aliases_and_file_bytes_unchanged']
assert not any((w/s/'analysis.json').exists()for s in ['primary-speed','speed','confirmation'])
identity=load(w/'production-speed/driver-identities-before-dispatch.json');assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in identity['files'].items())
utc=datetime.datetime.now(datetime.timezone.utc).isoformat();reason='Companion text derived decode regression32.133820% exceeds unchanged2%preservation gate.'
decision=dict(experiment='WS-M5-60',observed_utc=utc,terminal_session=68690,terminal_exit_code=0,decision='reject',reason=reason,stage_outcomes={'resource':True,'production-speed':False},promote=False,prototype_confirmed=False,no_unchanged_retry=True,later_stages_dispatched=False,normal_restored=normal,source_manifest_sha256=hashlib.sha256((w/'starting-source-manifest.json').read_bytes()).hexdigest())
assert not(w/'decision.json').exists();(w/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
(w/'production-speed-terminal.json').write_text(json.dumps(dict(controller_session=68690,terminal_exit_code=0,all20outputs_native_exact=True,all_stage_input_driver_identities_unchanged=True,passed=False,normal_restored=normal),indent=2)+'\n')
table='\n'.join(f"| {m} | {speed['metrics']['control-1'][m]['ttft']:.6f} / {speed['metrics']['candidate-1'][m]['ttft']:.6f} | {speed['metrics']['control-1'][m]['total']:.6f} / {speed['metrics']['candidate-1'][m]['total']:.6f} | {speed['metrics']['control-1'][m]['decode_tps']:.6f} / {speed['metrics']['candidate-1'][m]['decode_tps']:.6f} |"for m in ['Qwen3.5-4B','Qwen3-1.7B'])
a=post['workers']['control-1']['medians'];b=post['workers']['candidate-1']['medians']
report=f'''# WS-M5-60 — Ordered raw dense load lookahead rejected

REJECT. All20 actual WorldSense/Galaxy speed-screen outputs and native route checks passed, but text derived generation regressed32.133820%, beyond the original2%preservation limit. No unchanged retry, later speed/confirmation/canonical stages or promotion. WS10 remains selected; full optimization stays active.

## Prospective method and exact source

[Design before implementation](native-design-before-implementation.txt), [pre-source refinement](implementation-refinement-before-source.txt) and [protocol](protocol.json). Immutable WS54 source, not58/59 worker/QoS or56 residency. Prior27/28/29 changed thread-group geometry and failed preservation; those rejections remain. This is a different load schedule with original128threads/fourSIMD32groups/16outputs and the existing weight layout.

DefaultOFF GARNET_METAL_DENSE_PREFETCH. For raw row1 BF16 projections with inner2048/6144 and output257..99999, keep one iteration's activation float4 and four ushort4 weight vectors in registers while loading the next128K iteration before accumulating the current one. Per-lane j=0..3 BF16 expansion, FP32product/accumulation and SIMD32 reduction expressions/order remain original. Bounds guard the next load; tail output loads are individually guarded; initial/next packets are initialized. More live registers may reduce occupancy; register allocation/occupancy/frequency were not measured. No inference speed is inferred from source structure.

Raw-only admission excludes packed weights,4Binners2560/9216,row2,smalloutputs,vocabulary and unsupported shapes. New PSO cache is separate, mutex-protected and bounded128; missing/unsupported PSO falls back to original prepared pipeline and original geometry. A process-global atomic counts actual new encoder dispatches after dispatch encoding, potentially including aborted batches, not completed GPU work. Serving captures two boundary reads and a per-request delta, no per-token getter. Original native counters retain their original semantics.

Source1720files/seven changed/new versus1718-file54: Metal CMake, native entry, context header/implementation, serving telemetry, new shader header and new forced-raw test. Original embedded MSL blocks/body header/primary codec/graph/session/xModels/CUDA/image code/original projection tests stay exact54. Loader and wide-admission code independently checked exact54. Original precision, learned weights, FP32order, image/context/prefix/output budgets and stop rules remain. No engine/SDK changes; CUDA source is preserved but CUDA execution is untested on this Mac. C++/Metal/XLang3 handle images and inference; CPython only orchestrates/audits. No camera.

RAW all10method optionsOFF. PRIMARY existing PRIMARY/WIDE/LAZY plus newPREFETCH ON(4), other6OFF. Candidate all10ON; FAULT adds only existing allocation fault hooks. Matched-primary contrast remains six flags. Joint profile versus WS10 is not an isolated causal comparison of lookahead against54 or59. No historical ratio is used to claim a gain or a cause.

## Correctness and retained setup failures

All four builds passed. Initial native controller24503 ran8tests/fourroles(32runs), retaining the original186 full projection records plus186 forced-raw records from93shapes/two changed inputs under original FP64bounds. Forced raw prevents packed admission from masking new-kernel execution; native counts must match actual targets. Tails257/272, unsupportedinners,row2/small/vocab/prefill fallbacks, NaN output poison, unchanged inputs, malformed shapes, queued source release, Abort unreadability and exact retry passed. Every logical output word matches across the four roles. Original projection test is unchanged.

The initial native run did not save a separate pre-execution dylib SHA inventory. The complete before/after identity-frozen proof68305 deliberately repeated native tests (another32runs), then executed all original full learned/model guards. First native logs/manifests and forced-raw digest are preserved separately; final complete projection streams are from the frozen proof. Do not count deterministic repeat words as independent model outputs or timing samples.

All five packed GPU readers each checked4,200,857,600 original FP32words over201packed matrices. The245raw buffers/1,724,383,232 HOSTwords remain distinct. Retained packed+raw weights9,850,930,176B. Native direct-upload corpus546weights/334,609,152words passed. Full32VLM/28text finite states and generated tokens, trained layers, vision/multigrid/shared-owner close/repeat, prefix hit/miss and capacity rejection/recovery, ordinary/resident alternation, late sessions, malformed input, source close/rebuild and original fault paths passed. Actual API raw-text prefetch deltas positive only when enabled; VLMzero. Native OFFMAIN exact54; each other actual MAIN becomes exact54 after removing one exact new fragment; all four actual prefill programs exact54. Source identity is not GPU ISA/register identity. Short correctness capacities512VLM/256text differ from actual WorldSense2048.

Independent full proof/source/config admission passed. Ten invalid configurations rejected: old6 plus prefetch/body/native/interleaved/coalesced combinations. The original BODY_SHAPE/native-bfloat guard rejects that invalid combination before the new later guard; the first negative fixture expected the later message and failed. Its script/logs are retained; only expected diagnostic repaired, no guard relaxed. Initial prebuild preparation asserted on six namespace occurrences before writing and attempted a missing build driver(exit2); no build or inference started. The raw-only counter compile guard was corrected before all builds. A read-only source inspection guessed src/model/model_tensor_io.h; actual manifested path is src/tensor/model_tensor_io.h. Failed inspection metadata and correction remain. No numerical or performance failed attempt was overwritten or rerun unchanged.

All five analyzer numeric/golden/metric/threshold ASTs exact54 outside added native admission.15actual full API records (five for the actualWS10screen helper) and1952malformed copied fixtures tested new capability/type/count/scope checks; copies are offline diagnostics, not inference samples. OldWS10positive fixtures remain legacy/unavailable-memory. Checkpoint/config/tokenizer identities rehashed before and after model/speed work; source/libraries/runtime/drivers/input identities and1874build/SDK/host dependencies frozen and checked. No consumed edits, builds, competing investigation GPU jobs or archive copies during timing.

## Actual WorldSense results

Candidate-only functional64376 returned four exact calls over two parent Skill requests; original tokens/text/prompt/image/Skill/config/source/route/native memory matched. New route positive for text and zeroVLM. Timings are ineligible for speed; no control/comparison/promotion from this smoke.

Fresh resource87180:8calls/two fresh workers, first plus one followup Skill each. Post-decode Metal snapshot maxima21,903,376,384 ->16,870,899,712B(-22.975803%); sampled native-process footprint18,055,395,608 ->16,075,585,312B(-10.965200%). Original10%/5%gate passed. Single native Cantor process snapshots, not peak/free/unique physical RAM/residency or full UI/server totals. No memory ratio is imported into the separate speed screen.

Fresh actualWS10screen68690:20calls/two fresh workers, first plus four warm identical fixed-car Skills each. All20golden/native/source/profile/input checks passed; allstage driver/input hashes and1874dependencies unchanged. Selected1706-file WS10 source/library separately pinned from1720-file candidate. Warm medians:

| Model | TTFT seconds control / candidate | Total seconds control / candidate | Derived tokens/s control / candidate |
|---|---:|---:|---:|
{table}

Complete Skill wall {speed['native_skill_wall_p50']['control-1']:.6f} ->{speed['native_skill_wall_p50']['candidate-1']:.6f}s({speed['native_skill_wall_gain']*100:.6f}%improvement).4BTTFT improved73.288346%,total53.498613%,derivedgeneration23.006445%. Text TTFT/total improved but derivedgeneration fell32.133820%; original2%gate fails. Favorable other values do not override rejection. No matched-primary/raw/mirrored/canonical stages dispatched.

Posthoc retained text decode-loop medians {a['decode_phase_seconds']:.6f} ->{b['decode_phase_seconds']:.6f}s; GPUspans {a['decode_profile']['gpu_ms']:.6f} ->{b['decode_profile']['gpu_ms']:.6f}ms. Per-call post-decode residual medians {a['post_decode_residual']*1000:.6f} ->{b['post_decode_residual']*1000:.6f}ms. Candidate median encoded lookahead count {b['dense_prefetch_encoded_delta']:.0f}. These locate timing/route boundaries, not a register/shader/clock/VM/residency cause. Differences formed per call before aggregation; phase spans overlap and separate medians are not additive. No posthoc exclusion, causal cross-trial ranking, un-rejection or statistical significance.

All first/warm/functional records remain. Derivedgeneration=(generated-1)/(total-TTFT) includes host/stop/postloop/possibleextraEOS/snapshots; not pureGPUthroughput. ActualWS10memory API unavailable/null, never zero. Old readyAPI lacks separate worker/model-load duration; no hardware-cold claim. Fixed frames cover current Skill->Galaxy->Garnet, excluding live decode/scheduling/event persistence/shared-world-state writes/retrieval/ranking. Four ordinary files remain; secondcar is correlated, not independent-camera fusion. M5/24GiB/4B is separate from unmet original M4mini/smallmodel/vLLM handover.

NormalWS10 verified after terminal: PID{normal['pid']}, SHA{normal['WS10_sha256']}, fourfiles online, no camera, valid benchmark POST404. Full optimization active; no global optimum or promotion claim.
'''
d.mkdir(parents=True,exist_ok=False);(d/'README.md').write_text(report);(w/'result.txt').write_text(report);(root/'outputs/Garnet-4B-WS60-Result.txt').write_text(report)
print('WS60 valid rejection, full method, raw scopes and normal verification recorded')
