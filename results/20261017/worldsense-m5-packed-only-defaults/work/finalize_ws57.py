"""Record a valid rejected WS57 stage only AFTER observing its terminal exit0."""
from pathlib import Path
import json,datetime,sys
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=root/'work/worldsense-packed-only-defaults';d=Path('/Users/shawn/CantorAI/Garnet/results/20261017/worldsense-m5-packed-only-defaults');load=lambda p:json.loads(p.read_text());stage=sys.argv[1];session=int(sys.argv[2]);assert stage in ['resource','production-speed']and session>0
resource=load(w/'resource-analysis.json');rest=load(w/'restoration-verification.json');assert rest['benchmark_valid_request_http404'];assert all(x['exact']and x['native_admission']for x in resource['checks'])and len(resource['checks'])==8
outcomes={'resource':resource['passed']};speed=None
if stage=='production-speed':
 assert resource['passed'];speed=load(w/'production-speed/analysis.json');assert len(speed['checks'])==20 and all(x['exact']and x['native_lazy_defaults_valid']for x in speed['checks']);assert load(w/'dependencies-after-production-speed.json')['all_aliases_and_file_bytes_unchanged'];outcomes['production-speed']=speed['passed']
else:assert load(w/'dependencies-after-resource.json')['all_aliases_and_file_bytes_unchanged']
assert not outcomes[stage],'Use full confirmation/integration protocol after passes'
assert not any((w/f/'analysis.json').exists()for f in ['primary-speed','speed','confirmation'])
now=datetime.datetime.now(datetime.timezone.utc).isoformat();reason=('Original resource reduction gate failed; no speed stage dispatched.'if speed is None else'At least one original2%model/Skill speed preservation gate failed; no unchanged retry or later stages.')
terminal=dict(experiment='WS-M5-57',utc=now,controller_session=session,terminal_exit_code=0,passed=False,all_outputs_native_exact=True,normal_restored=rest,later_stages_dispatched=False);(w/(stage+'-terminal.json')).write_text(json.dumps(terminal,indent=2)+'\n')
decision=dict(experiment='WS-M5-57',observed_utc=now,terminal_session=session,terminal_exit_code=0,stage_outcomes=outcomes,prototype_confirmed=False,promote=False,decision='reject',reason=reason,no_unchanged_retry=True,later_stages_dispatched=False,normal_restored=rest);(w/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
res=resource['reduction'];sample=resource['sampled_maximum_bytes'];table='';extra='No speed measurements dispatched; resource timings are ineligible for speed evidence.'
if speed:
 table='\n| Model | TTFT seconds control / candidate | Total seconds control / candidate | Derived tokens/s control / candidate |\n|---|---:|---:|---:|\n'+'\n'.join(f"| {m} | {speed['metrics']['control-1'][m]['ttft']:.6f} / {speed['metrics']['candidate-1'][m]['ttft']:.6f} | {speed['metrics']['control-1'][m]['total']:.6f} / {speed['metrics']['candidate-1'][m]['total']:.6f} | {speed['metrics']['control-1'][m]['decode_tps']:.6f} / {speed['metrics']['candidate-1'][m]['decode_tps']:.6f} |"for m in ['Qwen3.5-4B','Qwen3-1.7B'])
 extra='Twenty additional actualWS10screen calls, twofresh workers, onefirst+fourwarm identical carSkills/model/worker. All original exact/native checks passed. OldWS10memory telemetry remains unavailable/null; separate resource denominator is not substituted.\n'+json.dumps(dict(comparisons=speed['comparisons'],native_skill_wall_p50=speed['native_skill_wall_p50'],native_skill_wall_gain=speed['native_skill_wall_gain']),indent=2)
report=f'''# WS-M5-57 — Packed-graph-only default retirement rejected at {stage}

REJECT; no promotion or unchanged retry. {reason} All returned WorldSense/Galaxy/native Garnet outputs and original policies matched. Normal exactWS10/four video files/no camera/valid benchmark POST404 restored. Full optimization remains active.

## Prospective hypothesis and implementation

Read WS15/17 global/default allocation tradeoffs, WS22/23/26 diagnostics and WS32/54/55/56 residency/isolation evidence before design. Preserve every rejection and mixed diagnostic. Immutable54seed, not56: no request-residency code/lease/budget changes. Distinct hypothesis: retire defaults only for graphs with actual PrimaryReady BF16 slots, leaving rawtext's original default buffers. No claimed cause of earlier text slowdowns and no cross-cohort causal ranking.

Default-OFF PRIMARY_ONLY_LAZY_DEFAULTS requires PRIMARY_BF16_BLOCKS/WIDE_PRIMARY_STORAGE/LAZY_RESIDENT_DEFAULTS. Exactly3source changes: CMake, compiled_graph.cpp admission guard and native entry marker. Source1718files/10overlays from canonical f21c1a2584031ee9d1c129d1e58818f9d1c826d2. Guard runs inside existing once-per-graph retirement planning after original graph validation, under existing locks. No primary slots: cache an empty plan; otherwise execute the exact original alias/weight/layout/root-liveness safety plan. Clear/rebuild resets existing plan. No model-name/global-owner heuristic, new buffers/owners/residency objects, shader/kernel/layout/math/reduction-order/loader/session/serving/xModel/CUDA changes. CUDA execution not tested on this Mac.

Tradeoff: retaining rawtext defaults costs Metal allocations and may affect placement/pressure; source/counter identity is not physical residency or a speed prediction. Candidate/storage rawtext planned/retired/rehydrated0; packedVLM positive planned/retired and0normalrehydration. Global primary owners201after VLM may coexist with rawtext and are not per-graph admission evidence. Cached metadata/isolation/K32/direct-upload seed policies remain global; this is not an exact WS10 text implementation or isolated policy causality.

Controls: raw all10method optionsOFF; storage PRIMARY/WIDE/LAZY/PRIMARY_ONLY ON(4), other6OFF; candidate all10ON; separate fault candidate+originalPRIMARY_TEST_FAULTS. Joint candidate rawcontrast10flags, matchedstoragecontrast6 with same packed partition/selective-retirement policy. Original8resource(10%Metal/5%footprint),20actualWS10(2%eachmodel TTFT/total/deriveddecode/parentSkillwall),20matchedstorage(5%4BTTFT/2%preservation),20raw,48full mirrored BOTHpairs, then canonical integration/fullreproof/48actualWS10 BOTHpairs/normalapp gates unchanged. Later stages remain undispatched after this valid failure.

## Correctness before timing

All7original native tests/four builds passed(28runs), all186complete projection records exact. Four actual generated MAIN/prefill programs byte-identical54. Exact3file inverse recovers original graph/entry; all original math/source/tests retained. Nine invalid configs rejected. Full201packed matrices/five GPU readers EACH4,200,857,600originalFP32words;245raw HOSTreader1,724,383,232words;446unique matrices/5,925,240,832originalwords, retained9,850,930,176B. Direct546weights/334,609,152words exact. These reader repetitions are not independent model outputs.

Original full32VLM/28text finite-state bits/tokens/vision grids/prefix/capacity/error/private-host isolation/ordinary-resident/shared-close/rebuild/late-session guards passed in all3API modes. Raw dense fixtures keep0retirement; real4Btrained linear ordinary/resident paths retain positive retirement/rehydration and exact outputs. Existing disabled allocation-injection branch remains unexecuted/unclaimed. API capacities512/256 differ from actualWorldSense2048 as declared. Every image freshly decoded/encoded; no answer/image cache. VLMchunk64/text16,image65536,textprefix64tokens/2entries64MiB/checkpoints/stops/outputlimits unchanged. Native C++/Metal/XLang3 images/inference; CPython only orchestrates/audits.

Original numeric/golden/input/model/metric AST gates unchanged; capability/retirement/owner/counter corruption guards exercised actual calls. Source/libs/checkpoints/assets/frame/Skill/cases/runtime/SDK/driver identities and1872compiler/host dependencies fixed before timing and verified afterward. No consumed edits, builds, competing investigation GPU jobs or archive copying during timing.

## Actual WorldSense measurements

Resource:8actualcalls/twofresh workers/onefirst+onefollowup Skill each. Sampled maximum post-decode Metal{sample['control-1']['metal_allocated_bytes']} ->{sample['candidate-1']['metal_allocated_bytes']}B, reduction{res['metal_allocated_bytes']*100:.6f}%; processfootprint{sample['control-1']['task_physical_footprint_bytes']} ->{sample['candidate-1']['task_physical_footprint_bytes']}B, reduction{res['task_physical_footprint_bytes']*100:.6f}%. Resource passed={resource['passed']}. Snapshots, not peaks/freeRAM/physical-residency measures. Resource timing excluded from speed claims.
{table}

{extra}

## Reporting and remaining uncertainty

Keep all first/warm/failure records and exact outputs. Deriveddecode=(generated-1)/(total-TTFT) includes host/stop/post-loop/possibleextraEOS; not pureGPUthroughput. ParentSkillwall counted once/frame. Phase/GPU/wait measures overlap/include gaps; medians not additive. Raw global VM/swap/power/thermal endpoints retained, no posthoc exclusion or attribution; not per-process residency/clock/temperature proof. Fixed video-derived frames traverse actual WorldSense Skill ->Galaxy ->Garnet and exclude live decoding/scheduling/event persistence/shared-state writes/retrieval/ranking. Four normal video files restored; secondcar is correlated, not independent-camera fusion. M5/24GiB/4B cohort remains separate from unmet M4mini/small-model/vLLM handover. Full confirmation, production acceptance and global optimum are unproven.
'''
d.mkdir(parents=True,exist_ok=True);(d/'README.md').write_text(report);(root/'outputs/Garnet-4B-WS57-Result.txt').write_text(report);print('WS57 valid failure decision/report saved')
