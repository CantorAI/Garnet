from pathlib import Path
import json,statistics,re,datetime
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=root/'work/worldsense-raw-runtime-diagnostic';repo=Path('/Users/shawn/CantorAI/Garnet');d=repo/'results/20261017/worldsense-m5-raw-runtime-diagnostic';load=lambda p:json.loads(p.read_text());assert load(w/'diagnostic-controller-complete.json')['forty_original_outputs_native_exact'];assert load(w/'dependencies-after-diagnostic.json')['all_aliases_and_bytes_unchanged'];rest=load(w/'restoration-verification.json');assert rest['benchmark_valid_request_http404'];m=load(w/'cohort-manifest.json');assert [x['label']for x in m]==load(w/'protocol.json')['schedule']and all(x['client_exit']==0 for x in m)
pairs=[load(w/('analysis-control-'+str(i)+'-candidate-'+str(i)+'.json'))for i in [1,2]];assert all(len(x['checks'])==20 and all(v['exact']and v['native_lazy_defaults_valid']for v in x['checks'])and not x['promote']for x in pairs)
summary={};first={};system={};raw_memory={}
for worker in m:
 label=worker['label'];data=load(w/(label+'.json'));summary[label]={};first[label]={}
 for model in ['Qwen3.5-4B','Qwen3-1.7B']:
  calls=[(i,c)for i in data['items']for c in i['result']['model_response'].values()if isinstance(c,dict)and c.get('model_id')==model];warm=[c for i,c in calls if not i['warmup']];assert len(warm)==4
  summary[label][model]={'ttft':statistics.median(c['ttft_seconds']for c in warm),'total':statistics.median(c['total_seconds']for c in warm),'decode_tps':statistics.median((c['generated_tokens']-1)/(c['total_seconds']-c['ttft_seconds'])for c in warm),'decode_phase_seconds':statistics.median(c['timings']['decode_seconds']for c in warm),'post_decode_residual_seconds':statistics.median(c['total_seconds']-c['ttft_seconds']-c['timings']['decode_seconds']for c in warm),'decode_profile':{k:statistics.median(c['timings']['decode_profile'][k]for c in warm)for k in ['prepare_ms','encode_ms','gpu_ms','wait_ms']}}
  first[label][model]={k:calls[0][1][k]for k in ['ttft_seconds','total_seconds','generated_tokens']}
  if label.startswith('candidate'):raw_memory[label+':'+model]=[{k:c['timings']['memory_after_decode'][k]for k in ['metal_allocated_bytes','metal_recommended_working_set_bytes','task_physical_footprint_bytes','task_compressed_bytes','task_resident_bytes']}for c in warm]
 summary[label]['Skill_wall']=statistics.median(i['native_call_seconds']for i in data['items']if not i['warmup'])
 size=int(re.search(r'page size of (\d+) bytes',worker['vm_stat_before']).group(1))
 def counters(text):return {k:int(v)for k,v in re.findall(r'^([^:\n]+):\s+(\d+)\.',text,re.M)}
 before,after=counters(worker['vm_stat_before']),counters(worker['vm_stat_after']);deltas={k:after[k]-before[k]for k in ['Swapins','Swapouts','Pageouts','Compressions','Decompressions']};assert all(v>=0 for v in deltas.values())
 system[label]={'page_size_bytes':size,'global_counter_deltas':deltas,'global_swapin_bytes':deltas['Swapins']*size,'global_swapout_bytes':deltas['Swapouts']*size,'swap_before':worker['swap_before'],'swap_after':worker['swap_after'],'power_before':worker['power_before'],'power_after':worker['power_after'],'thermal_before':worker['thermal_before'],'thermal_after':worker['thermal_after']}
drift={model:{k:summary['control-2'][model][k]/summary['control-1'][model][k]-1 for k in ['ttft','total','decode_tps']}for model in ['Qwen3.5-4B','Qwen3-1.7B']};drift['Skill_wall_change']=summary['control-2']['Skill_wall']/summary['control-1']['Skill_wall']-1
post={'descriptive_only':True,'workers':summary,'first_after_init':first,'newraw_memory_snapshots':raw_memory,'system_observations':system,'control_drift':drift,'limits':'Global counters are not per-process attribution or proof that swap alone caused a given latency. Snapshots are not peaks/residency/freeRAM; recommended working set is not a hard allocation cap. No performance exclusion, changed prior decision or pooled/historical denominator. Source generation equality is not ISA identity.'};(w/'posthoc-observations.json').write_text(json.dumps(post,indent=2)+'\n')
terminal={'experiment':'WS-M5-55','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_session':50693,'terminal_exit_code':0,'calls':40,'parent_requests':20,'diagnostic_only':True,'promote':False,'pair_2percent_indicators':[p['preserved_within_2percent']for p in pairs],'classification':'mixed_order_dependent_raw_comparison_with_global_memory_activity','all_outputs_native_valid':True,'prior_rejections_unchanged':True,'normal_restored':rest};(w/'terminal.json').write_text(json.dumps(terminal,indent=2)+'\n');(w/'decision.json').write_text(json.dumps(terminal,indent=2)+'\n')
rows='\n'.join(f"| {label} | {model} | {summary[label][model]['ttft']:.6f} | {summary[label][model]['total']:.6f} | {summary[label][model]['decode_tps']:.6f} |"for label in ['control-1','candidate-1','candidate-2','control-2']for model in ['Qwen3.5-4B','Qwen3-1.7B'])
swaprows='\n'.join(f"| {label} | {x['global_swapin_bytes']/2**30:.3f} | {x['global_swapout_bytes']/2**30:.3f} |"for label,x in system.items())
report=f'''# WS-M5-55 — Mirrored raw-runtime diagnostic is mixed, with large control drift

Forty actual WorldSense/Galaxy/native Garnet model calls over four fresh workers passed all original output/input/semantic/native-policy checks. All5298 pinned source/model/build/SDK/host/driver files remained unchanged. This diagnostic does not accept or promote an optimization and does not change WS53/54's valid rejections. Normal WS10/four files/no camera/valid benchmark POST404 was restored.

## Question and prospective method

Before implementing drivers or dispatch, design a diagnostic of the common newer RAW runtime. Exact existing54-OFF versus exact deployedWS10, both original raw BF16 storage, all experimental primary/vector/isolation/direct-upload/K32/lazy/cached/full optionsOFF. All common CMake booleans match. No Garnet inference source or binary was edited/rebuilt. Exact54OFF reuses its independently archived native7/full API/state/token/numerical proof at immutable59a882e9765d4c1f61cd6bc117aa620dd06abd16. Original source manifests1706/1718, library hashes, model/frame/Skill/prompt/stop/budget/config identities remain pinned.

CPU-only preprocessing with each recorded build's flags found equal selected Metal source-generation expressions (ignoring outer whitespace), byte-identical raw string literals and active body-shape header. This is static evidence, not an actual old constructor dump, CPU/GPU ISA identity or a source-versus-toolchain attribution. Both Mach-O binaries record platform1/minimum26.0/SDK27.0/tool27037.1; old SDK flag is default and newer explicit. Current SDK observations do not retroactively prove the original build environment.

Schedule fixed before timing: control1 ->newraw1 ->newraw2 ->control2, legacy candidate labels meaning NEW RAW, not optimized candidates. Each worker has one first-after-initialization request and four warm repeats of the identical single-car WorldSense Skill. Two models/request,40calls/20parents. The reversed pair runs regardless first-pair timing, not a favorable retry. Each worker's10 outputs/config/frame/Skill/route/library/source checks must pass before the next worker. Both raw modes use the same native model profiles and bounded prefix cache, no image/answer reuse, no CPython inference/images or camera. Cases/oracle are unchanged from accepted WS10.

## Results, separate pairs and first samples

Warm medians, four calls/model/worker; all first samples retained separately:

| Worker | Model | TTFT seconds | Total seconds | Derived tokens/s |
|---|---|---:|---:|---:|
{rows}

Pair1 newerRAW: text TTFT+94.346316%,total+76.190256%,derived decode-4.506885%;4BTTFT+124.233629%,total+84.624820%,decode-30.664589%;complete Skillwall+66.184916%. Original2%orientation false.

Pair2 (reversed worker order), newerRAW: text TTFT-9.483974%,total-8.728833%,decode+1.099710%;4BTTFT-8.192453%,total-7.975966%,decode+6.063409%;Skillwall-10.026769%. Original2%orientation true. Report independently, no pooling or selection of pair2. No5%VLM/resource/full-confirmation gate was waived: this protocol is diagnosis only, with no promotion path.

The same exact WS10 control's Skillwall changed26.837085896 ->47.499357604s ({drift['Skill_wall_change']*100:+.6f}%).4BTTFT10.227638 ->23.504769s and derived decode11.905916 ->7.314909tokens/s changed substantially across controls. These observations do not establish a stable intrinsic slowdown in the newer raw runtime or a packed-kernel cause.

## Recorded memory and system context

All worker endpoints retained global vm_stat/sysctl snapshots. Page size16384bytes. Cumulative global swap-counter deltas converted to GiB:

| Worker | Global swap-ins GiB | Global swap-outs GiB |
|---|---:|---:|
{swaprows}

These counters cover each complete worker window including initialization and are SYSTEM-wide, not per-process attribution or proof that swap alone caused a specific call. Stock swap usage before/after, compressor/pageout deltas and all first/warm profiles are preserved in posthoc-observations.json and raw worker manifests. No posthoc exclusion or old-result reclassification.

New RAW post-decode snapshots reported Metal21,903,376,384B against recommended working-set estimate19,069,665,280B. Recommendation is not a hard allocation cap; allocations are not physical residency. Example4B warm task snapshot: physical footprint18,052,676,032B, compressed field14,131,986,432B, resident field127,844,352B. Preserve these different OS/device measures distinctly; do not sum or reinterpret them as peak/freeRAM. Old WS10 lacks the new telemetry and remains unavailable/null, not zero or an invented matched memory ratio.

All four workers reported AC100%; OS tools reported no recorded thermal/performance warning and no CPU power status. These are not GPU clock/temperature measurements or proof that conditions were invariant. No hardware cache flushing, hardware-cold claim or statistical significance claim.

## Decision and next investigation

Diagnostic completed; outcomes mixed/order-dependent. No optimized candidate, promotion, favorable retry, source change or un-rejection. Exact old/new source-generation evidence plus variable controls argues for examining working-set pressure, allocation/lifetime/residency and measurement conditions before another shader bundle. Read previous residency/deferred-allocation failures first; a future method needs a new prospective design. Do not assert a specific memory cause or intrinsic library regression from these observations. Existing optimizer numerical,2%preservation,memory/speed/full mirrored/canonical promotion gates remain intact.

All40 model calls and20 parent requests are exported once. Derived decode=(generated-1)/(total-TTFT) includes host/stop/post-loop/possible extraEOS and is not isolated GPU throughput; GPUspan/wait/encode can overlap/include gaps. Subtract timings per-call before aggregating, never separate medians. Fixed video-derived frames exercise actual WorldSense Skill ->Galaxy ->Garnet, excluding live decoding/scheduling/event persistence/shared-state writes/retrieval/ranking. M5/24GiB/4B is separate from unmet M4mini/small-model/vLLM handover. No consumed code/driver edits, builds, competing investigation GPU jobs or archive copying during timing. Authentication helper/credentials are excluded from artifacts. Full optimization remains active and unproven.
'''
d.mkdir(parents=True,exist_ok=True);(d/'README.md').write_text(report);(root/'outputs/Garnet-4B-WS55-Result.txt').write_text(report);print('WS55 terminal/diagnostic report written; swap context and separate pair outcomes retained')
