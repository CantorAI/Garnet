from pathlib import Path
import json,datetime,statistics,re
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=root/'work/worldsense-request-weight-residency';d=Path('/Users/shawn/CantorAI/Garnet/results/20261017/worldsense-m5-request-weight-residency');load=lambda p:json.loads(p.read_text());a=load(w/'production-speed/analysis.json');r=load(w/'resource-analysis.json');rest=load(w/'restoration-verification.json')
assert r['passed']and not a['passed']and len(a['checks'])==20 and all(x['exact']and x['native_lazy_defaults_valid']for x in a['checks']);assert rest['benchmark_valid_request_http404']and load(w/'dependencies-after-production-speed.json')['all_aliases_and_file_bytes_unchanged'];assert all(x['client_exit']==0 for x in load(w/'production-speed/cohort-manifest.json'));assert not any((w/p/'analysis.json').exists()for p in ['primary-speed','speed','confirmation'])
now=datetime.datetime.now(datetime.timezone.utc).isoformat();terminal=dict(experiment='WS-M5-56',utc=now,controller_session=37967,terminal_exit_code=0,all20_outputs_native_exact=True,passed=False,comparisons=a['comparisons'],native_skill_wall_gain=a['native_skill_wall_gain'],normal_restored=rest,later_stages_dispatched=False,candidate_promoted=False);(w/'production-speed-terminal.json').write_text(json.dumps(terminal,indent=2)+'\n')
decision=dict(experiment='WS-M5-56',observed_utc=now,terminal_session=37967,terminal_exit_code=0,stage_outcomes={'resource':True,'production-speed':False},prototype_confirmed=False,promote=False,decision='reject',reason='Text derived decode -28.498098% exceeds unchanged2%limit despite favorable4B and completeSkill latency. No isolated request-residency causal claim from joint candidate.',no_unchanged_retry=True,later_stages_dispatched=False,normal_restored=rest);(w/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
observations={};system={};first={}
for worker in load(w/'production-speed/cohort-manifest.json'):
 label=worker['label'];data=load(w/'production-speed'/(label+'.json'));observations[label]={};first[label]={}
 for model in ['Qwen3.5-4B','Qwen3-1.7B']:
  calls=[(i,c)for i in data['items']for c in i['result']['model_response'].values()if isinstance(c,dict)and c.get('model_id')==model];warm=[c for i,c in calls if not i['warmup']];assert len(warm)==4
  observations[label][model]={k:statistics.median(c['timings'][k]for c in warm)for k in ['decode_seconds','prefill_seconds','reset_seconds']}
  observations[label][model]['post_decode_residual_seconds']=statistics.median(c['total_seconds']-c['ttft_seconds']-c['timings']['decode_seconds']for c in warm)
  observations[label][model]['decode_profile']={k:statistics.median(c['timings']['decode_profile'][k]for c in warm)for k in ['prepare_ms','encode_ms','gpu_ms','wait_ms']}
  first[label][model]={k:calls[0][1][k]for k in ['ttft_seconds','total_seconds','generated_tokens']}
  if label.startswith('candidate'):observations[label][model]['request_counters']=[c['timings']['request_weight_residency']for c in warm]
 size=int(re.search(r'page size of (\d+) bytes',worker['vm_stat_before']).group(1));assert size==int(re.search(r'page size of (\d+) bytes',worker['vm_stat_after']).group(1))
 def counters(text):return {k:int(v)for k,v in re.findall(r'^([^:\n]+):\s+(\d+)\.',text,re.M)}
 before,after=counters(worker['vm_stat_before']),counters(worker['vm_stat_after']);delta={k:after[k]-before[k]for k in ['Swapins','Swapouts','Pageouts','Compressions','Decompressions']};assert all(v>=0 for v in delta.values())
 system[label]=dict(page_size_bytes=size,global_counter_deltas=delta,global_swapin_counter_bytes=delta['Swapins']*size,global_swapout_counter_bytes=delta['Swapouts']*size,**{k:worker[k]for k in ['swap_before','swap_after','power_before','power_after','thermal_before','thermal_after']})
post=dict(descriptive_only=True,models=observations,first_after_initialization=first,system_observations=system,limits='No excluded samples or causal attribution. Per-call residuals before aggregation; component medians nonadditive. GPU spans include gaps and wait overlaps. VM counters system-wide/compressor page equivalents, not process attribution or direct disk I/O; API residency requests are not proof of physical residency.');(w/'posthoc-observations.json').write_text(json.dumps(post,indent=2)+'\n')
table='\n'.join(f"| {model} | {a['metrics']['control-1'][model]['ttft']:.6f} / {a['metrics']['candidate-1'][model]['ttft']:.6f} | {a['metrics']['control-1'][model]['total']:.6f} / {a['metrics']['candidate-1'][model]['total']:.6f} | {a['metrics']['control-1'][model]['decode_tps']:.6f} / {a['metrics']['candidate-1'][model]['decode_tps']:.6f} |"for model in ['Qwen3.5-4B','Qwen3-1.7B'])
swaptable='\n'.join(f"| {label} | {x['global_swapin_counter_bytes']/2**30:.3f} | {x['global_swapout_counter_bytes']/2**30:.3f} |"for label,x in system.items())
report=f'''# WS-M5-56 — Whole-request native raw-weight residency rejected by text decode gate

REJECT; no promotion. All28 actual WorldSense/Galaxy model calls/14 parent Skill requests matched original generated tokens, prompts, images, Skills, effective configuration and interpreted semantics. Correctness and resource gates passed, but text derived decoding regressed28.498098%, exceeding the unchanged2%limit. Best validated deployment remains WS10; four video files/no camera/normal benchmark POST404 restored after terminal session37967 exit0.

## Method declared before implementation

WS55's mirrored no-code diagnostic was mixed/order-dependent, with heavy system-wide swapping and large exact-control drift; it did not prove an intrinsic new-runtime or packed-kernel cause. Read the prior WS17 allocation and WS32 per-batch bounded residency failures. This distinct method amortizes one bounded native residency request across a complete serving inference, rather than acquiring/releasing per submitted batch. No unchanged favorable retry or historical-ratio causal comparison.

Seed is immutable WS54,9changed files,1720total files/15overlays from canonical f21c1a2584031ee9d1c129d1e58818f9d1c826d2. Default-OFF REQUEST_SCOPE_WEIGHT_RESIDENCY adds Context/Session begin/end/stats, graph/buffer coverage/generation validation and an XLang3 Model.infer try/finally lifecycle. Original BF16 bits, FP32 math/order, shader programs, xModels, geometry, model/checkpoint/config/token budgets and CUDA source remain unchanged. CUDA execution is not tested on this Mac. CPython orchestrates/audits only; native C++/Metal/XLang3 handle inference and image preparation. No camera or external CPython image/inference bridge.

Lease deduplicates actual MTLBuffer identities, holds strong weight owners and enforces4GiB global/per-device and2048unique-allocation limits. Same queue/set spans prefill and decode; normal Submit preserves it. Drain/end, Abort, Reset, Close and failure paths release lease/budget. Graph generation and actual supplied weight coverage are checked before state mutation; covered shared prefill graphs are cached with strong owners, bounded256graphs. Packed primary weights explicitly fall back(reason7); no packed payload or transient activations are leased. Real text model admits197masters/3,440,902,144B. Empty/unsupported/oversize/busy conditions retain fallback. Residency API success is not proof of physical residency or guaranteed speed. Test-only default-OFF hooks use64KiB and two injected failure stages; measured builds retain4GiB and no hooks.

Controls: raw all10optionsOFF; storage PRIMARY/WIDE/LAZY ON, other7OFF; candidate all10ON; separate fault candidate plus original/new fault hooks. This joint candidate differs from raw by10flags and storage by7; it does not isolate request-residency causality. VLM-only201packed/245raw partition, global lazy retirement, cached metadata, K32 vector prefill, full primary secondary-library isolation and native direct-upload remain seed policies.

## Correctness and evidence before timing

All8native tests x4builds passed(32runs);186original complete projection records were exact. New native test covers lease admission/dedup/owner lifetime, Submit/Abort/partial flush, nesting, count/byte limits, packed fallback, injected reserve/attach failure, contention and destructor release. Initial new128x128negative fixture failed to meet existing packed admission; changed ONLYnewfixture to512x128. Initial source/test/logs/binary identities and failure are retained before rebuild. No inference math, bounds, budgets or performance gates changed and no performance attempt preceded the correction.

All201packed learned matrices passed five GPU readers, EACH4,200,857,600original FP32words. HOSToriginal raw-storage reader checked245raw matrices/1,724,383,232words. Together446unique matrices/5,925,240,832originalwords, retained9,850,930,176B; reader repetitions are not independent outputs. Direct native upload checked546selected weights/334,609,152originalwords. Original32VLM/28text state bits, generated tokens, vision grids, trained layers, prefixes/capacity/error recovery, private/host/session isolation, close/rebuild/rehydration passed three API modes. Short native API capacities512/256 differ from actual WorldSense2048, as declared.

Additional trained native request guards passed actual alternate/uncovered/shared/rebuilt/closed graph handling, invalid active flags without private/host mutation, nesting, Reset/Close and real197-master4GiB global contention/readmission. Actual API text admission/cache hits/no coverage fallback/final active+global0 and VLM packed fallback/final0 were asserted. Seven invalid option configurations rejected. Exact inverse audits recover original Context/Session and Model.infer expressions; all other serving AST and four actual generated shader programs match WS54. Inference source1720files remained fixed. Original numerical/golden/input/model/metric AST gates retained; new request capability and corrupted telemetry guards exercised. Checkpoint/assets/source/build/input/frame/Skill/runtime/SDK plus1874compiler/host dependencies pinned and verified unchanged after each cohort.

## Actual WorldSense measurements

Resource: two fresh workers, one first and one followup car Skill each,8calls. Sampled post-decode Metal21,903,376,384 ->16,870,899,712B(-22.975803%); process footprint18,054,838,552 ->16,080,320,336B(-10.936227%). Original>=10%Metal/>=5%footprint gate passed. These are snapshots, not peaks/freeRAM/physical residency. Resource timings ineligible for speed evidence.

Actual WS10 screen: two fresh workers, one first-after-initialization and four warm identical car Skills/model/worker,20calls. Every returned call passed original exact/native checks. Four warm samples/model/worker; medians below:

| Model | TTFT seconds control / candidate | Total seconds control / candidate | Derived tokens/s control / candidate |
|---|---:|---:|---:|
{table}

4B TTFT-75.416149%, total-57.417677%, derived decode+26.364459%. Text TTFT-76.490891%,total-61.827662%,derived decode-28.498098%. Complete parent Skillwall45.214559 ->19.504299s(-56.862790%). These are unconfirmed single-screen joint candidate observations. The old deployed WS10 memory telemetry is unavailable/null; do not substitute the separate raw resource denominator. All first outputs/phase profiles/request counters are preserved separately, no favorable selection or exclusions.

Observed system-wide worker-window swap counter equivalents:

| Worker | Global swap-ins GiB | Global swap-outs GiB |
|---|---:|---:|
{swaptable}

vm_stat page size16384B; deltas cover initialization and requests, are system-wide/compressor page-counter equivalents, not per-process attribution or direct disk I/O, and do not identify a latency cause. All endpoints' AC/thermal tool outputs retained, not proof of GPU clocks or invariant thermal conditions. Task footprint/compression/resident fields and Metal allocation/recommended-working-set estimates are distinct, nonadditive and not peak/freeRAM measures.

## Decision, tradeoffs and coverage limits

REJECT this candidate. Favorable4B/Skill latency and memory do not override text decode failure. Request-scoped residency did not establish a practical fix under the complete original gates. It is not valid to compare this regression against WS54's8.54%as an isolated effect: cohorts/controls differ. Do not retry unchanged or loosen gates. Matched storage/raw screens, fresh mirrored full48, canonical rebuild/reproof and actual-production mirrored48 remain undispatched because this stage failed.

Export every first/warm/failure call once, parent Skillwall once per frame. Derived decode=(generated-1)/(total-TTFT), including host/stop/post-loop/possible extraEOS; not isolated GPU throughput. Profiles and per-call residuals are descriptive, component medians nonadditive. Fixed video frames traverse actual WorldSense Skill ->Galaxy ->Garnet; this screen excludes live decode/scheduling, event persistence/shared-state writes and retrieval/ranking. Four normal files are restored; second car view is correlated, not independent-camera fusion. M5/24GiB/4B remains separate from the unmet M4mini/small-model/vLLM handover. No consumed source/driver mutation, builds, competing investigation GPU jobs or archive copying during timing. Full optimization remains active.
'''
d.mkdir(parents=True,exist_ok=True);(d/'README.md').write_text(report);(root/'outputs/Garnet-4B-WS56-Result.txt').write_text(report);print('WS56 terminal rejection and paper-ready report recorded')
