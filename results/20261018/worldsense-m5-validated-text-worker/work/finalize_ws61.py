"""Record the terminal WS61 decision and paper-ready method, never promote."""
from pathlib import Path
import json, hashlib, datetime, shutil
base=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=base/'work/worldsense-validated-text-worker';repo=Path('/Users/shawn/CantorAI/Garnet');d=repo/'results/20261018/worldsense-m5-validated-text-worker'
load=lambda p:json.loads(p.read_text())
res=load(w/'resource-analysis.json');speed=load(w/'production-speed/analysis.json');post=load(w/'posthoc-text-decode-boundaries.json');normal=load(base/'work/worldsense-opt/ws61-normal-final.json')
assert res['passed']and not speed['passed']
assert load(w/'production-speed-terminal.json')['all_frozen_drivers_sources_models_libraries_and_dependencies_unchanged']
assert load(w/'resource-independent-audit.json')['no_new_inference_or_timing_retry']
assert load(w/'model-assets-after-screen.json')['files']==19
utc=datetime.datetime.now(datetime.timezone.utc).isoformat();gain=next(c['decode_gain']for c in speed['comparisons']if c['model']=='Qwen3-1.7B')
assert gain<-.02
decision=dict(experiment='WS-M5-61',utc=utc,decision='reject',reason=f'Text derived decode regressed {-gain*100:.6f}%, exceeding unchanged 2% preservation gate.',terminal_session=66771,terminal_exit_code=0,stage_outcomes={'resource':True,'actual_WS10_speed':False},promote=False,later_stages_dispatched=False,no_unchanged_retry=True,normal_restored=normal)
assert not(w/'decision.json').exists()
(w/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
shutil.copy2(w/'protocol.json',w/'initial-native-feasibility-protocol.json')
protocol=load(w/'protocol.json');protocol.update(phase='terminal_valid_rejection',initial_phase_history='initial-native-feasibility-protocol.json',parent_source_files=1726,legacy_source_files=1706,parent_source_created=True,no_Garnet_source_copy_or_edit_yet=False,no_model_or_performance_measurement_yet=False,prospective_parent_design='parent-integration-before-source.txt',original_gates_preserved=True,promotion=False)
(w/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
table='\n'.join(f"| {m} | {speed['metrics']['control-1'][m]['ttft']:.6f} → {speed['metrics']['candidate-1'][m]['ttft']:.6f} | {speed['metrics']['control-1'][m]['total']:.6f} → {speed['metrics']['candidate-1'][m]['total']:.6f} | {speed['metrics']['control-1'][m]['decode_tps']:.6f} → {speed['metrics']['candidate-1'][m]['decode_tps']:.6f} |"for m in ['Qwen3.5-4B','Qwen3-1.7B'])
a=post['workers']['control-1']['medians'];b=post['workers']['candidate-1']['medians']
report=f'''# WS-M5-61 — Pinned WS10 native text worker rejected

REJECT: the fresh actual WorldSense speed screen returned all 20 exact outputs, but text derived decode regressed {-gain*100:.6f}%, beyond the unchanged 2% preservation gate. No unchanged retry, matched-primary/raw/confirmation/canonical stages or promotion. WS10 remains selected. The optimization goal remains active.

## Hypothesis and architecture

WS58/59 native workers and WS60 raw lookahead failed text decode preservation. This distinct candidate keeps the exact validated WS10 DLL and its original text xModel in an owned native XLang3 child, beside the packed 4B parent. The old backend's performance in a child was unknown; neither earlier cohort ratios nor compatibility probes established it. Prospective design, instrumentation refinement and parent-integration refinement were saved before their respective source changes.

Parent source has 1,726 files, with six changed/new files versus immutable WS58: Metal CMake, entry marker, worker bootstrap/provenance, native instrumentation host, child adapter, and clock-only WS10 serving overlay. All shaders, codecs, arithmetic, graph/session/xModel/image code, CUDA and original tests remain exact WS58. Original Infer, Call, Stop, Serve, routing clocks and process-memory accounting bodies remain exact. CUDA execution is untested on this Mac; its source and build support remain preserved. No engine or SDK change.

The child uses the exact 1,706-file WS10 source and native DLL SHA dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72. The isolated serving overlay adds only already-existing start/first epochs to its return. A separate C++/Objective-C++ XLang3 module supplies native ownership/watchdog, clock, unique loaded-image/Load/ABI/path/SHA identity and real Mach/Metal process snapshots. Model/backend and instrumentation-host provenance are separate. The binary SHA is cached once before model startup, with complete before/after hashes; native image/symbol identity is checked on queries.

Old WS10 lacks packed-primary/lazy-retirement statistics. Those logical domains are structurally absent under the pinned source/binary, so their explicitly tagged counters are zero. This does not mean zero raw weights or zero memory. OS/Metal failures remain failures. No missing phase/init/peak observations are invented. An identical DLL copied to an unpinned path was rejected by all four native identity/memory/logical APIs; error origins were checked. These are fail-closed provenance proofs, not speed measurements.

RAW has all 11 method flags OFF. PRIMARY enables PRIMARY/WIDE/LAZY/WORKER/VALIDATED_RAW (five); candidate enables all 11. The matched-primary contrast remains the original six flags. FAULT adds existing test hooks only. This joint profile comparison is not isolated causal evidence about the old DLL, IPC or one kernel. CPython only orchestrates and audits; native Garnet C++/Metal and XLang3 handle model inference and image preparation. No camera or external CPython image/model helper.

## Correctness and measurement integrity

All four builds passed. Original seven native tests across four roles (28 runs) and 186 full projection records passed, preserving original FP64 bounds. The separately linked exact WS10 library passed 93 shapes/two inputs/186 records, bitwise equal to RAW-OFF, including tails, malformed controls, poison/fallback, queued ownership, Abort and retry. These synthetic records are not learned-weight or performance samples.

Full learned proof passed: each of five packed GPU readers checked 4,200,857,600 FP32 words over 201 packed matrices; 245 raw buffers checked 1,724,383,232 HOST words. Packed plus raw retained weights are 9,850,930,176 bytes. Direct reader checked 546 weights/334,609,152 words. Full 32 VLM/28 text states and generated tokens, trained layers, vision grids/embedding hashes/shared-owner close/repeat, ordinary/resident alternation, late private sessions, invalid inputs, close/rebuild, prefix hit/miss, capacity reset/recovery, reload/policy and original fault guards passed. All six Galaxy absent-image argument cases matched exact text states/tokens without hidden retry.

Native ownership passed setup, loaded-idle and actual active-inference parent death. A native decoder/graph stack was sampled before killing the owned parent; the watchdog stopped the same birth-bound child. Orphan exit status was unavailable, so the claim is observed termination, not a captured child exit code. Fresh post-fault inference reproduced all 28 states and tokens. Thirteen invalid configurations rejected. Four candidate-only actual WorldSense/Galaxy calls matched recorded Skill semantics, tokens, prompt/config/frame/Skill hashes, dual-backend identity and whole-process memory. Their timings are ineligible for speed.

All original numerical/golden/metric/threshold ASTs survive exactly after removing native admission and inverse-normalizing the explicit model/role serving-path selectors. Offline admission tested actual API records and 2,813 malformed copies. The first offline mutation fixture incorrectly assumed a child existed before text load; it failed on None and was preserved/repaired without inference or route changes.

The first resource analyzer stopped after successful workers because its legacy single-source serving-path assertion did not represent the declared child overlay. Original analyzer bytes/logs/frozen-driver inventory were preserved. A prospectively recorded post-terminal repair changed only exact model/role path selection. Independent byte inverse proves every other analyzer byte and every other frozen driver unchanged. The same original eight calls were analyzed; no new inference or timing retry. The fresh speed stage used the corrected identities before dispatch. Neither repair relaxed a numeric or performance gate or rewrote returned calls.

Source/library/runtime/driver/model/input identities were frozen and checked, including both source closures, native host libraries and 3,632 compiler/build/SDK/host/backend dependencies. All 19 learned assets were rehashed after the speed screen. During timing only bounded status observation was performed; no source edits, builds, competing investigation GPU work or archive copies.

## Actual WorldSense results

Original resource stage 8059: eight calls, two fresh workers, first plus one followup Skill each. Metal snapshot maximum {res['sampled_maximum_bytes']['control-1']['metal_allocated_bytes']:,} → {res['sampled_maximum_bytes']['candidate-1']['metal_allocated_bytes']:,} bytes ({res['reduction']['metal_allocated_bytes']*100:.6f}% reduction); sampled native-process footprint {res['sampled_maximum_bytes']['control-1']['task_physical_footprint_bytes']:,} → {res['sampled_maximum_bytes']['candidate-1']['task_physical_footprint_bytes']:,} bytes ({res['reduction']['task_physical_footprint_bytes']*100:.6f}% reduction). Original 10%/5% gates passed. Candidate parent and child are summed per category; shared pages can be double-counted. Device recommendation is counted once. Snapshots are not peaks, free RAM or unique physical residency. Resource control is the current same-source RAW build, not the separate selected WS10 speed control.

Fresh actual-WS10 speed stage 66771: 20 calls, two fresh workers, one first plus four warm identical fixed-car Skill requests each. All 20 outputs and native/source/profile/route checks passed. Warm medians:

| Model | TTFT seconds control → candidate | Total seconds control → candidate | Derived tokens/s control → candidate |
|---|---:|---:|---:|
{table}

Complete Skill wall {speed['native_skill_wall_p50']['control-1']:.6f} → {speed['native_skill_wall_p50']['candidate-1']:.6f} seconds ({speed['native_skill_wall_gain']*100:.6f}% improvement). Favorable 4B and Skill values do not override the text decode rejection. Legacy WS10 memory telemetry is unavailable/null; no resource-cohort memory ratio is imported into this speed cohort. First calls remain separately recorded. No promotion or independent confirmation is claimed.

Retained-record diagnostics: text decode-loop medians {a['decode_phase_seconds']:.6f} → {b['decode_phase_seconds']:.6f} seconds; reported GPU-span medians {a['decode_profile']['gpu_ms']:.6f} → {b['decode_profile']['gpu_ms']:.6f} ms. Candidate median gateway total minus raw-model total is {b['total_minus_raw_seconds']*1000:.6f} ms, computed per call before aggregation. The old DLL in the child still has larger decode-loop/GPU spans; gateway overhead alone does not explain the loss. No measured clock, thermal, VM, memory-residency, register or occupancy cause is established. GPU/wait spans overlap; medians are nonadditive. This is not a significance test or a cross-trial causal ranking.

Derived decode=(generated−1)/(total−TTFT), including host/stop/postloop/possible extra EOS/IPC/snapshot work, not isolated GPU throughput. Old readiness lacks separate startup/model-load decomposition; no hardware-cold claim. Fixed-frame coverage excludes live video decode/scheduling, persisted events, shared-world-state writes, retrieval and ranking. Four ordinary file sources remain mounted; the second car file is correlated and proves no independent-camera fusion. This M5/24 GiB/4B work is separate from the still-unmet M4 mini/small-model/vLLM handover objective.

Normal selected WS10 verified after terminal at PID {normal['pid']}; four files online, no camera, and a valid benchmark POST returns 404. Next work must use a distinct measured-bottleneck hypothesis; do not retry this valid failure unchanged.
'''
d.mkdir(parents=True,exist_ok=False);(d/'README.md').write_text(report);(w/'result.txt').write_text(report);(base/'outputs/Garnet-4B-WS61-Result.txt').write_text(report)
print('WS61 decision and full paper-ready method/results/tradeoffs recorded')
