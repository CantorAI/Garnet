"""Report actual independently audited outcome; no preset failed-metric claim."""
from pathlib import Path
import datetime,json
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-owned-storage-local';load=lambda p:json.loads(p.read_text())
a=load(w/'production-speed/independent-screen-audit.json');speed=load(w/'production-speed/analysis.json');resource=load(w/'resource-independent-audit.json')
assert a['decision']=='valid-reject' and not a['promotion'] and not a['speed_passed']
failed=[]
for row in speed['comparisons']:
 for key,invalid in [('ttft_change',lambda x:x>.02),('total_change',lambda x:x>.02),('decode_gain',lambda x:x<-.02)]:
  if invalid(row[key]):failed.append(dict(model=row['model'],metric=key,relative_change=row[key]))
if speed['native_skill_wall_gain']<-.02:failed.append(dict(model='whole Skill',metric='wall_gain',relative_change=speed['native_skill_wall_gain']))
assert failed
report='''WS-M5-77: externally retained selected-buffer locals
Decision: VALID REJECTION under the original deployed-WS10 speed preservation
gates. Keep WS10. Finish this round, push, then PAUSE as requested by the user;
the full optimization objective remains incomplete. No merge or successor round.
Canonical archival/publication pending.

Method saved before implementation
Seed immutable WS74/e15c3f4325682098bd4c7b4a17da62254e1e36c5, 1721 files.
Source1722/four changed-or-added paths: two guarded local declarations in
metal_context.mm plus compile-time attribute availability guard; default-OFF
Metal CMake option/test registration; active-profile marker in metal_main.cpp;
new complete-output lifetime test. Inverse operations recover all seed bytes;
1718 other files exact. Shared_ptr w and ARC strong Impl fields stay unchanged.
Clang objc_externally_retained makes the selected buffer local const and omits
its own ARC init/destruction operations while the separate weight owner lives.
Constructor-only raw/payload/metadata assignments, private Impl visibility,
aliases/exception paths and ordinary retained command-buffer contract reviewed.
No unretained command buffers, unsafe_unretained, weak pointers, owner reset,
retainCount instrumentation or speculative resource lifetime assumptions added.
Shaders/program literals/FP32 math/geometry/codec/weights/metadata/graph/session/
image/runtime/xModel/CUDA/engine/original tests unchanged. No raw-wide-head policy
or GPU-counter instrumentation imported. Original201 packed matrices retained.

Eight builds93228exit0. Actual matched HOST ARM64 objects: raw OFF selection
window1retain/0release and packed CONTROL2retain/1release become enabled0/0;
shared_ptr copy/increment and later normal cleanup survive. This is static
call-site/window evidence, not runtime call counts, isolated cost, complete
error-path safety, GPUISA/register/occupancy or end-to-end benefit. Ownership
is also tested by the original and additional numerical/lifecycle checks.

Correctness and prospective gates
Native18779exit0 and first independent audit:80tests/eight roles. Original186
complete projection streams equal old references under unchanged FP64 bounds;
20core-isolation records/432804words and empty CORE/counters exact per role.
New120records/4487160FP32words per role independently recomputed byte-for-byte:
raw/packed aliases released before Prepare; last owner dropped before commit
and after Flush; creator context closed before same-device use; invalid shape/
active-Prepare; flushed Abort/owner release and recovery. All live logical
weight accounting restored. Every actual MAIN/prefill/CORE program equals74.
Roles OFF,PRIMARY(newflagON),ON(newflagON),FAULT(newflagON),original COMPAT56/
COMPAT56-FAULT(newflagOFF),matched minimal CONTROL(newflagOFF),RAW-BORROW(newflagON).
Raw-to-candidate whole-profile contrast is eight flags; PRIMARY-to-candidate
is three vector/isolation flags with owned-local/selective both enabled. No
claim of isolated annotation speed from these contrasts. Future feature-ON
compatibility deployments require additional proof; original compat is retained.

Full learned preparation45912 frozen before execution; full30456exit0 and
independent audit pass18 subprocess checks: original16 plus CONTROL/RAW-BORROW
API2. Five original GPU readers EACH4200857600words across201 packed matrices;
raw245 HOST1724383232 and direct546 HOST334609152. Coverage overlaps; no unique
sum. Full32VLM/28text states, tokens, outputs/configs in six API modes/30calls;
64private steps; trained linear/vision/grid/prefix/capacity/error/shared-session/
close/rebuild/late/invalid-input/original4GiBrequest/fault lifecycles. All original
mathematical guards invert exactly77->74->71->70->56. Eighteen original invalid
CMake configurations retained; strict owned-local marker added to existing
14field/5delta/COREzero/selective/foreign-capability/legacy route audit. Original
five quantitative/golden/metric/threshold ASTs unchanged outside native_route.

Actual WorldSense/Galaxy results
Functional4 candidate calls retain complete original Skill/frame/prompt/config/
tokens/text/routes/native-cpp image/source/library/memory correctness. Functional
timing is ineligible for speed comparison. Resource8 uses two fresh workers,
one first and one follow-up each, original >=10%Metal and >=5%footprint gates.
Recomputed whole-profile reductions versus fresh RAW:
'''
report+=json.dumps(resource['recomputed_reductions'],indent=2)+'\n\n'
report+='''Deployed WS10 speed20: two fresh sequential workers, five identical fixed-car
Skill requests each. First retained separately, four following samples per
model/worker form the medians. No instrumented diagnostic used for acceptance.
Model | TTFT seconds WS10 -> candidate | total seconds | derived tokens/s
'''
for model in ['Qwen3.5-4B','Qwen3-1.7B']:
 x,y=speed['metrics']['control-1'][model],speed['metrics']['candidate-1'][model]
 report+=f"{model} | {x['ttft']:.12f} -> {y['ttft']:.12f} | {x['total']:.12f} -> {y['total']:.12f} | {x['decode_tps']:.12f} -> {y['decode_tps']:.12f}\n"
report+='\nAll within-pair relative changes:\n'+json.dumps(speed['comparisons'],indent=2)+'\nWhole Skill seconds: '+json.dumps(speed['native_skill_wall_p50'])+'; wall gain='+repr(speed['native_skill_wall_gain'])+'.\nFailed unchanged gates:\n'+json.dumps(failed,indent=2)+'\n'
report+='''
No unchanged timing retry/favorable subset/pooled rescue/memory or correctness
override. Valid failure stops matched PRIMARY20,RAW20,mirrored48,integration/
canonical48/app-smoke promotion. All32pipelinecalls/16Skillrequests across
functional/resource/speed retained with first/followups, source/build/model/
input/runtime/SDK/driver/guard identities, raw logs and keep/reject decision.
Legacy memory/counters unavailable/null, never fabricated zero or replaced
with resource-cohort denominators. New method fails the full practical gate;
no inference about its isolated cost or previous GPU/telemetry causes.

Normal VLM64/text16/capacity2048/image65536/shared vision/prefix64tokens2entries
64MiB/original budgets/stops/Skill thresholds. Short proof budgets declared
separately. Every image freshly prepares/encodes in native C++/Metal/XLang3;
no external CPython model/image processing, answer/image cache, speculation
or camera. CPython only orchestration/audit. Four video files restored after
each owning stage. Private authentication helper never inspected/read/copied/
hashed/archived; only runpy login/request. No consumed edits during jobs;
timing has bounded observations only, no builds/otherGPU/hash sweeps/archives.
Same live handle polled to authoritative terminal. Derived decode=(generated-1)/
(total-TTFT), includes host/stop/postloop/possibleEOS/telemetry, not pureGPU.
GPU spans overlap, separate medians nonadditive. Memory snapshots not peaks/
free/unique/physical residency. First not hardware-cold; no significance/
clock/thermal/ISA/register/occupancy/causal claim. Fixed-frame Skills exclude
live decoding/scheduling, persisted events/shared states/dedup, retrieval/
ranking/navigation/independent-camera fusion; second car correlated. M5/24GiB/
4B does not satisfy separate M4mini/small-model/vLLM objective.
'''
(w/'result.txt').write_text(report);(w/'failed-speed-gates.json').write_text(json.dumps(failed,indent=2)+'\n')
(b/'outputs/Garnet-4B-WS77-Result.txt').write_text(report)
p=b/'work/worldsense-next.json';d=load(p);d.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),state='WS77_VALID_REJECTION_ARCHIVE_PUSH_THEN_PAUSE',live_job=None,promotion=False,failed_speed_gates=failed,next_actions=['Canonical source/native/learned/actual data/metric/archive audit; publish experimental source and research evidence on feature/qwen35-metal.','Verify clean pushed branch and normal best validatedWS10/fourfiles/noCam/normalbenchmark404; then update_goal paused as requested. No successor or merge.']);p.write_text(json.dumps(d,indent=2)+'\n')
print('WS77 actual rejection and all metrics recorded; canonical archive/push then user-requested pause pending.')
