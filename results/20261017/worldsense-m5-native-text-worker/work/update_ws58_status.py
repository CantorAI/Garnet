from pathlib import Path
import datetime,json
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=root/'work/worldsense-native-text-worker';out=root/'outputs'
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
status='''WS58 native gateway built; correctness validation in progress. No WS58 performance result or promotion. Full optimization remains active.

All four same-source roles built successfully (latest build session66604); source1723files. Parent native C++ gateway now implements local4B/RAW native XLang3 text-child routing, checked nativeLRPC, owned PID/birth identity, watchdog, explicit reload/no hidden inference retry, clocks, and conservative parent+child process memory accounting. No CPython model/image inference, no precision/math/CUDA/model/budget changes. Basic no-model lifecycle/error/restart/shutdown guard passed on/primary/fault.

First actual-model guard22936 terminated with a fixture schema KeyError: call. Historical prefix fixture has top-level generated_token_ids; complete call oracles come from validated54OFF API records. Failed executed driver, logs, identities and diagnostic limits are preserved in failed-text-guard-attempt-22936. Raw first response was lost before assertion; no full token or timing claim for that attempt. Controller restored exactWS10/fourfiles/noCam/benchmark404 PID391.

Repaired correctness driver checks both existing oracles, saves every response/state/error before assertions, and keeps numerical/performance gates unchanged. Three-role actual28-layer text/token/prefix/capacity/profile/clock/cohort-memory controller16193 is running. Source/libraries frozen during execution. No timing/cohort launched. Full original native/VLM/corpus/lifecycle/fault tests and all original staged actualWorldSense gates remain required.

WS57 rejected: text derived decode25.697835->18.542763tokens/s (-27.8431%) fails2%preservation despite4B/memory/Skill improvements.347SHA+manifest348tracked evidence pushed91b1eaded0ce26ad4d03210615626f175f095a1c. WS10 remains selected. Services temporarily paused for isolated correctness; restoration runs in finally and must be verified at terminal.
'''
(out/'4B-Latest-Result.txt').write_text(status)
with(out/'4B-Optimization-Progress.txt').open('a')as f:f.write('\n\n'+stamp+' '+status)
p=out/'Garnet-4B-Handover-Optimization.md';p.write_text('> Latest continuation ('+stamp+'): WS58 parent gateway implemented/source1723/all4buildsPASS/no-model three-role guardsPASS. Initial actualtextguard22936 failed fixture-schema lookup; preserved, not numerical/performance failure. Repaired controller16193 live, raw diagnostics saved before validation. No58speed result/promotion. Keep original full gates and WS10 selection; restore services afterterminal. Earlier notices below historical.\n\n'+p.read_text())
(w/'NEXT_IMPLEMENTATION.txt').write_text(status+'\nNext after text guard passes: original native7x4/186projection and shader/source inverse; full32VLM/28text/corpus/vision/fault/owner-death/clock/memory proofs. Only then 8resource calls followed by production-speed/primary-speed/raw/fullmirrored/canonical gates. Preserve all first/warm/failures; no favorable unchanged retries.\n')
p=root/'work/worldsense-next.json';previous=json.loads(p.read_text())if p.exists()else{};previous.update(utc=stamp,current_experiment='WS-M5-58',phase='native_gateway_actual_text_correctness_repaired',session_id=16193,source_files=1723,performance_result=False,promotion=False,goal_active=True);p.write_text(json.dumps(previous,indent=2)+'\n')
print('Updated current status and retained historical notices')
