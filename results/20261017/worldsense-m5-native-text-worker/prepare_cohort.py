"""Fail-fast provenance preparation only. Never starts a benchmark worker."""
from pathlib import Path
import json,hashlib,subprocess,shutil,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');n=r/'out/build/macos-arm64-release/bin'
assert json.loads((w/'admission-complete.json').read_text())['nine_expected_config_rejections']
assert json.loads((w/'correctness-terminal.json').read_text())['controller_session']==int(json.loads((w/'reproof-observed-terminal.json').read_text())['session_id'])
assert json.loads((w/'reproof-complete.json').read_text())['full_native_text_VLM_corpus_vision_lifecycle_and_parent_telemetry_reproof_pass']
assert json.loads((w/'post-owner-death-recovery.json').read_text())['exact_28layer_native_state_tokens_text_prompt_profile_after_active_owner_death_pass']
assert json.loads((w/'acceptance-gate-audit.json').read_text())['all_five_original_golden_numeric_metric_gate_AST_unchanged']
assert json.loads((w/'acceptance-gate-audit.json').read_text())['full_route_mutation_audit_pass']
subprocess.run(['python3',str(w/'freeze.py')],check=True)
subprocess.run(['python3',str(w/'dependency_closure.py'),'capture'],check=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(n/'libgarnet.dylib')==sha(r/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib')
shutil.copy2(n/'libgarnet.dylib',w/'before-libgarnet.dylib')
c=json.loads((w/'cases.json').read_text());c['repeats']=4;(w/'planned-speed-screen-cases.json').write_text(json.dumps(c,indent=2));c['repeats']=1;(w/'cases.json').write_text(json.dumps(c,indent=2))
(w/'resource-case-identity.json').write_text(json.dumps({'sha256':sha(w/'cases.json'),'scope':'1warm+1followup per fresh worker;8calls;not speed evidence'},indent=2))
sdk=r/'xlang3/sdk';headers={str(p.relative_to(sdk)):sha(p)for p in sorted((sdk/'xlang3').rglob('*'))if p.is_file()};assert len(headers)>10
(w/'external-dependencies-pre-cohort.json').write_text(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Before benchmark dispatch; preparation process never dispatches benchmark','sdk_headers':headers,'runtime_binaries':{k:sha(n/k)for k in ['xlang3','libxlang3_runtime.dylib']},'xlang3_status':subprocess.check_output(['git','-C',str(r/'xlang3'),'status','--short'],text=True),'xcode':subprocess.check_output(['xcodebuild','-version'],text=True)},indent=2))
for mode in ['off','primary','on','fault']:
 b=r/('out/build/garnet-metal-worldsense-native-text-worker-'+mode)
 for rel,dest in [('CMakeCache.txt',mode+'-CMakeCache.txt'),('Testing/Temporary/LastTest.log',mode+'-LastTest.log'),('CMakeFiles/garnet_metal.dir/flags.make',mode+'-flags.make')]:
  assert (b/rel).exists(),(mode,rel)
  shutil.copy2(b/rel,w/dest)
(w/'cohort-preparation-complete.json').write_text(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_prerequisites_checked':True,'worker_dispatched':False},indent=2))
print('All prerequisites/source/model/build/dependencies/backup/request frozen; no worker dispatched',flush=True)

paths=[p for p in w.rglob('*')if p.is_file()and p.suffix in ['.py','.cmake']and '__pycache__'not in p.parts]
paths += [r/'WorldSense/tools/benchmark_recorded.py',n/'libgalaxy_garnetfilter.dylib',n/'libgarnet.dylib']
(w/'driver-identities-before-dispatch.json').write_text(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Hash-only dependency/driver identity capture before dispatch; no credentials or source contents copied','files':{str(p):sha(p)for p in paths}},indent=2))
