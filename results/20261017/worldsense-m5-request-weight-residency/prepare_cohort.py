"""Fail-fast provenance preparation only. Never starts a benchmark worker."""
from pathlib import Path
import json,hashlib,subprocess,shutil,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');n=r/'out/build/macos-arm64-release/bin'
assert all(v==0 for v in json.loads((w/'native-results.json').read_text()).values())
assert json.loads((w/'lifecycle-metadata-proof.json').read_text())['wide_admission_scope_verified']
assert json.loads((w/'lifecycle-metadata-proof.json').read_text())['cached_native_live_property_suite_ON_fault_executed']
assert all(x['expected_rejection']for x in json.loads((w/'flag-dependency-proof.json').read_text()).values())
assert json.loads((w/'independent-direct-upload-proof.json').read_text())['all546_learned_vision_norm_depthwise_patch_candidates_FP32_words_exact']
assert json.loads((w/'correctness-terminal.json').read_text())['all_three_actual_model_modes_and_five_readers_exact']
assert all(json.loads((w/'api-correctness-result.json').read_text()).values())
assert json.loads((w/'correctness-terminal.json').read_text())['all_three_actual_model_modes_and_five_readers_exact']
assert json.loads((w/'native-word-comparison.json').read_text())['all_four_builds_FP32_words_identical']
audit=json.loads((w/'acceptance-gate-audit.json').read_text())
assert all(audit[rel]['numeric_golden_source_metric_AST_unchanged']for rel in ['analyze_resource.py','speed/analyze.py','confirmation/analyze.py'])
assert audit['primary-speed/analyze.py']['shared_full_golden_input_model_metric_prefix_AST_unchanged']and audit['primary-speed/analyze.py']['no_original_stage_gate_relaxed']
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
 b=r/('out/build/garnet-metal-worldsense-request-weight-residency-'+mode)
 for rel,dest in [('CMakeCache.txt',mode+'-CMakeCache.txt'),('Testing/Temporary/LastTest.log',mode+'-LastTest.log'),('CMakeFiles/garnet_metal.dir/flags.make',mode+'-flags.make')]:
  assert (b/rel).exists(),(mode,rel)
  shutil.copy2(b/rel,w/dest)
(w/'cohort-preparation-complete.json').write_text(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_prerequisites_checked':True,'worker_dispatched':False},indent=2))
print('All prerequisites/source/model/build/dependencies/backup/request frozen; no worker dispatched',flush=True)

paths=[p for p in w.rglob('*')if p.is_file()and p.suffix in ['.py','.cmake']and '__pycache__'not in p.parts]
paths += [r/'WorldSense/tools/benchmark_recorded.py',n/'libgalaxy_garnetfilter.dylib',n/'libgarnet.dylib']
(w/'driver-identities-before-dispatch.json').write_text(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Hash-only dependency/driver identity capture before dispatch; no credentials or source contents copied','files':{str(p):sha(p)for p in paths}},indent=2))
