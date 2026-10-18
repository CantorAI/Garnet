"""Original resource8 requests/gates; explicit completed proof and INPUT closure."""
from pathlib import Path
import hashlib,json,subprocess,sys,datetime,shutil
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-primary-simd-metadata';r=Path('/Users/shawn/CantorAI');n=r/'out/build/macos-arm64-release/bin';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert load(w/'learned-independent-audit.json')['both_new_GPU_helpers_EACH_all201_original_FP32_words']==4200857600
assert load(w/'functional-smoke/functional-verification.json')['actualWorldSense_Galaxy_native_candidate_functional_pass']
assert load(w/'native-independent-audit.json')['native_tests_pass']==50
assert load(w/'admission-complete.json')['all_expected_config_rejections']
audit=load(w/'acceptance-gate-audit.json');assert audit['all_five_original_golden_numeric_metric_gate_AST_unchanged']and audit['full_route_mutation_audit_pass']
for rel,digest in audit['driver_sha256'].items():assert sha(w/rel)==digest
subprocess.run([sys.executable,str(w/'freeze.py')],check=True)
subprocess.run([sys.executable,str(w/'dependency_closure.py'),'capture'],check=True)
assert sha(n/'libgarnet.dylib')==sha(w/'before-libgarnet.dylib')==sha(r/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib')
cases=load(w/'cases.json');assert cases['repeats']==1;planned=dict(cases,repeats=4);(w/'planned-speed-screen-cases.json').write_text(json.dumps(planned,indent=2)+'\n')
(w/'resource-case-identity.json').write_text(json.dumps(dict(sha256=sha(w/'cases.json'),scope='Original resource onefirst+onefollowup per two freshworkers,8modelcalls,not speed evidence'),indent=2)+'\n')
sdk=r/'xlang3/sdk';headers={str(p.relative_to(sdk)):sha(p)for p in sorted((sdk/'xlang3').rglob('*'))if p.is_file()};assert len(headers)>10
(w/'external-dependencies-pre-cohort.json').write_text(json.dumps(dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sdk_headers=headers,runtime_binaries={k:sha(n/k)for k in ['xlang3','libxlang3_runtime.dylib']},xlang3_status=subprocess.check_output(['git','-C',str(r/'xlang3'),'status','--short'],text=True),xcode=subprocess.check_output(['xcodebuild','-version'],text=True)),indent=2)+'\n')
for role in ['off','primary','on','fault','compat67']:
 build=r/('out/build/garnet-metal-worldsense-primary-simd-metadata-'+role)
 for rel,name in [('CMakeCache.txt',role+'-CMakeCache.txt'),('Testing/Temporary/LastTest.log',role+'-LastTest.log'),('CMakeFiles/garnet_metal.dir/flags.make',role+'-flags.make')]:shutil.copy2(build/rel,w/name)
# Existing method/outer tools are pinned before resource, including verifiers.
outer={str(p):sha(p)for p in sorted((b/'work/worldsense-opt').glob('*ws68.py'))};(w/'outer-method-verifier-identities-before-resource.json').write_text(json.dumps(dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=outer,scope='All existing WS68 outer preparation/verifier helpers before first resource call. No generated result outputs.'),indent=2)+'\n')
files=[w/name for name in ['freeze.py','dependency_closure.py','controller.py','dispatch_prepared_resource.py','restore_normal.py','wait_cantor.py','analyze_resource.py','cases.json','starting-source-manifest.json','environment.json','build-flag-comparison.json','model-assets.json','resource-case-identity.json','acceptance-gate-audit.json','outer-method-verifier-identities-before-resource.json']]
files += [r/'WorldSense/tools/benchmark_recorded.py',n/'libgalaxy_garnetfilter.dylib',n/'libgarnet.dylib']
files += list((w/'build-profile').glob('*.cmake'))
(w/'driver-identities-before-dispatch.json').write_text(json.dumps(dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Enumerated actualresource INPUTS only, generated result files excluded. Each worker binds selected library and source SHA separately.',files={str(p):sha(p)for p in files}),indent=2)+'\n')
(w/'cohort-preparation-complete.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_prerequisites_checked=True,worker_dispatched=False),indent=2)+'\n');print('WS68 original resource8 and complete explicit runtime/dependency/driver identities prepared; no dispatch.')
