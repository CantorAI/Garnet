"""Freeze the prospectively declared paired actual-WorldSense capture."""
from pathlib import Path
import ast,datetime,hashlib,json,plistlib,subprocess,sys
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-paired-gpu-profile';p=w/'capture';r=Path('/Users/shawn/CantorAI');proof=b/'work/worldsense-paired-gpu-profile-current'
load=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
audit=load(proof/'learned-independent-audit.json')
assert audit['original_five_full_GPU_readers_EACH_words']==4200857600 and audit['legacy_three_API_modes15calls_and_three_vision_modes_exact']and audit['both_actual_learned_counter_errors_scope_cleanup_and28state_recovery_exact']
assert not p.exists();p.mkdir()
external=r/'out/research/worldsense-paired-gpu-profile/capture';external.mkdir(parents=True,exist_ok=False)
cases=load(w/'cases.json');cases['repeats']=1;assert not cases['queries']and len(cases['cases'])==1
(p/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
seed=b/'work/worldsense-minimal-selective-defaults/production-speed/analyze.py';text=seed.read_text();fn=next(node for node in ast.parse(text).body if isinstance(node,ast.FunctionDef)and node.name=='native_route')
route=ast.get_source_segment(text,fn)+'\n';(p/'native_route_admission.py').write_text(route)
namespace={};exec(compile(ast.parse(route),'unchanged original74 admission','exec'),namespace)
positives=[]
for family,folder,mode in [('current',proof,'on'),('legacy',b/'work/worldsense-paired-gpu-profile-legacy','trace')]:
 rows=load(folder/('api-state-progress-'+mode+'.json'))['records']
 assert len(rows)==5
 for row in rows:
  call=row['call'];assert namespace['native_route'](call,family=='current')and type(call['timings']['counter_trace_request_id'])is int and call['timings']['counter_trace_request_id']>0
  assert Path(call['timings']['serving_source']).resolve()==r/('out/research/worldsense-paired-gpu-profile-'+family)/'source/runtime/metal_serving.py'
  positives.append(dict(family=family,model=call['model_id'],original_native_route=True,trace_identity=True))
(p/'actual_API_native_admission_preflight.json').write_text(json.dumps(dict(existing_actual_API_calls_reused=10,no_new_inference=True,original74_function_exact=True,source_specific_positive_routes=positives),indent=2)+'\n')
learned_counter=load(w/'learned-counter-command-proof.json');counts={}
for route_proof in learned_counter['proofs']:
 for window in route_proof.get('windows',[]):counts.setdefault(window['model'],set()).add(window['samples'])
assert counts=={'Qwen3.5-4B':{749},'Qwen3-1.7B':{537}}
original=Path('/Users/shawn/Library/LaunchAgents/com.cantorai.cantor.plist');normal=plistlib.loads(original.read_bytes());assert set(normal.get('EnvironmentVariables',{}))=={'PATH'}
workers=[]
for label in ['legacy-1','current-1','current-2','legacy-2']:
 family=label.split('-')[0];trace=external/(label+'.counters.jsonl');assert not trace.exists()
 plist=plistlib.loads(original.read_bytes());plist['EnvironmentVariables'].update(GARNET_METAL_COUNTER_TRACE='1',GARNET_METAL_COUNTER_TRACE_FILE=str(trace))
 target=p/(label+'.plist');target.write_bytes(plistlib.dumps(plist))
 source=r/('out/research/worldsense-paired-gpu-profile-'+family)/'source';library=r/('out/build/garnet-metal-worldsense-paired-gpu-profile-'+family+'-trace')/'libgarnet.dylib'
 cache=(library.parent/'CMakeCache.txt').read_text();assert 'GARNET_METAL_GPU_PASS_COUNTER_PROFILE:BOOL=ON'in cache and 'GARNET_METAL_GPU_PASS_COUNTER_TEST_FAULTS:BOOL=OFF'in cache
 if family=='current':assert 'GARNET_METAL_PRIMARY_TEST_FAULTS:BOOL=OFF'in cache
 workers.append(dict(label=label,family=family,library=str(library),library_sha256=sha(library),serving_source=str(source/'runtime/metal_serving.py'),serving_sha256=sha(source/'runtime/metal_serving.py'),source_root=str(source),trace=str(trace),plist=str(target)))
subprocess.run([sys.executable,str(w/'capture_dependencies.py'),'capture'],check=True)
frozen=load(proof/'learned-identities-before.json');assert all(sha(Path(k))==v for k,v in frozen['external'].items())
for family,manifest in [('current',frozen['source']),('legacy',frozen['legacy_source'])]:
 source=r/('out/research/worldsense-paired-gpu-profile-'+family)/'source';assert all(sha(source/k)==v for k,v in manifest.items())
paths=[w/name for name in ['capture_dependencies.py','restore_normal.py','wait_cantor.py','before-libgarnet.dylib','protocol.json','native-design-before-implementation.txt','full-proof-addendum-before-supplemental-build.txt','learned-plan-before-helper-adaptation.txt','learned-build-aliases.json','learned-counter-command-proof.json','normal-after-learned.json']]
paths += [proof/'learned-independent-audit.json',proof/'learned-identities-before.json',original,p/'cases.json',p/'native_route_admission.py',p/'actual_API_native_admission_preflight.json',seed,r/'WorldSense/tools/benchmark_recorded.py',r/'Garnet/results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json']
paths += [b/'work/worldsense-opt'/name for name in ['prepare_capture_ws76.py','run_capture_ws76.py','verify_capture_ws76.py','summarize_counter_ws76.py']]
paths += [Path(item['plist'])for item in workers]
assert all(path.name!='worldsense-http.py'for path in paths)
frozen['external'].update({str(path):sha(path)for path in paths})
frozen.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),workers=workers,original_launchd_plist=str(original),scope='Before first actual paired capture: both source closures/all19model assets/tenlibraries/mathguards/SDKhost closure and ALL consumed capture/restore/admission/analysis/outer verifier drivers. Learned corpus evidence and recipes retained. No private authentication helper copied/read/hashed. No throughput acceptance or promotion.',actual_model_calls=16,actual_Skill_requests=8,expected_counter_commands=48,prior_learned_pass_counts={k:next(iter(v))for k,v in counts.items()},original_native_route_function_sha256=hashlib.sha256(route.encode()).hexdigest())
(p/'identities-before-dispatch.json').write_text(json.dumps(frozen,indent=2)+'\n')
print('Paired LEGACY/CURRENT/CURRENT/LEGACY actualWorldSense16call/48window capture frozen; no worker dispatched')
