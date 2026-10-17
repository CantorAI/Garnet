from pathlib import Path
import ast,json,shutil
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');old=root/'work/worldsense-isolated-primary';w=root/'work/worldsense-request-weight-residency'
names=['verify_direct_upload.py','prepare_admission.py','freeze.py','prepare_cohort.py','dependency_closure.py','run_resource.py','analyze_resource.py','audit_gates.py','controller.py']
for folder in ['speed','primary-speed','production-speed','confirmation']:
 names += [str(p.relative_to(old))for p in (old/folder).glob('*.py')]
for rel in names:
 p=w/rel;assert not p.exists(),rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text((old/rel).read_text().replace('worldsense-isolated-primary','worldsense-request-weight-residency'))
p=w/'verify_direct_upload.py';s=p.read_text().replace("seed=json.loads((w/'ws53-seed-source-manifest.json').read_text())","seed=json.loads((w/'ws54-seed-source-manifest.json').read_text())").replace("prior=w.parent/'worldsense-wide-cached-prefill'","prior=w.parent/'worldsense-isolated-primary'")
s=s.replace("['full_context_recovered_by_only_library_composition_and_lookup_inverse']","['original_context_and_session_code_exact_after_lifecycle_inverse']")
s=s.replace("for mode in ['off','primary']:assert p['programs'][mode]==before['programs'][mode]","assert p['programs']==before['programs']").replace("assert p['programs']['on']['main']==before['programs']['off']['main']","assert p['programs']['on']['main']==before['programs']['on']['main']")
s=s.replace('exact_WS53','exact_WS54').replace('byte_exact_WS53','byte_exact_WS54');p.write_text(s)
p=w/'prepare_admission.py';s=p.read_text().replace('len(source)==1718','len(source)==1720').replace("native['all_four_builds_seven_native_tests_pass']","native['original7_plus_request_native_each4builds_pass']").replace('predeclared54design','predeclared56design')
needle="('full-secondary',['-DGARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY=ON','-DGARNET_METAL_PRIMARY_VECTOR_PREFILL=ON','-DGARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY=OFF'],'Full primary isolation requires primary weights and isolated vector prefill')"
assert s.count(needle)==1;s=s.replace(needle,needle+",('request-fault',['-DGARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY=OFF','-DGARNET_METAL_REQUEST_RESIDENCY_TEST_FAULTS=ON'],'Request residency test hooks require request scope residency')")
s=s.replace("assert all(x['returncode']==0", "assert json.loads((w/'request-guards-result.json').read_text())['all_extra_request_trained_guards_pass']\nassert all(x['returncode']==0",1);p.write_text(s)
p=w/'freeze.py';s=p.read_text().replace("'GARNET_METAL_PRIMARY_BF16_BLOCKS':('OFF','ON'),","'GARNET_METAL_PRIMARY_BF16_BLOCKS':('OFF','ON'),'GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY':('OFF','ON'),",1)
s=s.replace("'GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY']}\n","'GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY','GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY']}\n")
s=s.replace('WS54 fully-isolated','WS56 request-residency fully-isolated').replace('rawcandidate9flags andstoragecontrolcandidate6','rawcandidate10flags andstoragecontrolcandidate7')
a=s.index("s=(w.parent/'worldsense-request-residency/controller.py')");b=s.index("print('Frozen source",a);s=s[:a]+s[b:];p.write_text(s)
# Retain the same metric/golden/gate AST; only extend native admission checks.
route_check=''' if ('request_scope_weight_residency'in call['backend'])!=enabled:return False
 if enabled:
  q=t.get('request_weight_residency',{})
  if not q.get('enabled')or q.get('scope_open',True)or q.get('active_bytes',-1)!=0 or q.get('global_used_bytes',-1)!=0 or q.get('calls',0)<=0 or q.get('ends')!=q['calls']or q.get('limit_bytes')!=4294967296:return False
  if call['model_id']=='Qwen3-1.7B':
   if q.get('reason')!=1 or q.get('admitted')!=q['calls']or q.get('fallback')!=0 or q.get('last_count')!=197 or q.get('last_bytes')!=3440902144 or q.get('coverage_cache_hits',0)<=0 or q.get('coverage_fallbacks')!=0 or not 0<q.get('last_coverage_graphs',0)<=256:return False
  elif call['model_id']=='Qwen3.5-4B':
   if q.get('reason')!=7 or q.get('admitted')!=0 or q.get('fallback')!=q['calls']:return False
  else:return False
 elif 'request_weight_residency'in t:return False
'''
for rel in ['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py']:
 p=w/rel;s=p.read_text();needle=" packed=enabled or primary_control;t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta']\n";assert s.count(needle)==1;s=s.replace(needle,needle+route_check,1)
 if rel=='production-speed/analyze.py':
  s=s.replace("'wide_primary_storage','direct_float_weight_upload'])","'wide_primary_storage','direct_float_weight_upload','request_scope_weight_residency'])").replace("'memory_after_decode'])","'memory_after_decode','request_weight_residency'])")
 p.write_text(s)
p=w/'primary-speed/prepare.py';s=p.read_text().replace("'GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY':('OFF','ON')}","'GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY':('OFF','ON'),'GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY':('OFF','ON')}").replace('Sixflags(vector/isolation/directupload/K32/cached/full-isolation)','Sevenflags(vector/isolation/directupload/K32/cached/full-isolation/request-residency)');p.write_text(s)
# Observe power and pressure outside inference; these introduce no exclusions.
p=w/'controller.py';s=p.read_text();s=s.replace("token=start(label);logfile=", "thermal=command(['/usr/bin/pmset','-g','therm'],False);state['thermal_before']=dict(status=thermal.returncode,output=thermal.stdout)\n  token=start(label);logfile=")
s=s.replace("state['ps_after']=", "thermal=command(['/usr/bin/pmset','-g','therm'],False);state['thermal_after']=dict(status=thermal.returncode,output=thermal.stdout)\n  state['ps_after']=")
p.write_text(s)
for p in w.rglob('*.py'):ast.parse(p.read_text(),str(p))
print('WS56 drivers prepared; audit and freeze still required, no timing worker dispatched')
