"""Create a canonical verifier from original full proof/oracles, without inference."""
from pathlib import Path
import textwrap,ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');folder=b/'work/worldsense-opt'
prior=(folder/'verify_archive_ws68.py').read_text();text=prior[:prior.index('current,recon=')].replace('worldsense-m5-primary-simd-metadata','worldsense-m5-packed-request-residency')
text+='''current,recon=reconstruct('');legacy,legacy_recon=reconstruct('legacy-');assert len(current)==1721 and len(legacy)==1706
seed_ref='290f5742ba51c41197f24179cfee4876e96e789a';seed_prefix='results/20261017/worldsense-m5-request-weight-residency/'
seed,seed_recon=[json.loads(v)for v in batch([seed_ref+':'+seed_prefix+'correctness-source-manifest.json',seed_ref+':'+seed_prefix+'source-reconstruction.json'])]
assert seed==load(d/'seed-source-manifest.json')and len(seed)==1720
seed_source=dict(zip(seed,batch([(seed_ref+':'+seed_prefix+'source/'+rel)if rel in seed_recon['overlay_files']else seed_recon['base_commit']+':'+rel for rel in seed])))
assert all(hashlib.sha256(seed_source[k]).hexdigest()==v for k,v in seed.items())
delta=sorted(k for k,v in current.items()if seed_source.get(k)!=v)
assert delta==sorted(['cmake/GarnetMetal.cmake','runtime/metal_serving.py','src/backends/metal/decoder_session.cpp','src/backends/metal/decoder_session.h','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm','src/entry/metal_main.cpp','test2026/metal/packed_request_residency_test.cpp'])
source=load(d/'correctness-source-manifest.json')
'''
native=(folder/'verify_native_ws69.py').read_text();core=native[native.index("transforms=load(w/'source-transformations.json')"):native.index('assert all(sha(Path(k))')]
core=core.replace('load(w/','load(d/').replace('(s/rel).read_text()','current[rel].decode()').replace('(old/rel).read_text()','seed_source[rel].decode()')
for rel in ['src/backends/metal/metal_context.mm','src/backends/metal/decoder_session.cpp']:
 core=core.replace("(old/'"+rel+"').read_text()","seed_source['"+rel+"'].decode()").replace("(s/'"+rel+"').read_text()","current['"+rel+"'].decode()")
text+=core
text+='''roles=['off','primary','on','fault','compat56','compat56-fault']
assert load(d/'native-results.json')=={role:0 for role in roles}
import re,struct,copy
for role in roles:
 log=(d/('native-'+role+'.log')).read_text();assert sorted(int(v)for v in re.findall(r'\\d+/9\\s+Test\\s+#\\s*(\\d+):[^\\n]*Passed',log))==list(range(1,10))and re.search(r'100% tests passed(?:, 0 tests failed)? out of 9',log)
 cache=(d/'build-identities'/role/'CMakeCache.txt').read_text();assert ('GARNET_METAL_PACKED_REQUEST_WEIGHT_RESIDENCY:BOOL=ON'in cache)==(role in ['on','fault'])
 assert ('GARNET_METAL_PACKED_REQUEST_RESIDENCY_TEST_FAULTS:BOOL=ON'in cache)==(role=='fault')
 assert ('GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY:BOOL=ON'in cache)==role.startswith('compat56')
 assert ('GARNET_METAL_REQUEST_RESIDENCY_TEST_FAULTS:BOOL=ON'in cache)==(role=='compat56-fault')
raw=(d/'native-off-primary_projection_words.bin').read_bytes();assert raw==batch([seed_ref+':'+seed_prefix+'native-off-projection-words.bin'])[0]
assert all((d/('native-'+role+'-primary_projection_words.bin')).read_bytes()==raw for role in roles)
proof=load(d/'native-word-comparison.json');assert proof['records']==186 and proof['bytes']==len(raw)and proof['sha256']==hashlib.sha256(raw).hexdigest()
reference=None
for role in roles:
 data=(d/('native-'+role+'-packed_request_projection_words.bin')).read_bytes()
 if reference is None:reference=data
 assert data==reference;position=0
 for pattern,value in [(0,128.),(1,-1.25)]:
  assert struct.unpack('4Q',data[position:position+32])==(260,128,pattern,260);position+=32;assert data[position:position+1040]==struct.pack('<f',value)*260;position+=1040
 assert position==len(data)and(d/('native-'+role+'-packed_request_projection_words.bin.leased.bin')).read_bytes()==(data if role in ['on','fault']else b'')
packed=load(d/'packed-native-word-comparison.json');assert packed['records']==2 and packed['words_per_reference']==520 and packed['bytes']==len(reference)and packed['sha256']==hashlib.sha256(reference).hexdigest()
programs={role:{kind:(d/('actual-'+role+'-'+kind+'-program.metal')).read_bytes()for kind in ['main','prefill']}for role in roles}
assert all(programs[role]==programs['primary']for role in ['primary','on','fault'])
for role,previous in [('off','off'),('primary','primary'),('compat56','on'),('compat56-fault','fault')]:
 for kind in ['main','prefill']:assert programs[role][kind]==batch([seed_ref+':'+seed_prefix+'actual-'+previous+'-'+kind+'-program.metal'])[0]
assert all(not programs[role]['prefill']for role in ['off','primary','on','fault'])
for role,groups in load(d/'actual-program-proof.json')['programs'].items():
 for kind,info in groups.items():assert len(programs[role][kind])==info['bytes']and hashlib.sha256(programs[role][kind]).hexdigest()==info['sha256']
for name,info in load(d/'learned-guard-adaptation-proof.json').items():
 original=batch([seed_ref+':'+seed_prefix+name])[0].decode();value=(d/name).read_text();assert hashlib.sha256(original.encode()).hexdigest()==info['seed_sha256']and hashlib.sha256(value.encode()).hexdigest()==info['current_sha256']
 for op in reversed(info['operations']):assert value.count(op['after'])==op['count'];value=value.replace(op['after'],op['before'])
 assert value.replace('worldsense-packed-request-residency','worldsense-request-weight-residency')==original
'''
learned=(folder/'verify_learned_ws69.py').read_text();core=learned[learned.index("terminal=load(w/'learned-controller-terminal.json')"):learned.index('result=dict(')]
core=core.replace("w.parent/'worldsense-primary-blocks/matrix-selection.json'","d/'frozen-native-guards/matrix-selection.json'").replace("w.parent/'worldsense-prefix-cache/state-guard.json'","d/'frozen-native-guards/state-guard.json'").replace("w.parent/'worldsense-prefix-cache/prefix-model-guard.json'","d/'frozen-native-guards/prefix-model-guard.json'").replace("w.parent/'worldsense-request-residency/vision-grid-off.json'","d/'frozen-native-guards/vision-grid-original.json'").replace('load(w/','load(d/').replace('(w/','(d/')
text+='def canonical_learned_checks():\n'+textwrap.indent(core,' ')+'\ncanonical_learned_checks()\n'
text+='''full=load(d/'packed-full-session-guard.json');actual=full['actual_allocation_bytes'];assert actual==6405324800
whole=next(v for v in full['records']if v['case']=='whole_actual_201owners_402allocations_nested_and_global_budget');active=whole['active'];contender=whole['contender']
assert active['scope_open']and active['attached']and active['requested']and active['active_bytes']==active['global_used_bytes']==active['last_bytes']==actual and active['owner_count']==201 and active['set_allocation_count']==active['last_count']==402
assert contender['reason']==5 and contender['active_bytes']==0 and contender['global_used_bytes']==actual
thread=next(v for v in full['records']if v['case']=='cross_thread_same_runtime_budget_contention')['stats'];assert thread['reason']==5 and thread['active_bytes']==0 and thread['global_used_bytes']==actual
errors=[v for v in full['records']if v['case']=='native_image_error_outer_finally_cleanup'];assert len(errors)==2
for row in errors:
 assert row['error']['status']=='error';q=row['after'];assert q['calls']==q['admitted']==q['ends']and q['fallback']==0 and q['active_bytes']==q['global_used_bytes']==q['owner_count']==q['set_allocation_count']==0 and not q['scope_open']and not q['attached']and not q['requested']
assert load(d/'native-independent-audit.json')['observed_exit_code']==0
assert len(load(d/'flag-dependency-proof.json'))==12 and load(d/'admission-complete.json')['all_expected_config_rejections']
for name,info in load(d/'flag-dependency-proof.json').items():assert info['returncode']!=0 and info['expected_diagnostic']in' '.join((d/('invalid-'+name+'-config.log')).read_text().split())
metric_ref='b600cf8de6e99e11d35f6727bc3c3cacfc1d5418';metric_prefix='results/20261018/worldsense-m5-primary-simd-metadata/'
class Normalize(ast.NodeTransformer):
 def visit_FunctionDef(self,node):return None if node.name=='native_route'else self.generic_visit(node)
for rel in ['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py']:
 original=batch([metric_ref+':'+metric_prefix+rel])[0];assert original is not None
 assert ast.dump(Normalize().visit(ast.parse(original)),include_attributes=False)==ast.dump(Normalize().visit(ast.parse((d/rel).read_text().replace('worldsense-packed-request-residency','worldsense-primary-simd-metadata'))),include_attributes=False),rel
# Replay the original offline adversarial audit in a separate namespace, only
# against archived calls/oracle. No file writes or new model calls are executed.
adversarial=(d/'work/audit_routes_ws69.py').read_text();body=adversarial[adversarial.index("for rel in ['analyze_resource.py'"):adversarial.index("p=w/'acceptance-gate-audit.json'")]
body=body.replace("b/'work/worldsense-isolated-primary/production-speed/control-1.json'","w/'oracles/legacy-native-route-control.json'")
namespace=dict(w=d,b=b,load=load,results={},ast=ast,copy=copy,json=json);exec(compile(body,'archived adversarial audit','exec'),namespace)
assert namespace['results']==load(d/'acceptance-gate-audit.json')['native_route_audit']and sum(v['malformed_copies_rejected']for v in namespace['results'].values())==8149
# Every bundled external input is byte checked; omitted external dependencies
# retain terminal-local rehash records and must be supplied/revalidated for reuse.
resolution=load(d/'external-input-resolution.json')
for original,info in resolution.items():
 assert Path(original).name!='worldsense-http.py';target=info['bundled']
 if target is None:continue
 if target.startswith('reconstructed-candidate:'):assert hashlib.sha256(current[target.split(':',1)[1]]).hexdigest()==info['sha256']
 else:assert sha(d/target)==info['sha256']
for filename,key in [('native-input-identities-before-dispatch.json','files'),('learned-identities-before.json','external'),('driver-identities-before-dispatch.json','files'),('outer-method-verifier-identities-before-resource.json','files')]:
 for path,digest in load(d/filename)[key].items():assert resolution[path]['sha256']==digest
assert len(load(d/'build-host-dependencies-before-dispatch.json')['files'])==1877
'''
tail=prior[prior.index("gold=load(d/'oracles/worldsense-golden.json')"):prior.index('result=dict(')]
text+=tail
text+='''cleanup=list(csv.DictReader((d/'post-cleanup-memory-samples.csv').open()));leases=list(csv.DictReader((d/'packed-request-lifecycle.csv').open()));assert len(cleanup)==len(leases)==18
lease_rows={(v['stage'],v['worker'],int(v['repetition']),v['phase']):v for v in leases};cleanup_rows={(v['stage'],v['worker'],int(v['repetition']),v['phase']):v for v in cleanup}
for stage,label,index,phase,item,call in rawcalls:
 t=call['timings'];folder=d/'production-speed'if stage=='production-speed'else d/'functional-smoke'if stage=='functional-correctness-only'else d
 assert load(folder/(label+'.json'))['request']==load(folder/'cases.json')
 cohort=next(v for v in load(folder/'cohort-manifest.json')if v['label']==label);environment=load((d/'production-speed/environment.json')if stage=='production-speed'else d/'environment.json');enabled=label=='candidate-1'
 assert cohort['client_exit']==0 and cohort['library_sha256']==environment['libraries']['on'if enabled else'off']
 assert cohort['serving_sha256']==(load(d/'correctness-source-manifest.json')['runtime/metal_serving.py']if enabled or stage!='production-speed'else environment['control_serving_sha256'])
 if'packed_request_weight_residency'not in t:continue
 key=(stage,label,index,phase);row=lease_rows[key];assert row['packed_request_eligible']==str(t['packed_request_eligible'])
 for name,value in t['packed_request_weight_residency'].items():assert row[name]==str(value)
 for name,value in t['memory_after_packed_request_cleanup'].items():assert cleanup_rows[key][name]==str(value)
scope=load(d/'export-scope.json');assert scope['post_cleanup_rows']==18 and scope['memory_rows']==22 and scope['calls']==32
assert load(d/'restoration-verification.json')['benchmark_valid_request_http404']
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_files=len(manifest),source_files=1721,source_overlays=len(recon['overlay_files']),legacy_files=1706,legacy_overlays=len(legacy_recon['overlay_files']),native_CTest_pass=54,original186_complete_projection_streams_and_new2records520words_exact=True,original_programs_and_raw_policy_preserved=True,original_guard_and_all_five_metric_gate_ASTs_inverse_exact=True,original_five_full_learned_readers_and_two_leased_readers_states_tokens_vision_lifecycle_exact=True,actual4B_owners=201,actual4B_allocations=402,actual4B_allocation_bytes=actual,offline_malformed_copies_rejected=8149,all32_pipeline_goldens_native_routes_CSV_resource_speed_and_rejection_recomputed=True,calls=32,requests=16,memory_rows=22,post_cleanup_rows=18,external_input_resolutions=len(resolution),decision='valid-reject',promotion=False)
(b/'work/worldsense-opt/ws69-archive-independent-audit.json').write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result,indent=2))
'''
ast.parse(text);p=folder/'verify_archive_ws69.py';assert not p.exists();p.write_text(text)
print('Canonical WS69 source/native/learned/adversarial/32call original-gate verifier prepared; no inference')
