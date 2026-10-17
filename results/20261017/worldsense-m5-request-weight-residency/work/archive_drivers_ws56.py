from pathlib import Path
import ast
here=Path(__file__).resolve().parent
s=(here/'archive_isolated_primary.py').read_text().replace('WS54','WS56').replace('WS-M5-54','WS-M5-56').replace('worldsense-isolated-primary','worldsense-request-weight-residency').replace('worldsense-m5-isolated-primary','worldsense-m5-request-weight-residency')
s=s.replace("'confirmation']:","'confirmation','initial-native-attempt']:").replace("'.f32','.cmake']:","'.f32','.cmake','.cpp','.make','.metal']:")
s=s.replace("['primary_storage_after','primary_encoded_delta','memory_after_decode']","['primary_storage_after','primary_encoded_delta','memory_after_decode','request_weight_residency']")
a=s.index("for name in ['archive_isolated_primary.py'");b=s.index("(d/'file-manifest.json')",a)
s=s[:a]+'''for name in ['archive_request_weight_residency.py','verify_request_weight_residency.py','design_ws56.py','seed_ws56.py','context_ws56.py','session_ws56.py','full_drivers_ws56.py','drivers_ws56.py','finalize_ws56.py','archive_drivers_ws56.py']:
    p=Path(__file__).parent/name
    if p.exists():copy(p,d/'work'/name)
'''+s[b:]
s=s.replace('prospective53 order','prospective56 order')
ast.parse(s);(here/'archive_request_weight_residency.py').write_text(s)
s=(here/'verify_isolated_primary.py').read_text().replace('WS54','WS56').replace('WS-M5-54','WS-M5-56').replace('worldsense-isolated-primary','worldsense-request-weight-residency').replace('worldsense-m5-isolated-primary','worldsense-m5-request-weight-residency').replace('==10 and len(closure)==1718','==15 and len(closure)==1720')
s=s.replace("['full_context_recovered_by_only_library_composition_and_lookup_inverse']","['original_context_and_session_code_exact_after_lifecycle_inverse']").replace("['raw_and_storage_control_programs_byte_exact_WS53']","['raw_and_storage_control_programs_byte_exact_WS54']")
a=s.index('# Reconstruct immutable53 context');b=s.index("for mode in ['on','fault']:\n assert 'Direct float upload",a)
s=s[:a]+'''# Independently reconstruct the exact WS54 seed and WS56 candidate, then
# execute the archived exact-inverse auditor with output writes disabled.
import tempfile
seedref='59a882e9765d4c1f61cd6bc117aa620dd06abd16'
seedroot='results/20261017/worldsense-m5-isolated-primary/'
seedrecon=json.loads(subprocess.check_output(['git','-C',str(r),'show',seedref+':'+seedroot+'source-reconstruction.json']))
seedmanifest=load(d/'ws54-seed-source-manifest.json')
assert len(seedmanifest)==1718
with tempfile.TemporaryDirectory(prefix='garnet-ws56-proof-')as scratch:
 scratch=Path(scratch)
 for label,inventory,overlays in [('candidate',closure,reconstruction['overlay_files']),('seed',seedmanifest,seedrecon['overlay_files'])]:
  references=[(seedref+':'+seedroot+'source/'+rel)if label=='seed'and rel in overlays else base+':'+rel for rel in inventory]
  data=subprocess.check_output(['git','-C',str(r),'cat-file','--batch'],input=('\\n'.join(references)+'\\n').encode());position=0
  for rel,digest in inventory.items():
   end=data.index(b'\\n',position);header=data[position:end].split();position=end+1
   if header[-1]==b'missing':payload=None
   else:size=int(header[2]);payload=data[position:position+size];position+=size+1
   if label=='candidate'and rel in overlays:payload=(d/'source'/rel).read_bytes()
   assert payload is not None and hashlib.sha256(payload).hexdigest()==digest
   dest=scratch/label/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
  assert position==len(data)
 programs=scratch/'programs';programs.mkdir()
 for mode in ['off','primary','on','fault']:
  for part in ['main','prefill']:
   name=f'actual-{mode}-{part}-program.metal'
   (programs/name).write_bytes(subprocess.check_output(['git','-C',str(r),'show',seedref+':'+seedroot+name]))
 code=(d/'verify_scope.py').read_text().replace("r/'out/research/worldsense-request-weight-residency/source'",'Path('+repr(str(scratch/'candidate'))+')').replace("r/'out/research/worldsense-isolated-primary/source'",'Path('+repr(str(scratch/'seed'))+')')
 output="(w/'wide-scope-proof.json').write_text(json.dumps(result,indent=2)+'\\\\n');print(json.dumps(result))"
 assert code.count(output)==1
 code=code.replace("w.parent/'worldsense-isolated-primary'",'Path('+repr(str(programs))+')')
 code=code.replace(output,"assert result==load(w/'wide-scope-proof.json')",1)
 exec(compile(code,str(d/'verify_scope.py'),'exec'),{'__file__':str(d/'verify_scope.py'),'load':load})
assert len(load(d/'flag-dependency-proof.json'))==7 and all(x['expected_rejection']for x in load(d/'flag-dependency-proof.json').values())
assert load(d/'request-guards-terminal.json')['terminal_exit_code']==0
assert load(d/'request-guards-result.json')['all_extra_request_trained_guards_pass']
assert load(d/'native-terminal.json')['original7_plus_request_native_each4builds_pass']
assert all(v==0 for v in load(d/'native-results.json').values())
for mode in ['off','primary','on','fault']:
 log=(d/(mode+'-LastTest.log')).read_text();assert 'metal_request_residency_reference'in log
assert load(d/'new-test-fixture-erratum.json')
initial=load(d/'initial-native-attempt/correctness-source-manifest.json')
assert {rel for rel,h in closure.items()if initial.get(rel)!=h}=={'test2026/metal/request_residency_test.cpp'}
assert sha(d/'initial-native-attempt/request_residency_test.cpp')==initial['test2026/metal/request_residency_test.cpp']
'''+s[b:]
# Check the native request lifecycle telemetry independently of model timings.
needle="                    source=env['control_source']"
idx=s.index(needle)
addition='''                    assert ('request_scope_weight_residency'in c['backend'])==enabled
                    if enabled:
                        q=tt['request_weight_residency'];assert q['enabled']and not q['scope_open']and q['active_bytes']==q['global_used_bytes']==0 and q['calls']==q['ends']and q['calls']>0 and q['limit_bytes']==4294967296
                        if c['model_id']=='Qwen3-1.7B':assert q['reason']==1 and q['admitted']==q['calls']and q['fallback']==0 and q['last_count']==197 and q['last_bytes']==3440902144 and q['coverage_cache_hits']>0 and q['coverage_fallbacks']==0 and 0<q['last_coverage_graphs']<=256
                        else:assert q['reason']==7 and q['admitted']==0 and q['fallback']==q['calls']
                    else:assert 'request_weight_residency'not in tt
'''
s=s[:idx]+addition+s[idx:]
s=s.replace("['primary_storage_after','primary_encoded_delta','memory_after_decode']","['primary_storage_after','primary_encoded_delta','memory_after_decode','request_weight_residency']")
point=s.index("for name in ['calls.csv','execution-routes.csv']:")
s=s[:point]+'''# Recompute descriptive system counters and per-call phase residuals from raw workers.
import re
post=load(d/'posthoc-observations.json');assert post['descriptive_only']
for worker in load(d/'production-speed/cohort-manifest.json'):
 label=worker['label'];system=post['system_observations'][label]
 size=int(re.search(r'page size of (\\d+) bytes',worker['vm_stat_before']).group(1))
 def counters(text):return {k:int(v)for k,v in re.findall(r'^([^:\\n]+):\\s+(\\d+)\\.',text,re.M)}
 before,after=counters(worker['vm_stat_before']),counters(worker['vm_stat_after'])
 delta={k:after[k]-before[k]for k in ['Swapins','Swapouts','Pageouts','Compressions','Decompressions']}
 assert system['page_size_bytes']==size and system['global_counter_deltas']==delta
 assert system['global_swapin_counter_bytes']==delta['Swapins']*size and system['global_swapout_counter_bytes']==delta['Swapouts']*size
 for key in ['swap_before','swap_after','power_before','power_after','thermal_before','thermal_after']:assert system[key]==worker[key]
 data=load(d/'production-speed'/(label+'.json'))
 for model in ['Qwen3.5-4B','Qwen3-1.7B']:
  calls=[(i,c)for i in data['items']for c in i['result']['model_response'].values()if isinstance(c,dict)and c.get('model_id')==model];warm=[c for i,c in calls if not i['warmup']]
  summary=post['models'][label][model]
  assert post['first_after_initialization'][label][model]=={k:calls[0][1][k]for k in ['ttft_seconds','total_seconds','generated_tokens']}
  for key in ['decode_seconds','prefill_seconds','reset_seconds']:assert summary[key]==statistics.median(c['timings'][key]for c in warm)
  assert summary['post_decode_residual_seconds']==statistics.median(c['total_seconds']-c['ttft_seconds']-c['timings']['decode_seconds']for c in warm)
  assert summary['decode_profile']=={k:statistics.median(c['timings']['decode_profile'][k]for c in warm)for k in ['prepare_ms','encode_ms','gpu_ms','wait_ms']}
  if label.startswith('candidate'):assert summary['request_counters']==[c['timings']['request_weight_residency']for c in warm]
'''+s[point:]
a=s.index("decision=load(d/'decision.json')");s=s[:a]+'''decision=load(d/'decision.json');assert decision['stage_outcomes']==outcomes
stage_order=load(d/'protocol.json')['stage_order'];reached=[]
for stage in stage_order:
 if stage not in outcomes:break
 reached.append(stage)
 terminal=load(d/(stage+'-terminal.json'));assert terminal['terminal_exit_code']==0 and terminal['controller_session']>0
 if not outcomes[stage]:break
assert set(outcomes)==set(reached), 'Stages after a valid failure are forbidden'
assert decision['prototype_confirmed']==(len(expected)==116 and all(outcomes.values())and len(outcomes)==6)
assert not decision['promote'],'Canonical production acceptance is a separate required archive'
assert load(d/'restoration-verification.json')['benchmark_valid_request_http404']
print(json.dumps(dict(files=len(manifest),source_files=len(closure),source_overlay=15,benchmark_calls=len(expected),native_requests=len(parents),stage_outcomes=outcomes,exact_outputs=True,request_lifecycle_exact=True,decision_consistent=True,scope='Independent artifact/source/call/parent-request/output/metric audit; original native tests retain their recorded numerical scope.')))
'''
ast.parse(s);(here/'verify_request_weight_residency.py').write_text(s)
print('WS56 archive/audit drivers drafted, not executed; terminal decision required.')
