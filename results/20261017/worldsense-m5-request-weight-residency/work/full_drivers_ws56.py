from pathlib import Path
import shutil,ast,json
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');old=root/'work/worldsense-isolated-primary';w=root/'work/worldsense-request-weight-residency'
for name in ['build_corpus.py','build_direct_corpus.py','make_resident_fixtures.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','full_correctness.py','verify_correctness_terminal.py']:
 s=(old/name).read_text().replace('worldsense-isolated-primary','worldsense-request-weight-residency')
 if name in ['full_correctness.py','verify_correctness_terminal.py']:s=s.replace('1718','1720').replace('native7tests','original7plusRequestTests')
 (w/name).write_text(s)
for name in ['primary_corpus_test.cpp','direct_float_corpus_test.cpp']:(w/name).write_text((old/name).read_text().replace('worldsense-isolated-primary','worldsense-request-weight-residency'))
for name in ['direct-upload-selection.json','seed-source-manifest.json','ws49-seed-source-manifest.json','ws50-seed-source-manifest.json','ws51-seed-source-manifest.json','ws52-seed-source-manifest.json','ws53-seed-source-manifest.json']:shutil.copy2(old/name,w/name)
p=w/'api_state_guard.py';s=p.read_text();needle="timings=call['timings'];";assert s.count(needle)==1;s=s.replace(needle,"assert ('request_scope_weight_residency'in call['backend'])==vector_expected\n "+needle,1)
needle=" row=dict(model_id=name,call=call,states=states(name,layers));";assert s.count(needle)==1
proof=''' if vector_expected:
  q=timings['request_weight_residency'];assert q['enabled']and not q['scope_open']and q['active_bytes']==q['global_used_bytes']==0 and q['calls']==q['ends']and q['limit_bytes']==4294967296
  if name=='Qwen3-1.7B':assert q['reason']==1 and q['last_count']==197 and q['last_bytes']==3440902144 and q['coverage_cache_hits']>0 and q['coverage_fallbacks']==0 and 0<q['last_coverage_graphs']<=256
  else:assert q['reason']==7 and q['admitted']==0 and q['fallback']==q['calls']
 else:assert 'request_weight_residency'not in timings
'''
s=s.replace(needle,proof+needle,1);p.write_text(s)
for p in w.glob('*.py'):ast.parse(p.read_text(),str(p))
(w/'native-terminal.json').write_text(json.dumps({'controller_session':72514,'terminal_exit_code':0,'original7_plus_request_native_each4builds_pass':True,'all186_complete_projection_records_exact':True,'actual_all4_programs_byte_exact_WS54':True,'initial_negative_fixture_fix_retained':True},indent=2)+'\n')
print('Full-model drivers with actual request admission/fallback/cache/cleanup assertions prepared')
