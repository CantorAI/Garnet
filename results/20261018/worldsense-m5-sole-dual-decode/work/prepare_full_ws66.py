from pathlib import Path
import json,hashlib,shutil,ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-sole-dual-decode';old=b/'work/worldsense-direct-interleaved-prefill';proof={}
for name in ['build_corpus.py','build_direct_corpus.py','make_resident_fixtures.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','full_correctness.py','verify_correctness_terminal.py','primary_corpus_test.cpp']:
 original=(old/name).read_text();text=original.replace('worldsense-direct-interleaved-prefill','worldsense-sole-dual-decode');ops=[]
 def change(a,z,count=1):
  global text
  assert text.count(a)==count,(name,a,text.count(a));text=text.replace(a,z);ops.append(dict(before=a,after=z,count=count))
 if name in ['resident_controller.py','api_state_guard.py','full_correctness.py','verify_correctness_terminal.py']:change("['off','primary','on','restore']","['off','primary','on','restore','direct64']")
 if name=='resident_controller.py':
  # Complete actual checkpoint layer fixtures, including B1/B2, plus an explicit original-op diagnostics run.
  extra='''
import os
for fixture in ['fixture-1-8','fixture-2-64']:
 env=dict(os.environ,GARNET_METAL_PROFILE_OPS='1');log=w/('resident-profile-on-'+fixture+'.log')
 with log.open('w')as stream:
  p=subprocess.run([n,'-B',str(w/'resident_guard.py'),str(w/fixture/'fixture.json'),'/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-sole-dual-decode-on',source,'profile'],env=env,stdout=stream,stderr=subprocess.STDOUT)
 results.append(dict(mode='profile-on',fixture=fixture,returncode=p.returncode,log=log.name));(w/'resident-results.json').write_text(json.dumps(results,indent=2))
 if p.returncode:raise RuntimeError(str(results[-1]))
'''
  change("print('OFF/ON ordinary-resident alternation, late second session, invalid inputs, closed/rebuilt graphs PASS',flush=True)",extra+"print('OFF/ON ordinary-resident alternation, late second session, invalid inputs, closed/rebuilt graphs PASS',flush=True)")
 if name=='resident_guard.py':
  change("endswith(('-on','-restore'))","endswith(('-on','-restore','-direct64'))")
  needle='def snapshot(s):'
  helper='''dual_expected=('sole_dual_projection_decode'in garnet.backend_info())and not(len(sys.argv)>4 and sys.argv[4]=='profile')
route_checks=[]
def checked_run(fn):
 before=json.loads(garnet.sole_dual_projection_stats_json());result=fn();after=json.loads(garnet.sole_dual_projection_stats_json())
 delta={key:after[key]-before[key]for key in ['encoded_pairs','represented_projections']}
 assert delta['encoded_pairs']==int(dual_expected)and delta['represented_projections']==2*int(dual_expected),'Actual generic/resident layer dual route mismatch'
 route_checks.append(delta);return result
''';change(needle,helper+needle)
  change('graph.run(control)','checked_run(lambda:graph.run(control))',3)
  change('session.run([graph],values[0],values[3],values[4],values[5],-1)','checked_run(lambda:session.run([graph],values[0],values[3],values[4],values[5],-1))',4)
  change('other.run([graph],values[0],values[3],values[4],values[5],-1)','checked_run(lambda:other.run([graph],values[0],values[3],values[4],values[5],-1))')
  needle="    print('private_kv_reference PASS'";assert text.count(needle)==1
  change(needle,"    assert len(route_checks)==34\n    print('dual_projection_generic_resident_routes',json.dumps(dict(enabled=dual_expected,checks=route_checks)),flush=True)\n"+needle)
 if name=='api_state_guard.py':
  change("vector_expected=mode in ['on','restore'];direct_expected=mode=='on'","vector_expected=mode in ['on','restore','direct64'];direct_expected=mode in ['on','direct64'];dual_expected=mode=='on'")
  needle=" assert ('sole_direct_interleaved_prefill'in call['backend'])==direct_expected"
  added='''
 assert ('sole_dual_projection_decode'in call['backend'])==dual_expected
 dual=timings['dual_projection_after'];db=timings['dual_projection_before'];dd=timings['dual_projection_delta'];assert set(dual)==set(db)==set(dd)=={'encoded_pairs','represented_projections'}
 assert dd=={key:dual[key]-db[key]for key in dual}and dual['represented_projections']==2*dual['encoded_pairs']and dd['represented_projections']==2*dd['encoded_pairs']
 if dual_expected and name=='Qwen3-1.7B':assert dd['encoded_pairs']>=28*(call['generated_tokens']-1)>0
 else:assert all(v==0 for v in dual.values())and all(v==0 for v in dd.values())
''';change(needle,needle+added)
 if name=='full_correctness.py':
  change("{'off','primary','on','fault','restore'}","{'off','primary','on','fault','restore','direct64'}")
  change("['all_five_builds_FP32_words_identical']","['all_six_builds_FP32_words_identical']",2)
  needle="assert json.loads((w/'sole-word-comparison.json').read_text())['all_six_builds_FP32_words_identical']";change(needle,needle+"\nassert json.loads((w/'dual-word-comparison.json').read_text())['all_six_builds_complete_FP32_words_identical']")
  needle=" print('DIRECT_NATIVE_LEARNED_RAW_WEIGHTS_FP32_BITS_EXACT',flush=True)"
  extra="\n run([str(r/'out/build/garnet-metal-worldsense-sole-dual-decode-on/dual_pair_corpus_test'),str(w.parent/'worldsense-primary-blocks/matrix-selection.json'),str(w/'dual-pair-corpus.json')],'dual-pair-corpus.log')\n assert json.loads((w/'dual-pair-corpus.json').read_text())['totals']=={'pairs':28,'matrices':56,'original_FP32_words_checked':704643072}\n print('ALL56LEARNED_GATE_UP_WEIGHTS_DUAL_VECTOR_READER_WORDS_EXACT',flush=True)";change(needle,needle+extra)
 if name=='verify_correctness_terminal.py':
  change("['all_five_builds_FP32_words_identical']","['all_six_builds_FP32_words_identical']")
  change("['primary','on','restore']","['primary','on','restore','direct64']")
  change("(mode in ['on','restore'])","(mode in ['on','restore','direct64'])",3)
  change('len(manifest)==1721','len(manifest)==1723')
 if name in ['full_correctness.py','verify_correctness_terminal.py']:change('native8tests','native9tests')
 text=text.replace('source1721','source1723').replace('all5builds','all6builds').replace('each5builds','each6builds')
 if name.endswith('.py'):ast.parse(text)
 inverse=text.replace('source1723','source1721').replace('all6builds','all5builds').replace('each6builds','each5builds')
 for op in reversed(ops):assert inverse.count(op['after'])==op['count'];inverse=inverse.replace(op['after'],op['before'])
 assert inverse.replace('worldsense-sole-dual-decode','worldsense-direct-interleaved-prefill')==original,name
 (w/name).write_text(text);proof[name]=dict(seed_sha256=hashlib.sha256(original.encode()).hexdigest(),current_sha256=hashlib.sha256(text.encode()).hexdigest(),operations=ops,original_mathematical_bounds_oracles_inverse_exact=True)
for name in ['direct_float_corpus_test.cpp','direct-upload-selection.json','model-assets.json']:shutil.copy2(old/name,w/name)
(w/'learned-guard-adaptation-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print('WS66 original full learned guards preserved; six native roles/fiveAPI modes, actual resident pair routes and profile fallback added')
