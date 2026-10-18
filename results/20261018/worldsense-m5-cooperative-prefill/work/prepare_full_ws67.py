from pathlib import Path
import json,hashlib,shutil,ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-cooperative-prefill';old=b/'work/worldsense-sole-dual-decode';proof={}
for name in ['build_corpus.py','build_direct_corpus.py','build_dual_pair_corpus.py','make_resident_fixtures.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','full_correctness.py','verify_correctness_terminal.py','primary_corpus_test.cpp','dual_pair_corpus_test.cpp']:
 original=(old/name).read_text();text=original.replace('worldsense-sole-dual-decode','worldsense-cooperative-prefill');ops=[]
 def change(x,z,count=1):
  global text
  assert text.count(x)==count,(name,x,text.count(x));text=text.replace(x,z);ops.append(dict(before=x,after=z,count=count))
 if name in ['resident_controller.py','api_state_guard.py','full_correctness.py','verify_correctness_terminal.py']:change("['off','primary','on','restore','direct64']","['off','primary','on','restore','direct64','dual66']")
 if name=='resident_guard.py':change("endswith(('-on','-restore','-direct64'))","endswith(('-on','-restore','-direct64','-dual66'))")
 if name=='api_state_guard.py':
  change("vector_expected=mode in ['on','restore','direct64'];direct_expected=mode in ['on','direct64'];dual_expected=mode=='on'","vector_expected=mode in ['on','restore','direct64','dual66'];direct_expected=mode in ['on','direct64','dual66'];dual_expected=mode in ['on','dual66'];cooperative_expected=mode=='on'")
  change(" assert ('sole_dual_projection_decode'in call['backend'])==dual_expected"," assert ('sole_dual_projection_decode'in call['backend'])==dual_expected\n assert ('sole_cooperative_prefill'in call['backend'])==cooperative_expected")
 if name=='full_correctness.py':
  change("{'off','primary','on','fault','restore','direct64'}","{'off','primary','on','fault','restore','direct64','dual66'}")
  change("['all_six_builds_FP32_words_identical']","['all_seven_builds_FP32_words_identical']",2)
  change("['all_six_builds_complete_FP32_words_identical']","['all_seven_builds_complete_FP32_words_identical']")
 if name=='verify_correctness_terminal.py':
  change("['all_six_builds_FP32_words_identical']","['all_seven_builds_FP32_words_identical']")
  change("['primary','on','restore','direct64']","['primary','on','restore','direct64','dual66']")
  change("(mode in ['on','restore','direct64'])","(mode in ['on','restore','direct64','dual66'])",3)
  change('len(manifest)==1723','len(manifest)==1724')
 text=text.replace('source1723','source1724').replace('all6builds','all7builds').replace('each6builds','each7builds')
 if name.endswith('.py'):ast.parse(text)
 inverse=text.replace('source1724','source1723').replace('all7builds','all6builds').replace('each7builds','each6builds')
 for op in reversed(ops):assert inverse.count(op['after'])==op['count'];inverse=inverse.replace(op['after'],op['before'])
 assert inverse.replace('worldsense-cooperative-prefill','worldsense-sole-dual-decode')==original,name
 (w/name).write_text(text);proof[name]=dict(seed_sha256=hashlib.sha256(original.encode()).hexdigest(),current_sha256=hashlib.sha256(text.encode()).hexdigest(),operations=ops,original_mathematical_bounds_oracles_inverse_exact=True)
for name in ['direct_float_corpus_test.cpp','direct-upload-selection.json','model-assets.json']:shutil.copy2(old/name,w/name)
(w/'learned-guard-adaptation-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print('WS67 all original guards and GPU readers retained; seven native/sixAPI/DUAL66 compatibility modes added, math inverse exact')
