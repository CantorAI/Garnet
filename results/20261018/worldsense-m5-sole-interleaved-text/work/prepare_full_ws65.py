from pathlib import Path
import json,hashlib,ast,shutil
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-sole-interleaved-text';old=b/'work/worldsense-direct-interleaved-prefill';proof={}
for name in ['build_corpus.py','build_direct_corpus.py','make_resident_fixtures.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','full_correctness.py','verify_correctness_terminal.py','primary_corpus_test.cpp']:
 original=(old/name).read_text();text=original.replace('worldsense-direct-interleaved-prefill','worldsense-sole-interleaved-text');ops=[]
 def change(a,z,count=1):
  global text
  assert text.count(a)==count,(name,a,text.count(a));text=text.replace(a,z);ops.append(dict(before=a,after=z,count=count))
 if name in ['resident_controller.py','api_state_guard.py','full_correctness.py','verify_correctness_terminal.py']:change("['off','primary','on','restore']","['off','primary','on','restore','direct64']")
 if name=='resident_guard.py':change("endswith(('-on','-restore'))","endswith(('-on','-restore','-direct64'))")
 if name=='api_state_guard.py':
  change("vector_expected=mode in ['on','restore'];direct_expected=mode=='on'","vector_expected=mode in ['on','restore','direct64'];direct_expected=mode in ['on','direct64'];expanded_expected=mode=='on'")
  change("assert ('sole_direct_interleaved_prefill'in call['backend'])==direct_expected","assert ('sole_direct_interleaved_prefill'in call['backend'])==direct_expected\n assert ('sole_interleaved_text'in call['backend'])==expanded_expected")
  change("assert sole['active_weights']==84 and sole['allocated_bytes']==84*2048*6144*2","assert sole['active_weights']==(197 if expanded_expected else 84) and sole['allocated_bytes']==(3440902144 if expanded_expected else 2113929216)")
 if name=='full_correctness.py':
  change("{'off','primary','on','fault','restore'}","{'off','primary','on','fault','restore','direct64'}")
  change("['all_five_builds_FP32_words_identical']","['all_six_builds_FP32_words_identical']",2)
 if name=='verify_correctness_terminal.py':
  change("['all_five_builds_FP32_words_identical']","['all_six_builds_FP32_words_identical']")
  change("['primary','on','restore']","['primary','on','restore','direct64']")
  change("(mode in ['on','restore'])","(mode in ['on','restore','direct64'])",3)
  change('len(manifest)==1721','len(manifest)==1722')
 if name in ['full_correctness.py','verify_correctness_terminal.py']:
  change("['sole_interleaved_matrices']==84","['sole_interleaved_matrices']==197")
  change("['GPU_sole_original_FP32_words_checked']==1056964608","['GPU_sole_original_FP32_words_checked']==1720451072")
  change("{'32x32':1056964608,'64x64':1056964608,'64x32':1056964608}","{'32x32':1720451072,'64x64':1720451072,'64x32':1720451072}")
 if name=='primary_corpus_test.cpp':
  change('((rows==6144&&inner==2048)||(rows==2048&&inner==6144))','((rows==6144&&inner==2048)||(rows==2048&&inner==6144)||(inner==2048&&(rows==2048||rows==1024||rows==151936)))')
  change('soleMatrices==84&&soleWordsChecked==1056964608ULL','soleMatrices==197&&soleWordsChecked==1720451072ULL')
  change('tile.second==1056964608ULL','tile.second==1720451072ULL')
 if name in ['full_correctness.py','verify_correctness_terminal.py']:
  change('native8tests','native9tests')
 text=text.replace('source1721','source1722').replace('all5builds','all6builds').replace('each5builds','each6builds')
 if name.endswith('.py'):ast.parse(text)
 inverse=text.replace('source1722','source1721').replace('all6builds','all5builds').replace('each6builds','each5builds')
 for op in reversed(ops):assert inverse.count(op['after'])==op['count'];inverse=inverse.replace(op['after'],op['before'])
 assert inverse.replace('worldsense-sole-interleaved-text','worldsense-direct-interleaved-prefill')==original,name
 (w/name).write_text(text);proof[name]=dict(seed_sha256=hashlib.sha256(original.encode()).hexdigest(),current_sha256=hashlib.sha256(text.encode()).hexdigest(),operations=ops,original_math_oracles_bounds_inverse_exact=True)
for name in ['direct_float_corpus_test.cpp','direct-upload-selection.json','model-assets.json']:shutil.copy2(old/name,w/name)
(w/'learned-guard-adaptation-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print('WS65 expanded learned readers, five API modes and default-OFF direct64/restore compatibility prepared; original math inverse exact')
