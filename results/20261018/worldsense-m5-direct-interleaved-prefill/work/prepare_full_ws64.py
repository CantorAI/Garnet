"""Keep63 mathematical/oracle tests; add three full learned tile readers and RESTORE."""
from pathlib import Path
import json,hashlib,shutil,ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-direct-interleaved-prefill';old=b/'work/worldsense-sole-interleaved';proof={}
for name in ['build_corpus.py','build_direct_corpus.py','make_resident_fixtures.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','full_correctness.py','verify_correctness_terminal.py','primary_corpus_test.cpp']:
 original=(old/name).read_text();text=original.replace('worldsense-sole-interleaved','worldsense-direct-interleaved-prefill');ops=[]
 def change(a,z,count=1):
  global text
  assert text.count(a)==count,(name,a,text.count(a));text=text.replace(a,z);ops.append(dict(before=a,after=z,count=count))
 if name=='resident_controller.py':change("for mode in ['off','primary','on']:","for mode in ['off','primary','on','restore']:")
 if name=='resident_guard.py':change("sys.argv[2].endswith('-on')","sys.argv[2].endswith(('-on','-restore'))")
 if name=='api_state_guard.py':
  change("mode in ['off','primary','on'];expected=0 if mode=='off'else 1;vector_expected=mode=='on'","mode in ['off','primary','on','restore'];expected=0 if mode=='off'else 1;vector_expected=mode in ['on','restore'];direct_expected=mode=='on'")
  before=""" sole=timings['sole_storage_after'];sd=timings['sole_encoded_delta'];assert ('sole_interleaved_mlp'in call['backend'])==vector_expected
 if vector_expected and name=='Qwen3-1.7B':
  assert sole['active_weights']==84 and sole['allocated_bytes']==84*2048*6144*2 and sole['scratch_bytes']>0
  assert sd['decode_encoded']>0 and sd['restore_encoded']>=0
  if not timings['prefix_cache_hit']:assert sd['restore_encoded']>0
 elif vector_expected:
  assert all(v==0 for v in sole.values())and all(v==0 for v in sd.values())
 else:assert all(v==0 for v in sole.values())and all(v==0 for v in sd.values())"""
  after=""" sole=timings['sole_storage_after'];sd=timings['sole_encoded_delta'];assert ('sole_interleaved_mlp'in call['backend'])==vector_expected
 assert ('sole_direct_interleaved_prefill'in call['backend'])==direct_expected
 if vector_expected and name=='Qwen3-1.7B':
  assert sole['active_weights']==84 and sole['allocated_bytes']==84*2048*6144*2
  assert sd['decode_encoded']>0
  if direct_expected:
   assert sole['scratch_bytes']==0 and sole['restore_encoded']==0 and sd['restore_encoded']==0 and sd['prefill_encoded']>=0
   if not timings['prefix_cache_hit']:assert sd['prefill_encoded']>0
  else:
   assert sole['scratch_bytes']>0 and sd['restore_encoded']>=0 and sole['prefill_encoded']==0 and sd['prefill_encoded']==0
   if not timings['prefix_cache_hit']:assert sd['restore_encoded']>0
 elif vector_expected:
  assert all(v==0 for v in sole.values())and all(v==0 for v in sd.values())
 else:assert all(v==0 for v in sole.values())and all(v==0 for v in sd.values())"""
  change(before,after)
 if name=='full_correctness.py':
  change("=={'off','primary','on','fault'}","=={'off','primary','on','fault','restore'}")
  change("['all_four_builds_FP32_words_identical']","['all_five_builds_FP32_words_identical']",2)
  change("for mode in ['off','primary','on']:","for mode in ['off','primary','on','restore']:")
  needle="assert c['sole_interleaved_matrices']==84 and c['GPU_sole_original_FP32_words_checked']==1056964608"
  change(needle,needle+"\n assert c['GPU_sole_prefill_tile_FP32_words_checked']=={'32x32':1056964608,'64x64':1056964608,'64x32':1056964608}")
 if name=='verify_correctness_terminal.py':
  change("['all_four_builds_FP32_words_identical']","['all_five_builds_FP32_words_identical']")
  change("for mode in ['off','primary','on']}","for mode in ['off','primary','on','restore']}")
  change("for mode in ['primary','on']:","for mode in ['primary','on','restore']:")
  change("(mode=='on')","(mode in ['on','restore'])",3)
  change('len(manifest)==1720','len(manifest)==1721')
  needle="assert t['sole_interleaved_matrices']==84 and t['GPU_sole_original_FP32_words_checked']==1056964608"
  change(needle,needle+"\nassert t['GPU_sole_prefill_tile_FP32_words_checked']=={'32x32':1056964608,'64x64':1056964608,'64x32':1056964608}")
 if name=='primary_corpus_test.cpp':
  change('uint64_t soleWordsChecked=0,soleMatrices=0;','std::map<std::string,uint64_t> soleAllTiles{{"32x32",0},{"64x64",0},{"64x32",0}};uint64_t soleWordsChecked=0,soleMatrices=0;')
  change('uint64_t rawChecked=0,checked=0,soleChecked=0;','std::map<std::string,uint64_t> soleTiles{{"32x32",0},{"64x64",0},{"64x32",0}};uint64_t rawChecked=0,checked=0,soleChecked=0;')
  needle='  rawBytes+=entry->dataSize;'
  extra='''  if(soleChecked)for(auto geometry:std::vector<std::pair<size_t,size_t>>{{32,32},{64,64},{64,32}}){
   size_t n=geometry.first,k=geometry.second,chunk=(8*1024*1024/inner/n)*n;const auto key=std::to_string(n)+"x"+std::to_string(k);need(chunk>0,"Empty sole tile chunk");
   for(size_t first=0;first<rows;first+=chunk){size_t nr=std::min(chunk,rows-first);auto actual=gpu.SoleInterleavedReadPrefillTileWords(w,first,nr,n,k);need(actual.size()==nr*inner,"Sole tile readback length");
    for(size_t i=0;i<actual.size();++i){uint16_t word;std::memcpy(&word,data+2*(first*inner+i),2);need(actual[i]==uint32_t(word)<<16,"Sole direct-prefill learned tile word mismatch");}
    soleTiles[key]+=actual.size();soleAllTiles[key]+=actual.size();
   }need(soleTiles[key]==soleChecked,"Missing whole learned sole tile coverage");
  }
''';change(needle,extra+needle)
  needle='{{"GPU_sole_original_FP32_words_checked",soleChecked},'
  change(needle,'{{"GPU_sole_prefill_tile_FP32_words_checked",soleTiles},{"GPU_sole_original_FP32_words_checked",soleChecked},')
  needle='report["totals"]["GPU_sole_original_FP32_words_checked"]=soleWordsChecked;'
  change(needle,needle+'report["totals"]["GPU_sole_prefill_tile_FP32_words_checked"]=soleAllTiles;')
  needle='need(soleMatrices==84&&soleWordsChecked==1056964608ULL,"Missing learned sole-layout GPU bit coverage");'
  change(needle,needle+'for(const auto& tile:soleAllTiles)need(tile.second==1056964608ULL,"Missing whole learned sole direct-tile GPU words");')
 text=text.replace('source1720','source1721').replace('native8tests_all4builds','native8tests_all5builds').replace('native8tests_each4builds','native8tests_each5builds')
 if name.endswith('.py'):ast.parse(text)
 (w/name).write_text(text)
 inverse=text.replace('source1721','source1720').replace('native8tests_all5builds','native8tests_all4builds').replace('native8tests_each5builds','native8tests_each4builds')
 for op in reversed(ops):assert inverse.count(op['after'])==op['count'],(name,'inverse');inverse=inverse.replace(op['after'],op['before'])
 assert inverse.replace('worldsense-direct-interleaved-prefill','worldsense-sole-interleaved')==original,name
 proof[name]=dict(seed_sha256=hashlib.sha256(original.encode()).hexdigest(),current_sha256=hashlib.sha256(text.encode()).hexdigest(),operations=ops,original_mathematical_word_state_token_golden_assertions_inverse_exact=True)
for name in ['direct_float_corpus_test.cpp','direct-upload-selection.json','model-assets.json']:shutil.copy2(old/name,w/name)
(w/'learned-guard-adaptation-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print('Original63 math/golden bounds retained; three full learned tile readers and RESTORE compatibility added')
