"""Preserve original mathematical/oracle checks; extend explicit sole-layout admission."""
from pathlib import Path
import json,hashlib,shutil,ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-sole-interleaved';old=b/'work/worldsense-isolated-primary';proof={}
for name in ['build_corpus.py','build_direct_corpus.py','make_resident_fixtures.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','full_correctness.py','verify_correctness_terminal.py','primary_corpus_test.cpp']:
 original=(old/name).read_text();text=original.replace('worldsense-isolated-primary','worldsense-sole-interleaved');ops=[]
 def change(a,z,count=1):
  global text
  assert text.count(a)==count,(name,a,text.count(a));text=text.replace(a,z);ops.append(dict(before=a,after=z,count=count))
 if name=='api_state_guard.py':
  needle=" stats=timings['primary_storage_after'];delta=timings['primary_encoded_delta']"
  add="""
 sole=timings['sole_storage_after'];sd=timings['sole_encoded_delta'];assert ('sole_interleaved_mlp'in call['backend'])==vector_expected
 if vector_expected and name=='Qwen3-1.7B':
  assert sole['active_weights']==84 and sole['allocated_bytes']==84*2048*6144*2 and sole['scratch_bytes']>0
  assert sd['decode_encoded']>0 and (sd['restore_encoded']>0)==(not timings['prefix_cache_hit'])
 elif vector_expected:
  assert all(v==0 for v in sole.values())and all(v==0 for v in sd.values())
 else:assert all(v==0 for v in sole.values())and all(v==0 for v in sd.values())
"""
  # A prefix hit may still prefill a suffix; never require zero restores merely from a hit.
  add=add.replace("assert sd['decode_encoded']>0 and (sd['restore_encoded']>0)==(not timings['prefix_cache_hit'])","assert sd['decode_encoded']>0 and sd['restore_encoded']>=0\n  if not timings['prefix_cache_hit']:assert sd['restore_encoded']>0")
  change(needle,needle+add)
 if name in ['full_correctness.py','verify_correctness_terminal.py']:
  if name=='full_correctness.py':
   change("assert c['HOST_raw_FP32_words_checked']==1724383232","assert c['HOST_raw_FP32_words_checked']==1724383232\n assert c['sole_interleaved_matrices']==84 and c['GPU_sole_original_FP32_words_checked']==1056964608")
   needle="assert json.loads((w/'native-word-comparison.json').read_text())['all_four_builds_FP32_words_identical']"
   change(needle,needle+"\nassert json.loads((w/'sole-word-comparison.json').read_text())['all_four_builds_FP32_words_identical']")
  else:
   change('len(manifest)==1718','len(manifest)==1720')
   change("assert t['HOST_raw_FP32_words_checked']==1724383232","assert t['HOST_raw_FP32_words_checked']==1724383232\nassert t['sole_interleaved_matrices']==84 and t['GPU_sole_original_FP32_words_checked']==1056964608")
  text=text.replace('source1718','source1720').replace('native7tests_all4builds','native8tests_all4builds').replace('native7tests_each4builds','native8tests_each4builds')
 if name=='primary_corpus_test.cpp':
  change('uint64_t rawWordsChecked=0,wordsChecked=0,','uint64_t soleWordsChecked=0,soleMatrices=0;uint64_t rawWordsChecked=0,wordsChecked=0,')
  change('uint64_t rawChecked=0,checked=0;','uint64_t rawChecked=0,checked=0,soleChecked=0;const auto soleBefore=MetalContext::SoleStats();')
  change('info[0].representation=="bf16"','info[0].representation==(((rows==6144&&inner==2048)||(rows==2048&&inner==6144))?"sole_bf16_interleaved":"bf16")')
  needle='rawChecked+=count;rawWordsChecked+=count;'
  extra='''if(info[0].representation=="sole_bf16_interleaved"){auto gpuActual=gpu.SoleInterleavedReadWords(w,start,count);need(gpuActual==actual,"Sole GPU words differ from every original HOST/checkpoint word");soleChecked+=count;soleWordsChecked+=count;}'''
  change(needle,needle+extra)
  needle='  rawBytes+=entry->dataSize;'
  extra='''  if(soleChecked){need(!expected&&model=="Qwen3-1.7B"&&soleChecked==rows*inner,"Unexpected sole learned matrix/domain");++soleMatrices;}
'''
  change(needle,extra+needle)
  needle='need(after.activeWeights==before.activeWeights&&after.allocatedBytes==before.allocatedBytes,"Corpus weight owner leak");'
  change(needle,needle+'auto soleAfter=MetalContext::SoleStats();need(soleAfter.activeWeights==soleBefore.activeWeights-uint64_t(soleChecked>0)&&soleAfter.allocatedBytes==soleBefore.allocatedBytes-uint64_t(soleChecked>0)*entry->dataSize,"Sole corpus owner leak");')
  # soleBefore is observed after upload; a successful owner release must subtract it.
  needle='{{"HOST_raw_FP32_words_checked",rawChecked},'
  change(needle,'{{"GPU_sole_original_FP32_words_checked",soleChecked},{"HOST_raw_FP32_words_checked",rawChecked},')
  needle='save(argv[2],report);\n  std::cout'
  change(needle,'report["totals"]["sole_interleaved_matrices"]=soleMatrices;report["totals"]["GPU_sole_original_FP32_words_checked"]=soleWordsChecked;'+needle)
  needle=' report["status"]="full_actual_two_original_and_three_prefill_tile_readers_GPU_bits_exact_not_inference_validated";'
  change(needle,' need(soleMatrices==84&&soleWordsChecked==1056964608ULL,"Missing learned sole-layout GPU bit coverage");\n'+needle)
 if name.endswith('.py'):ast.parse(text)
 (w/name).write_text(text)
 inverse=text.replace('source1720','source1718').replace('native8tests_all4builds','native7tests_all4builds').replace('native8tests_each4builds','native7tests_each4builds')
 for op in reversed(ops):
  assert inverse.count(op['after'])==op['count'],(name,'inverse count');inverse=inverse.replace(op['after'],op['before'])
 assert inverse.replace('worldsense-sole-interleaved','worldsense-isolated-primary')==original,name
 proof[name]=dict(seed_sha256=hashlib.sha256(original.encode()).hexdigest(),current_sha256=hashlib.sha256(text.encode()).hexdigest(),operations=ops,original_mathematical_word_state_token_golden_assertions_inverse_exact=True)
for name in ['direct_float_corpus_test.cpp','direct-upload-selection.json','model-assets.json']:shutil.copy2(old/name,w/name)
(w/'learned-guard-adaptation-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
print('All original learned guards retained; 84-matrix GPU logical readback and exact native route admission added')
