"""Preserve all original learned guards; add diagnostic scope and OFF coverage."""
from pathlib import Path
import ast,json,hashlib,shutil
opt=Path(__file__).resolve().parent;w=opt.parent/'worldsense-gpu-counter-profile';old=opt.parent/'worldsense-isolated-primary';proof={}
for name in ['build_corpus.py','build_direct_corpus.py','make_resident_fixtures.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','full_correctness.py','verify_correctness_terminal.py']:
 original=(old/name).read_text();text=original.replace('worldsense-isolated-primary','worldsense-gpu-counter-profile');changes=[]
 def change(a,b,count=1):
  global text
  assert text.count(a)==count,(name,a);text=text.replace(a,b);changes.append((a,b))
 if name=='api_state_guard.py':
  change("assert mode in ['off','primary','on'];expected=0 if mode=='off'else 1;vector_expected=mode=='on'","assert mode in ['off','primary','on','quiet'];expected=0 if mode=='off'else 1;vector_expected=mode in ['on','quiet']")
  change('import garnet,metal_serving','import garnet,metal_serving\nassert hasattr(garnet,"counter_trace_step")==hasattr(garnet,"counter_trace_request_id")==(mode!="quiet")')
  needle="assert call['status']=='ok'\n";extra=''' if mode!='quiet':
  assert isinstance(call['timings']['counter_trace_request_id'],int)and call['timings']['counter_trace_request_id']>0
  assert garnet.counter_trace_request_id()==0
 else:assert 'counter_trace_request_id'not in call['timings']
''';change(needle,needle+extra)
 if name=='full_correctness.py':
  change("=={'off','primary','on','fault'}","=={'off','primary','on','quiet','fault'}")
  change("['all_four_builds_FP32_words_identical']","['all_five_builds_FP32_words_identical']")
  change("program['actual_candidate_main_equals_raw_control']and program['actual_candidate_prefill_equals_minimal_preamble_plus_full_primary_and_vector_fragment']","program['all_five_actual_MAIN_and_prefill_sources_exact_seed']")
  change("for mode in ['off','primary','on']:","for mode in ['off','primary','on','quiet']:")
  change('with(w/log).open(\'w\')as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)',"with(w/log).open('w')as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True,env=dict(__import__('os').environ,GARNET_METAL_COUNTER_TRACE='1',GARNET_METAL_COUNTER_TRACE_FILE=str(w/(log+'.counters.jsonl'))))")
 if name=='verify_correctness_terminal.py':
  change("['all_four_builds_FP32_words_identical']","['all_five_builds_FP32_words_identical']")
  change("program['actual_candidate_main_equals_raw_control']and program['actual_candidate_prefill_equals_minimal_preamble_plus_full_primary_and_vector_fragment']","program['all_five_actual_MAIN_and_prefill_sources_exact_seed']")
  change("for mode in ['off','primary','on']}","for mode in ['off','primary','on','quiet']}")
  change("for mode in ['primary','on']:","for mode in ['primary','on','quiet']:")
  change("(mode=='on')","(mode in ['on','quiet'])",3)
  change('len(manifest)==1718','len(manifest)==1720')
 text=text.replace('source1718','source1720').replace('native7tests_all4builds','native8tests_all5builds').replace('native7tests_each4builds','native8tests_each5builds')
 ast.parse(text);(w/name).write_text(text)
 inverse=text.replace('source1720','source1718').replace('native8tests_all5builds','native7tests_all4builds').replace('native8tests_each5builds','native7tests_each4builds')
 for a,b in reversed(changes):
  assert b in inverse;inverse=inverse.replace(b,a)
 assert inverse.replace('worldsense-gpu-counter-profile','worldsense-isolated-primary')==original,name
 proof[name]=dict(seed_sha256=hashlib.sha256(original.encode()).hexdigest(),current_sha256=hashlib.sha256(text.encode()).hexdigest(),original_numeric_golden_assertions_inverse_exact=True)
for name in ['primary_corpus_test.cpp','direct_float_corpus_test.cpp','direct-upload-selection.json','model-assets.json']:shutil.copy2(old/name,w/name)
(w/'learned-guard-adaptation-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
print('WS62 all original learned/numeric/golden guards preserved; scope admission and defaultOFF quiet API coverage added')
