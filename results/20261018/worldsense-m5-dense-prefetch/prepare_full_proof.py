"""Prepare original full model guards; only additional raw-prefetch admission."""
from pathlib import Path
import ast,json,shutil
w=Path(__file__).resolve().parent;old=w.parent/'worldsense-isolated-primary'
names=['full_correctness.py','verify_correctness_terminal.py','build_corpus.py','build_direct_corpus.py','make_resident_fixtures.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py']
for name in names:
 t=(old/name).read_text().replace('worldsense-isolated-primary','worldsense-dense-prefetch')
 ast.parse(t);(w/name).write_text(t)
for name in ['primary_corpus_test.cpp','direct_float_corpus_test.cpp','direct-upload-selection.json','model-assets.json']:
 shutil.copy2(old/name,w/name)
for name in ['full_correctness.py','verify_correctness_terminal.py']:
 p=w/name;t=p.read_text()
 t=t.replace("program['actual_candidate_main_equals_raw_control']and program['actual_candidate_prefill_equals_minimal_preamble_plus_full_primary_and_vector_fragment']","program['OFF_main_exact54']and program['other_main_exact54_after_removing_one_exact_new_fragment']and program['all_four_prefill_programs_exact54']")
 t=t.replace('source1718','source1720').replace('==1718','==1720').replace('native7tests','native8tests').replace('186_projection_words_exact','186_original_and186_forced_raw_projection_words_exact').replace('186projection_records_exact','186original_and186forcedraw_projection_records_exact')
 if name=='verify_correctness_terminal.py':
  needle="  t=b['call']['timings'];"
  assert t.count(needle)==1;t=t.replace(needle,"  assert (b['call']['timings']['dense_prefetch_encoded_delta']>0)==(b['model_id']=='Qwen3-1.7B')\n"+needle)
  needle="program=json.loads((w/'actual-program-proof.json').read_text());"
  assert t.count(needle)==1;t=t.replace(needle,"assert json.loads((w/'dense-prefetch-word-comparison.json').read_text())['all_four_forced_raw_full_FP32_projection_files_identical']\n"+needle)
 ast.parse(t);p.write_text(t)
p=w/'api_state_guard.py';t=p.read_text()
needle="call=json.loads(garnet.infer_json(prompt,image,limit,name));"
assert t.count(needle)==1;t=t.replace(needle,needle+"(w/('raw-last-native-call-'+mode+'.json')).write_text(json.dumps(call,indent=2));")
needle=" assert ('primary_full_isolated_library'in call['backend'])==vector_expected"
assert t.count(needle)==1;t=t.replace(needle,needle+"\n assert ('dense_raw_decode_prefetch'in call['backend'])==bool(expected)\n assert (call['timings']['dense_prefetch_encoded_delta']>0)==(bool(expected)and name=='Qwen3-1.7B')")
ast.parse(t);p.write_text(t)
# Retain the initial numerical-only native proof before the complete identity-
# frozen proof repeats native tests. This is not a favorable performance retry.
folder=w/'initial-native-proof';folder.mkdir(exist_ok=False)
for p in w.glob('native-*'):
 if p.is_file():shutil.copy2(p,folder/p.name)
for name in ['actual-program-proof.json','dense-prefetch-word-comparison.json']:
 shutil.copy2(w/name,folder/name)
(w/'full-proof-preparation.json').write_text(json.dumps(dict(original_numerical_bounds_models_tokens_inputs_and_full_corpus_unchanged=True,additional_prefetch_capability_and_actual_text_positive_VLMzero_guards=True,initial_native_proof_retained=True,native_repeat_reason='Complete before/after source, driver, library, model and input identity closure; no performance measurement or prior failure retry',no_model_inference_dispatched=True),indent=2)+'\n')
print('Original full model/corpus/session/vision guards plus explicit raw-prefetch admission prepared')
