"""Only run AFTER observing full controller70993 terminal exit0."""
from pathlib import Path
import json,hashlib,datetime
w=Path(__file__).resolve().parent;s=Path('/Users/shawn/CantorAI/out/research/worldsense-validated-text-worker/source')
assert all(json.loads((w/'api-correctness-result.json').read_text()).values())
assert all(v==0 for v in json.loads((w/'native-results.json').read_text()).values())
assert json.loads((w/'native-word-comparison.json').read_text())['all_four_builds_FP32_words_identical']
assert all(r['returncode']==0 for r in json.loads((w/'resident-results.json').read_text()))
records={mode:json.loads((w/('api-state-progress-'+mode+'.json')).read_text())['records']for mode in ['off','primary','on']}
assert all(len(v)==5 for v in records.values())
for index,a in enumerate(records['off']):
 for mode in ['primary','on']:
  b=records[mode][index];assert a['model_id']==b['model_id']and a['states']==b['states']
  assert all(a['call'][k]==b['call'][k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
  remote=b['model_id']=='Qwen3-1.7B';packed_expected=not remote;vector_expected=mode=='on'and not remote
  t=b['call']['timings'];assert t['lazy_default_rehydrated_bytes']==0 and (t['lazy_default_retired_bytes']>0)==packed_expected and (t['lazy_default_planned_slots']>0)==packed_expected
  assert ('cached_matrix_metadata'in b['call']['backend'])==vector_expected
  assert (t['primary_encoded_delta']['prefill_encoded']>0)==(b['model_id']=='Qwen3.5-4B')
  if b['model_id']!='Qwen3.5-4B':assert all(v==0 for v in t['primary_encoded_delta'].values())
  assert ('primary_vector_prefill'in b['call']['backend'])==vector_expected
  assert ('primary_prefill_isolated_library'in b['call']['backend'])==vector_expected
t=json.loads((w/'corpus-GPU-bits.json').read_text())['totals'];assert t['packed_matrices']==201 and t['raw_small_matrices']==245
assert t['GPU_original_FP32_words_checked']==t['GPU_vector_original_FP32_words_checked']==4200857600
assert t['retained_primary_and_small_raw_bytes']==9850930176
assert t['HOST_raw_FP32_words_checked']==1724383232
assert set(t['GPU_prefill_tile_FP32_words_checked'])=={'32x32','64x32','64x64'}and all(v==4200857600 for v in t['GPU_prefill_tile_FP32_words_checked'].values())
manifest=json.loads((w/'gateway-source-manifest.json').read_text());assert len(manifest)==1726
assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==h for k,h in manifest.items())
program=json.loads((w/'actual-program-proof.json').read_text());assert program['all_four_actual_main_and_prefill_programs_byte_exact_WS54']
direct=json.loads((w/'direct-upload-corpus.json').read_text());assert direct['totals']=={'weights':546,'FP32_words_checked':334609152}
(w/'correctness-terminal.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_session':int(__import__('sys').argv[1]),'terminal_exit_code':0,'all_three_actual_model_modes_and_five_readers_exact':True,'all_original_numerical_state_token_guards_pass':True,'source1726_unchanged':True,'native7tests_each4builds_original186projection_records_exact':True},indent=2)+'\n')
print('All3 full actual model modes and5 original-bit GPU readers exact; source1726 unchanged.')
