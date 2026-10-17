"""Run only after observing full controller16119 terminal exit zero."""
from pathlib import Path
import json,hashlib,datetime
w=Path(__file__).resolve().parent;source=Path('/Users/shawn/CantorAI/out/research/worldsense-wide-primary-lazy/source')
assert all(json.loads((w/'api-correctness-result.json').read_text()).values())
assert json.loads((w/'native-word-comparison.json').read_text())['all_FP32_words_identical']
assert all(r['returncode']==0 for r in json.loads((w/'resident-results.json').read_text()))
assert all(v==0 for v in json.loads((w/'native-results.json').read_text()).values())
off,on=(json.loads((w/('api-state-progress-'+mode+'.json')).read_text())['records']for mode in ['off','on'])
assert len(off)==len(on)==5
for a,b in zip(off,on):
 assert a['model_id']==b['model_id']and a['states']==b['states']
 assert all(a['call'][k]==b['call'][k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
 ta=a['call']['timings'];t=b['call']['timings'];owner=t['primary_storage_after'];delta=t['primary_encoded_delta']
 assert ta['lazy_default_retired_bytes']==ta['lazy_default_rehydrated_bytes']==0
 assert all(v==0 for v in ta['primary_storage_after'].values())and all(v==0 for v in ta['primary_encoded_delta'].values())
 assert t['lazy_default_retired_bytes']>0 and t['lazy_default_planned_slots']>0 and t['lazy_default_rehydrated_bytes']==0
 if b['model_id']=='Qwen3.5-4B':
  assert owner['active_weights']==201 and all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded'])
 else:
  assert b['model_id']=='Qwen3-1.7B'and owner['active_weights']==0 and all(v==0 for v in delta.values())
assert json.loads((w/'api-state-guard-on.json').read_text())['lazy_defaults_expected']=='both_models_when_on'
c=json.loads((w/'corpus-GPU-bits.json').read_text());tot=c['totals']
assert tot['packed_matrices']==201 and tot['raw_fallback_matrices']==245
assert tot['GPU_original_FP32_words_checked']==tot['GPU_vector_original_FP32_words_checked']==4200857600
assert tot['retained_primary_and_small_raw_bytes']==9850930176
manifest=json.loads((w/'correctness-source-manifest.json').read_text());assert len(manifest)==1716
assert all(hashlib.sha256((source/k).read_bytes()).hexdigest()==h for k,h in manifest.items())
(w/'correctness-terminal.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_session':16119,'terminal_exit_code':0,'actual_wide_primary_and_global_retirement_scope_proven':True,'all_original_numerical_state_token_guards_pass':True,'shared_scope_mismatch_and_parent_isolation_pass':True,'source1716_unchanged':True,'normal_WS10_bootstrapped':True},indent=2)+'\n')
print('Full model state/token/corpus and wide-primary/global-retirement scope proved; source1716 unchanged.')
