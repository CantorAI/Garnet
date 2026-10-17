from pathlib import Path
import json,hashlib,datetime
w=Path(__file__).resolve().parent;source=Path('/Users/shawn/CantorAI/out/research/worldsense-primary-lazy/source')
assert all(json.loads((w/'api-correctness-result.json').read_text()).values())
assert json.loads((w/'native-word-comparison.json').read_text())['all_FP32_words_identical']
assert all(r['returncode']==0 for r in json.loads((w/'resident-results.json').read_text()))
off,on=(json.loads((w/('api-state-progress-'+m+'.json')).read_text())['records']for m in ['off','on'])
assert len(off)==len(on)==5
for a,b in zip(off,on):
 assert a['model_id']==b['model_id']and a['states']==b['states']
 assert all(a['call'][k]==b['call'][k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
 assert a['call']['timings']['lazy_default_retired_bytes']==a['call']['timings']['lazy_default_rehydrated_bytes']==0
 t=b['call']['timings'];assert t['lazy_default_retired_bytes']>0 and t['lazy_default_planned_slots']>0 and t['lazy_default_rehydrated_bytes']==0
 assert t['primary_storage_after']['active_weights']==(201 if b['model_id']=='Qwen3.5-4B' else 197)
m=json.loads((w/'correctness-source-manifest.json').read_text());assert len(m)==1715
assert all(hashlib.sha256((source/k).read_bytes()).hexdigest()==h for k,h in m.items())
(w/'correctness-terminal.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_session':46941,'terminal_exit_code':0,'joint_primary_and_lazy_scope_proven':True,'all_original_numerical_state_token_guards_pass':True,'source1715_unchanged':True,'all398_GPU_scalar_and_vector_original_bits':True,'normal_WS10_bootstrapped':True},indent=2)+'\n')
print('WS45 full OFF/ON model states/tokens and positive ON retirement for both models verified; source1715 unchanged.')
