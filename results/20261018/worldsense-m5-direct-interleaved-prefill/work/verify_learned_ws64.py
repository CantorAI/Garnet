"""Independent frozen identities, original oracles and complete direct-tile coverage."""
from pathlib import Path
import json,hashlib,subprocess,sys,datetime
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-direct-interleaved-prefill';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-direct-interleaved-prefill/source'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
frozen=json.loads((w/'learned-identities-before.json').read_text());assert len(frozen['source'])==1721 and all(sha(s/k)==v for k,v in frozen['source'].items())and all(sha(Path(k))==v for k,v in frozen['external'].items())
subprocess.run([sys.executable,str(w/'verify_correctness_terminal.py'),sys.argv[1]],check=True)
corpus=json.loads((w/'corpus-GPU-bits.json').read_text());sole=[row for row in corpus['matrices']if row['GPU_sole_original_FP32_words_checked']]
assert len(sole)==84 and all(row['model']=='Qwen3-1.7B'and(row['rows'],row['inner'])in[(6144,2048),(2048,6144)]and row['GPU_sole_original_FP32_words_checked']==row['rows']*row['inner']for row in sole)
assert sum(row['GPU_sole_original_FP32_words_checked']for row in sole)==1056964608
assert all(row['GPU_sole_prefill_tile_FP32_words_checked']=={key:row['rows']*row['inner']for key in ['32x32','64x64','64x32']}for row in sole)
assert corpus['totals']['GPU_sole_prefill_tile_FP32_words_checked']=={'32x32':1056964608,'64x64':1056964608,'64x32':1056964608}
refs={row['prompt_index']:row for row in json.loads((w.parent/'worldsense-prefix-cache/prefix-model-guard.json').read_text())['records']};vlm=json.loads((w.parent/'worldsense-prefix-cache/state-guard.json').read_text())['records'][1];routes={}
for mode in ['off','primary','on','restore']:
 rows=json.loads((w/('api-state-progress-'+mode+'.json')).read_text())['records'];assert len(rows)==5
 for i,row in enumerate(rows):
  original=vlm if i<2 else refs[[0,1,0][i-2]];c=row['call'];assert row['states']==original['states']and c['generated_token_ids']==original['generated_token_ids'];t=c['timings'];stats=t['sole_storage_after'];delta=t['sole_encoded_delta']
  assert ('sole_direct_interleaved_prefill'in c['backend'])==(mode=='on')
  if mode in ['on','restore']and i>=2:
   assert stats['active_weights']==84 and stats['allocated_bytes']==2113929216 and delta['decode_encoded']>0
   if mode=='on':
    assert stats['scratch_bytes']==0 and stats['restore_encoded']==0 and delta['restore_encoded']==0 and delta['prefill_encoded']>=0
    if not t['prefix_cache_hit']:assert delta['prefill_encoded']>0
   else:
    assert stats['scratch_bytes']>0 and stats['prefill_encoded']==0 and delta['prefill_encoded']==0
    if not t['prefix_cache_hit']:assert delta['restore_encoded']>0
  else:assert all(v==0 for v in stats.values())and all(v==0 for v in delta.values())
 routes[mode]=[dict(model=row['model_id'],after=row['call']['timings']['sole_storage_after'],delta=row['call']['timings']['sole_encoded_delta'])for row in rows]
for mode in ['off','on']:assert json.loads((w/('vision-grid-'+mode+'.json')).read_text())==json.loads((w.parent/'worldsense-request-residency/vision-grid-off.json').read_text())
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_terminal_session=int(sys.argv[1]),observed_exit_code=0,source_runtime_models_binaries_math_guards_and_outer_verifier_frozen_before_launch_unchanged=True,all_original_model_word_state_token_vision_gates_pass=True,all84_actual_text_MLP_weights_GPU_original_FP32_words=1056964608,three_actual_direct_prefill_tile_readers_each_FP32_words=1056964608,four_actual_API_modes_original_states_and_tokens_exact=True,defaultOFF_restore_compatibility_pass=True,actual_native_routes_and_scratch=routes,no_speed_or_promotion_claim=True)
(w/'learned-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('WS64 full original learned proof, scalar+three tile readers, directzero-scratch and defaultOFFrestore compatibility PASS')
