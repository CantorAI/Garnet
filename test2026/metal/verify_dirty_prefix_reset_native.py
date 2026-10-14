"""Audit full numerical/output scope and exact observed cache clearing bytes."""
from pathlib import Path
ns={'__file__':__file__};wrapper=Path(__file__).with_name('verify_vision_global_alpha_native.py').read_text()
exec(compile(wrapper[:wrapper.index("if '--confirm'")],__file__,'exec'),ns);source=ns['source']
source=source.replace('vision-global-alpha-staging','dirty-prefix-reset-staging').replace('vision-global-alpha-original-source-20261014','dirty-prefix-reset-original-source-20261014').replace('vision-global-alpha-checks-20261014','dirty-prefix-reset-checks-20261014').replace('vision-global-alpha-native-20261014','dirty-prefix-reset-native-20261014')
source=source.replace('METAL_VISION_GLOBAL_ALPHA','METAL_DIRTY_PREFIX_RESET')
source=source.replace('Vision global alpha tiny/full rows, FP64, decoder limits, fallback and abort PASS','Dirty prefix reset complete CPU/GPU positive zero, bounds, uninitialized and abort PASS')
source=source.replace("assert r['vision_global_alpha_enabled']==candidate", "assert not r['vision_global_alpha_enabled'] and r['dirty_prefix_reset_enabled']==candidate")
needle="assert r['vision_global_alpha_dispatches']==(48 if candidate and p['model']=='Qwen3.5-0.8B' else 0)"
assert source.count(needle)==1
source=source.replace(needle,'''assert r['vision_global_alpha_dispatches']==0
  configfile=base/'models'/p['model']/'config.json';assert digest(configfile)==metadata['assets'][p['model']]['config.json']
  config=json.loads(configfile.read_text());config=config.get('text_config',config)
  kinds=config.get('layer_types',['full_attention']*config['num_hidden_layers']);full=kinds.count('full_attention');linear=kinds.count('linear_attention')
  if candidate:
   assert r['dirty_prefix_reset_states']==2*full and r['dirty_prefix_reset_fallback_states']==2*linear
   extent=0 if r['request']==0 else len(r['prompt_token_ids'])+r['decode_steps_after_first']
   columns=2*full*config['num_key_value_heads']*extent*config.get('head_dim',config['hidden_size']//config['num_attention_heads'])
   if linear:
    channels=2*config['linear_num_key_heads']*config['linear_key_head_dim']+config['linear_num_value_heads']*config['linear_value_head_dim']
    columns+=linear*(channels*config['linear_conv_kernel_dim']+config['linear_num_value_heads']*config['linear_key_head_dim']*config['linear_value_head_dim'])
   assert r['dirty_prefix_reset_bytes']==p['batch']*columns*4
  else:assert r['dirty_prefix_reset_bytes']==r['dirty_prefix_reset_states']==r['dirty_prefix_reset_fallback_states']==0''')
needle="meta=json.loads((checks/'metadata.json').read_text())"
source=source.replace(needle,'''gates=base/'logs/dirty-prefix-reset-native-20261014/jobs-session-gates'
gate_records=json.loads((gates/'statuses.json').read_text());assert len(gate_records)==3 and all(v['exit_code']==0 for v in gate_records)
for gate in gate_records:
 text=(gates/(gate['name']+'.log')).read_text();results=[json.loads(line.split(' ',2)[2]) for line in text.splitlines() if line.startswith('prefix_reset_recovery PASS ')]
 assert len(results)==2 and {v['batch'] for v in results}=={1,2}
 assert all(all(v[k] for k in ('partial_clear_verified','unknown_writer_full_fallback','nonfinite_request_poisoned','full_recovery_verified','private_sessions_isolated')) for v in results)
 assert json.loads((gates/(gate['name']+'-monitor.json')).read_text())
meta=json.loads((checks/'metadata.json').read_text())''')
source=source.replace('all_actual_vision_global_alpha_counts_checked','all_actual_dirty_prefix_reset_bytes_checked').replace('Vision block global alpha versus global alpha; no fresh vLLM comparison','Dirty prefix reset versus global alpha; no fresh vLLM comparison')
source=source.replace("actual_vision_global_alpha_counts={policy:[r['vision_global_alpha_dispatches']", "actual_dirty_prefix_reset_bytes={policy:[r['dirty_prefix_reset_bytes']")
exec(compile(source,__file__,'exec'))
