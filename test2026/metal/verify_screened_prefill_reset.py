"""Audit original gates, both recovery suites, full outputs and exact counters."""
from pathlib import Path
import ast
wrapper=Path(__file__).with_name('verify_screened_head_inference.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
for old,new in [('screened-head-checks-20261015','screened-prefill-reset-checks-20261015'),('screened-head-native-20261015','screened-prefill-reset-native-20261015'),('build-screened-head','build-screened-prefill-reset'),('build-global-half-alpha','build-screened-head'),('887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f','2b44d93ab1e1980845be89ea59cbaedbebcb66ad385ba24df3d52af2b6bacc23')]:source=source.replace(old,new)
source=source.replace('len(stages)==20','len(stages)==26')
source=source.replace("('METAL_PREFILL_STREAM','METAL_DIRTY_PREFIX_RESET','METAL_JOINED_GLOBAL_MLP','METAL_VISION_GLOBAL_ALPHA')","('METAL_JOINED_GLOBAL_MLP','METAL_VISION_GLOBAL_ALPHA')")
needle="assert {v['name'] for v in stages}==names";assert source.count(needle)==1
source=source.replace(needle,"names|={m+'-'+label+'-recovery' for m in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B') for label in ('prefix','screened')}\nassert all(v in meta['options'] for v in ('METAL_PREFILL_STREAM','METAL_DIRTY_PREFIX_RESET'))\n"+needle)
start=source.index("for phase in ('recovery','native'):");end=source.index('if confirmation_trial:',start)
source=source[:start]+'''dependencies=json.loads((trial/'dependency.json').read_text());assert dependencies['pid']==29706 and dependencies['completed_stages']==26 and dependencies['source_manifest']==proof['candidate']
for name,sha in json.loads((trial/'helper-identities.json').read_text()).items():assert digest(trial/'helper-snapshot'/name)==sha
recovered=0;prefix_recovered=0
for stage in stages:
    if not stage['name'].endswith('-recovery'):continue
    assert len(stage['accepted_cases'])==1
    text=(checks/stage['accepted_cases'][0]['evidence']/(stage['name']+'.log')).read_text()
    screened=stage['name'].endswith('-screened-recovery');prefix='screened_head_session_reference PASS ' if screened else 'prefix_reset_recovery PASS '
    rows=[json.loads(line.split(' ',2)[2]) for line in text.splitlines() if line.startswith(prefix)]
    assert len(rows)==2 and {r['batch'] for r in rows}=={1,2}
    keys=('actual_screened_head','generic_full_logits_preserved','float_head_fallback','nonfinite_poison_recovery','private_sessions_isolated','rebuild_and_close') if screened else ('partial_clear_verified','unknown_writer_full_fallback','nonfinite_request_poisoned','full_recovery_verified','private_sessions_isolated')
    assert all(all(r[k] for k in keys) for r in rows)
    if screened:recovered+=2
    else:prefix_recovered+=2
assert recovered==prefix_recovered==6
assert all(name in ctext for name in ('metal_prefill_stream_reference','metal_dirty_prefix_reset_reference'))
'''+source[end:]
source=source.replace("row['screened_head_enabled']==(policy=='candidate')","row['screened_head_enabled']").replace("all(v==(1 if policy=='candidate' else 0) for v in counts)",'all(v==1 for v in counts)')
source=source.replace("assert not row['vision_global_alpha_enabled'] and not row['prefill_stream_enabled'] and not row['dirty_prefix_reset_enabled']","assert not row['vision_global_alpha_enabled']\n            assert row['prefill_stream_enabled']==(policy=='candidate') and row['dirty_prefix_reset_enabled']==(policy=='candidate')")
tree=ast.parse(Path(__file__).with_name('compare_dirty_prefix_reset.py').read_text())
blocks=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and "if policy=='candidate':" in n.value and 'dirty_prefix_reset_bytes' in n.value];assert len(blocks)==1
block=blocks[0].replace('value[','row[')
block='\n'.join(line[4:] if line.startswith('    ') else line for line in block.splitlines())
needle="            row.update(build_policy=policy,repeat=repeat)";assert source.count(needle)==1
source=source.replace(needle,'''            configfile=root/'models'/model/'config.json';assert digest(configfile)==meta['assets'][model]['config.json']
            config=json.loads(configfile.read_text());config=config.get('text_config',config)
            submissions=((config['num_hidden_layers']-1)//4)*sum(chunk>1 for chunk in row['prefill_chunk_plan']) if policy=='candidate' else 0
            assert row['decode_partial_submissions']==0 and row['prefill_profile'].get('prefill_partial_submissions',0)==submissions
'''+block+'\n'+needle)
source=source.replace('trained_recovery_cases=6','trained_recovery_cases=6,prefix_recovery_cases=6,all_actual_submission_and_clear_counts_checked=True')
exec(compile(source,__file__,'exec'))
