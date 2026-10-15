"""Six frozen native profiles with exact body dispatch and fallback counters."""
from pathlib import Path
source=Path(__file__).with_name('compare_vision_global_alpha.py').read_text().replace('build-vision-global-alpha','build-body-shape').replace('vision-global-alpha-checks-20261014/jobs/metadata.json','body-shape-checks-20261015/jobs/metadata.json')
source=source.replace("assert value['vision_global_alpha_enabled']==(policy=='candidate')", "assert not value['vision_global_alpha_enabled']\n                assert value['body_shape_enabled']==(policy=='candidate')")
source=source.replace("expected_vision=4*json.loads((folder/'config.json').read_text())['vision_config']['depth'] if vl and policy=='candidate' else 0",'expected_vision=0')
needle="                assert not value['single_coder_head_enabled']";assert source.count(needle)==1
source=source.replace(needle,'''                counts=value['body_shape_counts'];misses=value['body_shape_misses']
                assert len(counts)==len(misses)==value['decode_steps_after_first']+1 and not any(misses)
                kinds=config.get('layer_types',['full_attention']*config['num_hidden_layers'])
                expected_body=7*kinds.count('full_attention')+8*kinds.count('linear_attention')
                assert counts==[0]+[(expected_body if policy=='candidate' else 0)]*value['decode_steps_after_first']
                assert (0<value['body_shape_pipeline_count']<=128) if policy=='candidate' else value['body_shape_pipeline_count']==0
                assert not value['reused_zero_source_enabled'] and not value['deferred_state_zero_enabled']
                assert not value['fused_head_argmax_enabled'] and not value['fused_head_decode_only_enabled']
'''+needle)
exec(compile(source,__file__,'exec'))
