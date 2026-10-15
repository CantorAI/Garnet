"""Native bounded-reset ABBA versus global alpha with unchanged image/text work."""
from pathlib import Path
source=Path(__file__).with_name('compare_vision_global_alpha.py').read_text().replace('build-vision-global-alpha','build-dirty-prefix-reset').replace('vision-global-alpha-checks-20261014','dirty-prefix-reset-checks-20261014')
source=source.replace("assert value['vision_global_alpha_enabled']==(policy=='candidate')","assert not value['vision_global_alpha_enabled']\n                assert value['dirty_prefix_reset_enabled']==(policy=='candidate')")
start=source.index('                expected_vision=');end=source.index("                assert not value['single_coder_head_enabled']",start)
source=source[:start]+'''                assert value['vision_global_alpha_dispatches']==0
                if policy=='candidate':
                    kinds=config.get('layer_types',['full_attention']*config['num_hidden_layers'])
                    full=kinds.count('full_attention');linear=kinds.count('linear_attention')
                    assert value['dirty_prefix_reset_states']==2*full
                    assert value['dirty_prefix_reset_fallback_states']==2*linear
                    extent=0 if value['request']==0 else len(value['prompt_token_ids'])+value['decode_steps_after_first']
                    dim=config.get('head_dim',config['hidden_size']//config['num_attention_heads'])
                    columns=2*full*config['num_key_value_heads']*extent*dim
                    if linear:
                        channels=2*config['linear_num_key_heads']*config['linear_key_head_dim']+config['linear_num_value_heads']*config['linear_value_head_dim']
                        columns+=linear*(channels*config['linear_conv_kernel_dim']+config['linear_num_value_heads']*config['linear_key_head_dim']*config['linear_value_head_dim'])
                    assert value['dirty_prefix_reset_bytes']==batch*columns*4
                else:assert value['dirty_prefix_reset_bytes']==value['dirty_prefix_reset_states']==value['dirty_prefix_reset_fallback_states']==0
'''+source[end:]
exec(compile(source,str(Path(__file__)),'exec'))
