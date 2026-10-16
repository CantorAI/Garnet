"""Frozen native ABBA; verify every state cleared before prefill and none during decode."""
from pathlib import Path
source=Path(__file__).with_name('compare_vision_global_alpha.py').read_text().replace('build-vision-global-alpha','build-deferred-state-zero').replace('vision-global-alpha-checks-20261014/jobs/metadata.json','deferred-state-zero-checks-20261015/jobs/metadata.json')
source=source.replace("assert value['vision_global_alpha_enabled']==(policy=='candidate')","assert not value['vision_global_alpha_enabled']\n                assert value['deferred_state_zero_enabled']==(policy=='candidate')")
start=source.index('                expected_vision=');end=source.index("                assert not value['single_coder_head_enabled']",start)
source=source[:start]+'''                assert value['vision_global_alpha_dispatches']==0
                assert value['decode_deferred_zero_bytes']==value['decode_deferred_zero_states']==0
                if policy=='candidate':
                    kinds=config.get('layer_types',['full_attention']*config['num_hidden_layers'])
                    full=kinds.count('full_attention');linear=kinds.count('linear_attention')
                    dim=config.get('head_dim',config['hidden_size']//config['num_attention_heads'])
                    columns=2*full*config['num_key_value_heads']*256*dim
                    if linear:
                        channels=2*config['linear_num_key_heads']*config['linear_key_head_dim']+config['linear_num_value_heads']*config['linear_value_head_dim']
                        columns+=linear*(channels*config['linear_conv_kernel_dim']+config['linear_num_value_heads']*config['linear_key_head_dim']*config['linear_value_head_dim'])
                    assert value['prefill_profile']['deferred_zero_states']==2*config['num_hidden_layers']
                    assert value['prefill_profile']['deferred_zero_bytes']==batch*columns*4
                else:
                    assert value['prefill_profile'].get('deferred_zero_states',0)==value['prefill_profile'].get('deferred_zero_bytes',0)==0
                assert not value['fused_head_argmax_enabled'] and not value['fused_head_decode_only_enabled']
'''+source[end:]
exec(compile(source,__file__,'exec'))
