"""Independent full gates, logical snapshot/recovery and exact GPU-clear counters."""
from pathlib import Path
source=Path(__file__).with_name('verify_fused_head_argmax_inference.py').read_text()
for old,new in (
    ('fused-head-argmax-checks-20261015','deferred-state-zero-checks-20261015'),
    ('fused-head-argmax-native-20261015','deferred-state-zero-native-20261015'),
    ('fused-head-argmax-original-source-20261015','deferred-state-zero-original-source-20261015'),
    ('build-fused-head-argmax','build-deferred-state-zero'),
    ("dependencies['pid']==68935","dependencies['pid']==26187"),
    ("'METAL_FUSED_HEAD_ARGMAX' in meta['options']","'METAL_DEFERRED_STATE_ZERO' in meta['options']"),
    ('metal_fused_head_argmax_reference','metal_deferred_state_zero_reference'),
    ('fused_head_session_reference PASS','deferred_zero_session_reference PASS'),
    ("'actual_fused_head'","'actual_deferred_clear'"),("'float_head_fallback'","'generic_float_head'")):
    source=source.replace(old,new)
start=source.index("            assert row['fused_head_argmax_enabled']");end=source.index("            assert not row['vision_global_alpha_enabled']",start)
source=source[:start]+'''            assert not row['fused_head_argmax_enabled'] and not row['fused_head_decode_only_enabled']
            assert row['deferred_state_zero_enabled']==(policy=='candidate') and row['global_half_alpha_enabled']
            assert row['decode_deferred_zero_states']==row['decode_deferred_zero_bytes']==0
            if policy=='candidate':
                assert digest(root/'models'/model/'config.json')==meta['assets'][model]['config.json']
                config=json.loads((root/'models'/model/'config.json').read_text())
                config=config.get('text_config',config)
                kinds=config.get('layer_types',['full_attention']*config['num_hidden_layers'])
                full=kinds.count('full_attention');linear=kinds.count('linear_attention')
                dim=config.get('head_dim',config['hidden_size']//config['num_attention_heads'])
                columns=2*full*config['num_key_value_heads']*256*dim
                if linear:
                    channels=2*config['linear_num_key_heads']*config['linear_key_head_dim']+config['linear_num_value_heads']*config['linear_value_head_dim']
                    columns+=linear*(channels*config['linear_conv_kernel_dim']+config['linear_num_value_heads']*config['linear_key_head_dim']*config['linear_value_head_dim'])
                assert row['prefill_profile']['deferred_zero_states']==2*config['num_hidden_layers']
                assert row['prefill_profile']['deferred_zero_bytes']==batch*columns*4
            else:assert row['prefill_profile'].get('deferred_zero_states',0)==row['prefill_profile'].get('deferred_zero_bytes',0)==0
'''+source[end:]
exec(compile(source,__file__,'exec'))
