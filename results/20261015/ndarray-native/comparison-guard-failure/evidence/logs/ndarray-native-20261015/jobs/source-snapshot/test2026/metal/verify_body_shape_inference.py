"""Independent full-model audit of eager bounded body specialization."""
from pathlib import Path
source=Path(__file__).with_name('verify_fused_head_argmax_inference.py').read_text()
for old,new in (('fused-head-argmax-checks-20261015','body-shape-checks-20261015'),('fused-head-argmax-native-20261015','body-shape-native-20261015'),('fused-head-argmax-original-source-20261015','body-shape-original-source-20261015'),('build-fused-head-argmax','build-body-shape'),("dependencies['pid']==68935","dependencies['pid']==88191"),("'METAL_FUSED_HEAD_ARGMAX' in meta['options']","'METAL_BODY_SHAPE' in meta['options']"),('fused_head_session_reference PASS','body_shape_session_reference PASS'),("'actual_fused_head'","'actual_body_shape','eager_pipeline_cache','head_excluded','rebuild_and_close'"),("'float_head_fallback'","'generic_float_head'")):
    source=source.replace(old,new)
source=source.replace(" and 'metal_fused_head_argmax_reference' in ctext",'')
start=source.index("            assert row['fused_head_argmax_enabled']");end=source.index("            assert not row['vision_global_alpha_enabled']",start)
source=source[:start]+'''            assert row['body_shape_enabled']==(policy=='candidate') and row['global_half_alpha_enabled']
            assert not row['fused_head_argmax_enabled'] and not row['fused_head_decode_only_enabled']
            assert not row['reused_zero_source_enabled'] and not row['deferred_state_zero_enabled']
            counts=row['body_shape_counts'];misses=row['body_shape_misses']
            assert len(counts)==len(misses)==row['decode_steps_after_first']+1 and not any(misses)
            assert all(chunk*batch>2 for chunk in row['prefill_chunk_plan'])
            if policy=='candidate':
                assert digest(root/'models'/model/'config.json')==meta['assets'][model]['config.json']
                config=json.loads((root/'models'/model/'config.json').read_text());config=config.get('text_config',config)
                kinds=config.get('layer_types',['full_attention']*config['num_hidden_layers'])
                expected_body=7*kinds.count('full_attention')+8*kinds.count('linear_attention')
                assert counts==[0]+[expected_body]*row['decode_steps_after_first']
                assert 0<row['body_shape_pipeline_count']<=128
            else:
                assert not any(counts) and row['body_shape_pipeline_count']==0
'''+source[end:]
source=source.replace("'METAL_PREFILL_STREAM','METAL_DIRTY_PREFIX_RESET'", "'METAL_REUSED_ZERO_SOURCE','METAL_DEFERRED_STATE_ZERO','METAL_NATIVE_BFLOAT_BODY','METAL_FUSED_HEAD_ARGMAX','METAL_PREFILL_STREAM','METAL_DIRTY_PREFIX_RESET'")
exec(compile(source,__file__,'exec'))
