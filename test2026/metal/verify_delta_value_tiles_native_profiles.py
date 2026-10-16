"""All72 original complete-output/clock gates, native width64 routes and regenerated comparators."""
from pathlib import Path
import ast
p=Path(__file__).with_name('verify_vector_vision_native_profiles.py');source=p.read_text()
source=source.replace('vector-vision-native','delta-value-tile64-native').replace('verify_vector_vision_native_checks.py','verify_delta_value_tiles_native_checks.py').replace('compare_vector_vision_native.py','compare_delta_value_tiles_native.py')
old="source=source.replace('build-mlp-prefill-graph-guardfix','build-vector-vision-width4')";assert source.count(old)==1
source=source.replace(old,"source=source.replace('build-mlp-prefill-graph-guardfix','build-delta-value-tile64-prefill')\nsource=source.replace('build-coalesced-bf16-body','build-vector-vision-width4')")
source=source.replace("assert row['vector_vision_context_enabled']==(policy=='candidate')","assert row['vector_vision_context_enabled']").replace("vision_depth if model=='Qwen3.5-0.8B' and policy=='candidate' else 0","vision_depth if model=='Qwen3.5-0.8B' else 0")
needle="            assert row['vision_vector_dispatches']==expected_vector";assert source.count(needle)==1
source=source.replace(needle,needle+"""
            assert row['delta_value_tile64_enabled']==(policy=='candidate')
            routing_config=json.loads((root/'models'/model/'config.json').read_text());routing_config=routing_config.get('text_config',routing_config)
            linear_depth=sum(kind=='linear_attention' for kind in routing_config.get('layer_types',[])) if model=='Qwen3.5-0.8B' and policy=='candidate' else 0
            if linear_depth:assert linear_depth==18 and routing_config['linear_num_value_heads']==16 and routing_config['linear_key_head_dim']==routing_config['linear_value_head_dim']==128
            expected_prefill=linear_depth*sum(chunk==80 for chunk in row['prefill_chunk_plan'])
            assert row['delta_value_tile64_counts']==[expected_prefill]+[0]*row['decode_steps_after_first']
            assert row['prefill_profile'].get('delta_value_tile64_dispatches',0)==expected_prefill""")
source=source.replace("assert (native/(folder.name+'.executed.py')).read_bytes()==expanded","assert (native/folder.with_suffix('.executed.py').name).read_bytes()==expanded")
needle="result['regenerated_native_comparators']=6";assert source.count(needle)==1;source=source.replace(needle,needle+"\nresult['actual_delta_value_tile64_routes']=True\nresult['complete_delta_value_tile64_route_requests']=72")
source=source.replace("scope='Native complete-model ordered vector vision only; first-after-init, warm, initialization and RSS distinct; caches unflushed; no fresh vLLM ratio'","scope='Native complete-model width64 chunk80 recurrence versus accepted d52 only; init/first/warm/RSS separate; no fresh vLLM ratio'")
needle="    assert meta['wide_batch_prefill']==dict(candidate_vl_b2_plan=[80],control_vl_b2_plan=[80],logical_inputs_unchanged=True,physical_total_slots=80,no_fresh_vllm_ratio=True)";assert source.count(needle)==1
source=source.replace(needle,needle+"\n    assert meta['delta_value_tile64']==dict(batch=[1,2],chunk=80,heads=16,key_dim=128,value_dim=128,tile=64,threads_per_group=128,original_values=True,fp32_arithmetic=True,original_arithmetic=True,extra_weights=False)")
ast.parse(source);exec(compile(source,__file__,'exec'))
