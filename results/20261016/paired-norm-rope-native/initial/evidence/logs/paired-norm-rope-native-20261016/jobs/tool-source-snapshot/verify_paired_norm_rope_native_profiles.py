"""Complete72-output native paired norm-RoPE audit, exact routes and regenerated six comparator invocations."""
from pathlib import Path
import ast
p=Path(__file__).with_name('verify_vector_vision_native_profiles.py');source=p.read_text()
source=source.replace('vector-vision-native','paired-norm-rope-native').replace('verify_vector_vision_native_checks.py','verify_paired_norm_rope_native_checks.py').replace('compare_vector_vision_native.py','compare_paired_norm_rope_native.py')
old="source=source.replace('build-mlp-prefill-graph-guardfix','build-vector-vision-width4')";assert source.count(old)==1
source=source.replace(old,"source=source.replace('build-mlp-prefill-graph-guardfix','build-paired-norm-rope-decode')\nsource=source.replace('build-coalesced-bf16-body','build-vector-vision-width4')")
source=source.replace("assert row['vector_vision_context_enabled']==(policy=='candidate')","assert row['vector_vision_context_enabled']").replace("vision_depth if model=='Qwen3.5-0.8B' and policy=='candidate' else 0","vision_depth if model=='Qwen3.5-0.8B' else 0")
needle="            assert row['vision_vector_dispatches']==expected_vector";assert source.count(needle)==1
source=source.replace(needle,needle+"""
            assert row['paired_norm_rope_enabled']==(policy=='candidate')
            routing_config=json.loads((root/'models'/model/'config.json').read_text());routing_config=routing_config.get('text_config',routing_config)
            selected_pair=model in ['Qwen2.5-Coder-0.5B-Instruct','Qwen3-0.6B'] and policy=='candidate'
            pair_depth=routing_config['num_hidden_layers'] if selected_pair else 0
            expected_prefill=pair_depth*sum(batch*c in (1,2) for c in row['prefill_chunk_plan'])
            assert row['paired_norm_rope_counts']==[expected_prefill]+[pair_depth]*row['decode_steps_after_first']
            assert row['prefill_profile'].get('paired_norm_rope_dispatches',0)==expected_prefill""")
source=source.replace("assert (native/(folder.name+'.executed.py')).read_bytes()==expanded","assert (native/folder.with_suffix('.executed.py').name).read_bytes()==expanded")
needle="result['regenerated_native_comparators']=6";assert source.count(needle)==1
source=source.replace(needle,needle+"\nresult['actual_paired_norm_rope_routes']=True\nresult['complete_paired_norm_rope_route_requests']=72")
source=source.replace("scope='Native complete-model ordered vector vision only; first-after-init, warm, initialization and RSS distinct; caches unflushed; no fresh vLLM ratio'","scope='Native complete-model row1/2 paired text/coder norm-RoPE versus accepted vector d52 only; init/first/warm/RSS separate; no fresh vLLM ratio'")
needle="    assert meta['wide_batch_prefill']==dict(candidate_vl_b2_plan=[80],control_vl_b2_plan=[80],logical_inputs_unchanged=True,physical_total_slots=80,no_fresh_vllm_ratio=True)";assert source.count(needle)==1
source=source.replace(needle,needle+"\n    assert meta['paired_norm_rope']==dict(selected_models=['Qwen2.5-Coder-0.5B-Instruct','Qwen3-0.6B'],physical_rows=[1,2],independent_outputs=True,original_values=True,fp32_arithmetic=True,original_reduction_and_precise_trig=True,extra_weights=False)")
ast.parse(source);exec(compile(source,__file__,'exec'))
