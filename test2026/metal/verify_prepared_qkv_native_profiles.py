"""Complete72-output native QKV audit, exact routes and regenerated six comparator invocations."""
from pathlib import Path
import ast
p=Path(__file__).with_name('verify_vector_vision_native_profiles.py');source=p.read_text()
source=source.replace('vector-vision-native','prepared-qkv-native').replace('verify_vector_vision_native_checks.py','verify_prepared_qkv_native_checks.py').replace('compare_vector_vision_native.py','compare_prepared_qkv_native.py')
old="source=source.replace('build-mlp-prefill-graph-guardfix','build-vector-vision-width4')";assert source.count(old)==1
source=source.replace(old,"source=source.replace('build-mlp-prefill-graph-guardfix','build-prepared-qkv-text')\nsource=source.replace('build-coalesced-bf16-body','build-vector-vision-width4')")
source=source.replace("assert row['vector_vision_context_enabled']==(policy=='candidate')","assert row['vector_vision_context_enabled']").replace("vision_depth if model=='Qwen3.5-0.8B' and policy=='candidate' else 0","vision_depth if model=='Qwen3.5-0.8B' else 0")
needle="            assert row['vision_vector_dispatches']==expected_vector";assert source.count(needle)==1
source=source.replace(needle,needle+"""
            assert row['prepared_qkv_text_enabled']==(policy=='candidate')
            routing_config=json.loads((root/'models'/model/'config.json').read_text());routing_config=routing_config.get('text_config',routing_config)
            expected_qkv=routing_config['num_hidden_layers']*sum(batch*c in (24,44) for c in row['prefill_chunk_plan']) if model=='Qwen3-0.6B' and policy=='candidate' else 0
            assert row['prepared_qkv_counts']==[expected_qkv]+[0]*row['decode_steps_after_first']
            assert row['prefill_profile'].get('prepared_qkv_dispatches',0)==expected_qkv""")
source=source.replace("assert (native/(folder.name+'.executed.py')).read_bytes()==expanded","assert (native/folder.with_suffix('.executed.py').name).read_bytes()==expanded")
needle="result['regenerated_native_comparators']=6";assert source.count(needle)==1
source=source.replace(needle,needle+"\nresult['actual_prepared_qkv_routes']=True\nresult['complete_qkv_route_requests']=72")
source=source.replace("scope='Native complete-model ordered vector vision only; first-after-init, warm, initialization and RSS distinct; caches unflushed; no fresh vLLM ratio'","scope='Native complete-model prepared text QKV versus accepted vector d52 only; init/first/warm/RSS separate; no fresh vLLM ratio'")
ast.parse(source);exec(compile(source,__file__,'exec'))
