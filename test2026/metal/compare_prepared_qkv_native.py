"""Complete native ABBA: confirmed text QKV versus accepted vector d52 control."""
from pathlib import Path
import ast,sys
p=Path(__file__).with_name('compare_vector_vision_native.py');ns={'__file__':str(p)};wrapper=p.read_text();exec(compile(wrapper[:wrapper.rindex('\ndestination=')],str(p),'exec'),ns);source=ns['source']
def replace(old,new,count=1):
 global source
 assert source.count(old)==count,(old,source.count(old));source=source.replace(old,new)
replace("[('control','build-coalesced-bf16-body'),('candidate','build-vector-vision-width4')]","[('control','build-vector-vision-width4'),('candidate','build-prepared-qkv-text')]")
replace("assert metadata['library_sha256']['control']=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'","assert metadata['library_sha256']['control']=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'")
replace('logs/vector-vision-native-checks-20261016/jobs/metadata.json','logs/prepared-qkv-native-checks-countfix-20261016/jobs/metadata.json')
replace("assert value['vector_vision_context_enabled']==(policy=='candidate')","assert value['vector_vision_context_enabled']")
replace("(vision_depth if vl and policy=='candidate' else 0)","(vision_depth if vl else 0)")
needle="                assert value['vector_vision_context_enabled']"
replace(needle,"""                assert value['prepared_qkv_text_enabled']==(policy=='candidate')
                routing_config=json.loads((folder/'config.json').read_text());routing_config=routing_config.get('text_config',routing_config)
                expected_qkv=routing_config['num_hidden_layers']*sum(batch*c in (24,44) for c in value['prefill_chunk_plan']) if model=='Qwen3-0.6B' and policy=='candidate' else 0
                assert value['prepared_qkv_counts']==[expected_qkv]+[0]*value['decode_steps_after_first']
                assert value['prefill_profile'].get('prepared_qkv_dispatches',0)==expected_qkv
"""+needle)
needle="    metadata['vector_vision']=dict(width=4,tokens=216,hidden=768,heads=12,rows=[216,432],original_fma_order=True,scalar_fallback=True,added_weight_storage=False)"
replace(needle,needle+"\n    metadata['prepared_qkv']=dict(selected_models=['Qwen3-0.6B'],physical_rows=[24,44],all_three_outputs=True,original_values=True,fp32_arithmetic=True,session_prepared_during_build=True,extra_union_weights=True)")
replace("metadata['scope']='Full native ABBA: original learned values/FP32 arithmetic, all inputs and limits; ordered float4 vision context only. Init/first/warm and RSS separate; no fresh vLLM ratio.'","metadata['scope']='Full native ABBA: prepared exact text QKV against accepted vector d52; all six profiles and complete outputs. Init/first/warm and RSS separate, no fresh vLLM ratio.'")
source=repr(__doc__)+source[source.index('\n'):];ast.parse(source);destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,str(destination),'exec'))
