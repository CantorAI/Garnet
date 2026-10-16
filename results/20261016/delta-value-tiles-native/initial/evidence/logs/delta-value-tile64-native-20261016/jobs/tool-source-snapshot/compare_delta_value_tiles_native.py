"""Complete native ABBA: width64 chunk80 recurrence versus accepted vector d52."""
from pathlib import Path
import ast,sys
p=Path(__file__).with_name('compare_vector_vision_native.py');ns={'__file__':str(p)};wrapper=p.read_text();exec(compile(wrapper[:wrapper.rindex('\ndestination=')],str(p),'exec'),ns);source=ns['source']

def replace(old,new,count=1):
 global source
 assert source.count(old)==count,(old,source.count(old),count);source=source.replace(old,new)

replace("[('control','build-coalesced-bf16-body'),('candidate','build-vector-vision-width4')]","[('control','build-vector-vision-width4'),('candidate','build-delta-value-tile64-prefill')]")
replace("assert metadata['library_sha256']['control']=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'","assert metadata['library_sha256']['control']=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'")
replace('logs/vector-vision-native-checks-20261016/jobs/metadata.json','logs/delta-value-tile64-native-checks-20261016/jobs/metadata.json')
replace("assert value['vector_vision_context_enabled']==(policy=='candidate')","assert value['vector_vision_context_enabled']");replace('(vision_depth if vl and policy==\'candidate\' else 0)','(vision_depth if vl else 0)')
needle="                assert value['vector_vision_context_enabled']"
replace(needle,"""                assert value['delta_value_tile64_enabled']==(policy=='candidate')
                routing_config=json.loads((folder/'config.json').read_text());routing_config=routing_config.get('text_config',routing_config)
                linear_depth=sum(kind=='linear_attention' for kind in routing_config.get('layer_types',[])) if vl and policy=='candidate' else 0
                if linear_depth:assert linear_depth==18 and routing_config['linear_num_value_heads']==16 and routing_config['linear_key_head_dim']==routing_config['linear_value_head_dim']==128
                expected_prefill=linear_depth*sum(chunk==80 for chunk in value['prefill_chunk_plan'])
                assert value['delta_value_tile64_counts']==[expected_prefill]+[0]*value['decode_steps_after_first']
                assert value['prefill_profile'].get('delta_value_tile64_dispatches',0)==expected_prefill
"""+needle)
needle="    metadata['vector_vision']=dict(width=4,tokens=216,hidden=768,heads=12,rows=[216,432],original_fma_order=True,scalar_fallback=True,added_weight_storage=False)"
replace(needle,needle+"\n    metadata['delta_value_tile64']=dict(batch=[1,2],chunk=80,heads=16,key_dim=128,value_dim=128,tile=64,threads_per_group=128,original_values=True,fp32_arithmetic=True,original_arithmetic=True,extra_weights=False)")
replace("metadata['scope']='Full native ABBA: original learned values/FP32 arithmetic, all inputs and limits; ordered float4 vision context only. Init/first/warm and RSS separate; no fresh vLLM ratio.'","metadata['scope']='Full native ABBA: width64 chunk80 recurrence versus accepted d52, all six profiles and complete outputs. Init/first/warm and RSS separate; no fresh vLLM ratio.'")
source=repr(__doc__)+source[source.index('\n'):];ast.parse(source);destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,str(destination),'exec'))
