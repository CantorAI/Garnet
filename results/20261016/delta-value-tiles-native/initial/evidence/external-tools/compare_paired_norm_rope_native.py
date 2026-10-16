"""Complete native ABBA: confirmed text/coder normalization-RoPE versus accepted vector d52 control."""
from pathlib import Path
import ast,sys
p=Path(__file__).with_name('compare_vector_vision_native.py');ns={'__file__':str(p)};wrapper=p.read_text();exec(compile(wrapper[:wrapper.rindex('\ndestination=')],str(p),'exec'),ns);source=ns['source']
def replace(old,new,count=1):
 global source
 assert source.count(old)==count,(old,source.count(old));source=source.replace(old,new)
replace("[('control','build-coalesced-bf16-body'),('candidate','build-vector-vision-width4')]","[('control','build-vector-vision-width4'),('candidate','build-paired-norm-rope-decode')]")
replace("assert metadata['library_sha256']['control']=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'","assert metadata['library_sha256']['control']=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'")
replace('logs/vector-vision-native-checks-20261016/jobs/metadata.json','logs/paired-norm-rope-native-checks-20261016/jobs/metadata.json')
replace("assert value['vector_vision_context_enabled']==(policy=='candidate')","assert value['vector_vision_context_enabled']")
replace("(vision_depth if vl and policy=='candidate' else 0)","(vision_depth if vl else 0)")
needle="                assert value['vector_vision_context_enabled']"
replace(needle,"""                assert value['paired_norm_rope_enabled']==(policy=='candidate')
                routing_config=json.loads((folder/'config.json').read_text());routing_config=routing_config.get('text_config',routing_config)
                selected_pair=model in ['Qwen2.5-Coder-0.5B-Instruct','Qwen3-0.6B'] and policy=='candidate'
                pair_depth=routing_config['num_hidden_layers'] if selected_pair else 0
                expected_prefill=pair_depth*sum(batch*c in (1,2) for c in value['prefill_chunk_plan'])
                assert value['paired_norm_rope_counts']==[expected_prefill]+[pair_depth]*value['decode_steps_after_first']
                assert value['prefill_profile'].get('paired_norm_rope_dispatches',0)==expected_prefill
"""+needle)
needle="    metadata['vector_vision']=dict(width=4,tokens=216,hidden=768,heads=12,rows=[216,432],original_fma_order=True,scalar_fallback=True,added_weight_storage=False)"
replace(needle,needle+"\n    metadata['paired_norm_rope']=dict(selected_models=['Qwen2.5-Coder-0.5B-Instruct','Qwen3-0.6B'],physical_rows=[1,2],independent_outputs=True,original_values=True,fp32_arithmetic=True,original_reduction_and_precise_trig=True,extra_weights=False)")
replace("metadata['scope']='Full native ABBA: original learned values/FP32 arithmetic, all inputs and limits; ordered float4 vision context only. Init/first/warm and RSS separate; no fresh vLLM ratio.'","metadata['scope']='Full native ABBA: row1/2 paired text/coder normalization-RoPE against accepted vector d52; all six profiles and complete outputs. Init/first/warm and RSS separate, no fresh vLLM ratio.'")
source=repr(__doc__)+source[source.index('\n'):];ast.parse(source);destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,str(destination),'exec'))
