"""Complete native ABBA: ordered vector vision versus retained scalar c186."""
from pathlib import Path
import ast,sys
p=Path(__file__).with_name('compare_coalesced_body_native.py');ns={'__file__':str(p)};wrapper=p.read_text()
exec(compile(wrapper[:wrapper.rindex('\ndestination=')],str(p),'exec'),ns);source=ns['source']
def replace(old,new):
    global source
    assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
replace("[('control','build-wide-batch-prefill'),('candidate','build-coalesced-bf16-body')]","[('control','build-coalesced-bf16-body'),('candidate','build-vector-vision-width4')]")
replace("assert metadata['library_sha256']['control']==json.loads((r/'work/wide-batch-next-comparator-selection.json').read_text())['builds']['wide_batch']['sha256']","assert metadata['library_sha256']['control']=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'")
replace('logs/coalesced-body-native-checks-20261015/jobs/metadata.json','logs/vector-vision-native-checks-20261016/jobs/metadata.json')
replace("assert value['interleaved_body_enabled']==(policy=='candidate')","assert value['interleaved_body_enabled']")
replace("assert value['coalesced_block_enabled']==(policy=='candidate')","assert value['coalesced_block_enabled']")
replace("all((v>0)==(policy=='candidate') for v in compressed[1:])","all(v>0 for v in compressed[1:])")
replace("all((v>0)==(policy=='candidate') for v in body[1:])","all(v>0 for v in body[1:])")
replace("    metadata['scope']='Full native ABBA: original latency bundle, exact coalesced BF16 body blocks with64-thread decode; both VL B2 plans80. No fresh vLLM ratio.'","    metadata['scope']='Full native ABBA: original learned values/FP32 arithmetic, all inputs and limits; ordered float4 vision context only. Init/first/warm and RSS separate; no fresh vLLM ratio.'\n    metadata['vector_vision']=dict(width=4,tokens=216,hidden=768,heads=12,rows=[216,432],original_fma_order=True,scalar_fallback=True,added_weight_storage=False)")
replace("                wide_count=value['prefill_profile'].get('wide_batch_prefill_dispatches',0)","""                assert value['vector_vision_context_enabled']==(policy=='candidate')
                vision_depth=json.loads((folder/'config.json').read_text()).get('vision_config',{}).get('depth',0)
                assert value['vision_vector_dispatches']==(vision_depth if vl and policy=='candidate' else 0)
                assert not value['mlp_prefill_graph_enabled'] and not any(value['mlp_prefill_graph_counts'])
                wide_count=value['prefill_profile'].get('wide_batch_prefill_dispatches',0)""")
source=repr(__doc__)+source[source.index('\n'):];ast.parse(source)
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,str(destination),'exec'))
