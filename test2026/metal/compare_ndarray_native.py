"""Full native ABBA: repeated row80 NDArray candidate versus retained c186."""
from pathlib import Path
import sys
p=Path(__file__).with_name('compare_coalesced_body_native.py');ns={'__file__':str(p)};wrapper=p.read_text()
exec(compile(wrapper[:wrapper.rindex('\ndestination=')],str(p),'exec'),ns);source=ns['source']
def replace(old,new):
    global source
    assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
replace("[('control','build-wide-batch-prefill'),('candidate','build-coalesced-bf16-body')]","[('control','build-coalesced-bf16-body'),('candidate','build-ndarray-prefill-shape80-guardrepair')]")
replace("assert metadata['library_sha256']['control']==json.loads((r/'work/wide-batch-next-comparator-selection.json').read_text())['builds']['wide_batch']['sha256']","assert metadata['library_sha256']['control']=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'")
replace('logs/coalesced-body-native-checks-20261015/jobs/metadata.json','logs/ndarray-native-checks-repaired-20261015/jobs/metadata.json')
replace("assert value['interleaved_body_enabled']==(policy=='candidate')","assert value['interleaved_body_enabled']")
replace("assert value['coalesced_block_enabled']==(policy=='candidate')","assert value['coalesced_block_enabled']")
replace("all((v>0)==(policy=='candidate') for v in compressed[1:])","all(v>0 for v in compressed[1:])")
replace("all((v>0)==(policy=='candidate') for v in body[1:])","all(v>0 for v in body[1:])")
replace("    metadata['scope']='Full native ABBA: original latency bundle, exact coalesced BF16 body blocks with64-thread decode; both VL B2 plans80. No fresh vLLM ratio.'","    metadata['scope']='Full native ABBA: original learned values, FP32 arithmetic, inputs/limits/chunks; row80 exact-expanded-F32 NDArray projection candidate. Init/first/warm and peak RSS separate; no fresh vLLM ratio.'")
replace("    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))","""    import struct
    with (folder/'model.safetensors').open('rb') as checkpoint:
        header_size=struct.unpack('<Q',checkpoint.read(8))[0];tensor_header=json.loads(checkpoint.read(header_size))
    ndarray_tensors=[name for name,item in tensor_header.items() if name.startswith(('model.layers.','model.language_model.layers.'))
        and item.get('dtype')=='BF16' and len(item.get('shape',[]))==2
        and not(name.startswith('model.layers.') and name.endswith('.self_attn.k_proj.weight'))]
    assert len(ndarray_tensors)<=256
    metadata['ndarray_prefill']=dict(physical_rows=80,policy='exact-expanded-F32',fp32_activations=True,fp32_results=True,
        source_count=2,alpha=1,beta=0,cache_limit=256,native_fallback=True,added_weight_storage=True,tensors=ndarray_tensors)
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))""")
replace("                wide_count=value['prefill_profile'].get('wide_batch_prefill_dispatches',0)","""                assert value['ndarray_prefill_available']==(policy=='candidate')
                expected_nd=len(ndarray_tensors)*sum(batch*c==80 for c in value['prefill_chunk_plan']) if policy=='candidate' else 0
                assert value['ndarray_prefill_counts']==[expected_nd]+[0]*value['decode_steps_after_first']
                assert value['prefill_profile'].get('ndarray_prefill_dispatches',0)==expected_nd
                assert not value['mlp_prefill_graph_enabled'] and not any(value['mlp_prefill_graph_counts'])
                wide_count=value['prefill_profile'].get('wide_batch_prefill_dispatches',0)""")
replace("                assert value['prefill_profile'].get('global_half_alpha_dispatches',0)>0","""                legacy_alpha=value['prefill_profile'].get('global_half_alpha_dispatches',0)
                if expected_nd:assert legacy_alpha==0
                else:assert legacy_alpha>0""")
source=repr(__doc__)+source[source.index('\n'):]
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,str(destination),'exec'))
