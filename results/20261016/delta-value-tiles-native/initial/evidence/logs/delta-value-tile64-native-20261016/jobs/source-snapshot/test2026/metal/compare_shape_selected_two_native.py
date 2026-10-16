"""Complete native ABBA: selected two-output shapes versus retained c186."""
from pathlib import Path
import sys
p=Path(__file__).with_name('compare_coalesced_body_native.py')
wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('\ndestination=')],str(p),'exec'),ns);source=ns['source']
def replace(old,new):
    global source
    assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
replace("[('control','build-wide-batch-prefill'),('candidate','build-coalesced-bf16-body')]",
        "[('control','build-coalesced-bf16-body'),('candidate','build-shape-selected-two-body')]")
replace("assert metadata['library_sha256']['control']==json.loads((r/'work/wide-batch-next-comparator-selection.json').read_text())['builds']['wide_batch']['sha256']",
        "assert metadata['library_sha256']['control']=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'")
replace('logs/coalesced-body-native-checks-20261015/jobs/metadata.json','logs/shape-selected-two-checks-20261015/jobs/metadata.json')
replace("assert value['interleaved_body_enabled']==(policy=='candidate')","assert value['interleaved_body_enabled']")
replace("assert value['coalesced_block_enabled']==(policy=='candidate')","assert value['coalesced_block_enabled']")
replace("all((v>0)==(policy=='candidate') for v in compressed[1:])","all(v>0 for v in compressed[1:])")
replace("all((v>0)==(policy=='candidate') for v in body[1:])","all(v>0 for v in body[1:])")
replace("    metadata['scope']='Full native ABBA: original latency bundle, exact coalesced BF16 body blocks with64-thread decode; both VL B2 plans80. No fresh vLLM ratio.'", "    metadata['scope']='Full native ABBA: identical learned values/FP32 arithmetic and latency bundle; three selected two-output decode shapes, original four-output elsewhere. No fresh vLLM ratio.'")
replace("    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))",'''    # Count actual eligible learned tensors independently of measured counters.
    import struct
    with (folder/'model.safetensors').open('rb') as checkpoint:
        header_size=struct.unpack('<Q',checkpoint.read(8))[0]
        tensor_header=json.loads(checkpoint.read(header_size))
    from make_shape_selected_two_body import TABLE
    two_tensors={str(b):[name for name,item in tensor_header.items() if name.startswith(('model.layers.','model.language_model.layers.'))
        and item.get('dtype')=='BF16' and len(item.get('shape',[]))==2
        and any((b,*item['shape'])==(rows,n,k) for rows,n,k,t in TABLE)
        and not (name.startswith('model.layers.') and name.endswith('.self_attn.k_proj.weight'))] for b in (1,2)}
    metadata['selected_two_body']=dict(table=TABLE,tensors=two_tensors)
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))''')
replace("                wide_count=value['prefill_profile'].get('wide_batch_prefill_dispatches',0)",'''                assert value['two_body_available']==(policy=='candidate')
                expected_two=len(two_tensors[str(batch)]) if policy=='candidate' else 0
                assert value['two_body_counts']==[0]+[expected_two]*value['decode_steps_after_first']
                assert not value['mlp_prefill_graph_enabled'] and not any(value['mlp_prefill_graph_counts'])
                wide_count=value['prefill_profile'].get('wide_batch_prefill_dispatches',0)''')
source=repr(__doc__)+source[source.index('\n'):]
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,str(destination),'exec'))
