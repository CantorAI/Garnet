"""Native ABBA with identical coalesced decode, complete MLP graph only changed."""
from pathlib import Path
import sys

p=Path(__file__).with_name('compare_coalesced_body_native.py')
wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('\ndestination=')],str(p),'exec'),ns)
source=ns['source']

def replace(old,new):
    global source
    assert source.count(old)==1,(old,source.count(old))
    source=source.replace(old,new)

replace("[('control','build-wide-batch-prefill'),('candidate','build-coalesced-bf16-body')]",
        "[('control','build-coalesced-bf16-body'),('candidate','build-mlp-prefill-graph-guardfix')]")
replace("assert metadata['library_sha256']['control']==json.loads((r/'work/wide-batch-next-comparator-selection.json').read_text())['builds']['wide_batch']['sha256']",
        "assert metadata['library_sha256']['control']=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'")
replace('logs/coalesced-body-native-checks-20261015/jobs/metadata.json',
        'logs/mlp-prefill-native-checks-resume-20261015/jobs/metadata.json')
replace("metadata['scope']='Full native ABBA: original latency bundle, exact coalesced BF16 body blocks with64-thread decode; both VL B2 plans80. No fresh vLLM ratio.'",
        "metadata['scope']='Full native ABBA: identical coalesced decode and latency bundle; OFF-default synchronous same-queue complete MLP graph during prefill. No fresh vLLM ratio.'\n    metadata['mlp_graph']=dict(candidate=True,control=False,fp32_arithmetic=True,original_bf16_values=True,queue='same-native-queue',synchronous=True,lazy_graph_compile=True,encode_ms_includes_graph_gpu_work=True)")
replace("assert value['interleaved_body_enabled']==(policy=='candidate')","assert value['interleaved_body_enabled']")
replace("assert value['coalesced_block_enabled']==(policy=='candidate')","assert value['coalesced_block_enabled']")
replace("all((v>0)==(policy=='candidate') for v in compressed[1:])","all(v>0 for v in compressed[1:])")
replace("all((v>0)==(policy=='candidate') for v in body[1:])","all(v>0 for v in body[1:])")
replace("                assert value['prefill_profile'].get('prefill_partial_submissions',0)==submissions",'''                # Derived from layer count and physical profile rows, not the
                # measured dispatch count. Unsupported decode/tiny rows use native.
                mlps=config['num_hidden_layers']*sum(2<batch*chunk<=128 or batch*chunk==160 for chunk in value['prefill_chunk_plan']) if policy=='candidate' else 0
                assert value['mlp_prefill_graph_enabled']==(policy=='candidate')
                graph_counts=value['mlp_prefill_graph_counts']
                assert graph_counts==[mlps]+[0]*value['decode_steps_after_first']
                assert value['prefill_profile'].get('mlp_prefill_graph_dispatches',0)==mlps
                assert value['prefill_profile'].get('prefill_partial_submissions',0)==submissions+mlps''')
source=repr(__doc__)+source[source.index('\n'):]
destination=Path(sys.argv[2]).with_suffix('.executed.py')
destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
