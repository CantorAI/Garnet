"""Original72 full-output/clock checks plus independently derived NDArray prefill routes."""
from pathlib import Path
import hashlib,json,re,statistics,sys
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
native_trial=sys.argv[3] if len(sys.argv)>3 else 'ndarray-native-20261015'
assert native_trial in ('ndarray-native-20261015','ndarray-native-confirm-20261015') and proof['jobs'][native_trial]['exit_code']==0
saved=sys.argv[:]
try:
    sys.argv=[saved[0],str(base),str(base/'checks-reverification.json')]
    helper=Path(__file__).with_name('verify_ndarray_native_checks.py')
    exec(compile(helper.read_text(),str(helper),'exec'),{'__file__':str(helper)})
finally:sys.argv=saved
checks_audit=json.loads((base/'checks-reverification.json').read_text());jobs=root/'logs'/native_trial/'jobs'
audit=json.loads((jobs/'checks-verification.json').read_text());assert audit['archive_sha256']==proof['checks_archive_sha256']
for key,value in checks_audit.items():
    if key=='verified_files':assert value>=audit[key]
    else:assert value==audit[key],key
identities=json.loads((jobs/'source-identities.json').read_text());assert identities==checks_audit['source_manifest']
for name,sha in identities.items():assert digest(root/'source'/name)==digest(jobs/'source-snapshot'/name)==sha
external=json.loads((jobs/'external-tool-identities.json').read_text())['files']
for name,sha in external.items():assert digest(root/'external-tools'/name)==digest(jobs/'tool-source-snapshot'/name)==sha
template=root/'source/test2026/metal/verify_mlp_prefill_native_profiles.py'
assert digest(template)=='027b85935a861df5040a4a39128cda09f3fe9e814348eb58c4d5e7198eb6ecf8'
source=template.read_text();source=source[source.index("native=root/'logs'/native_trial/'native'"):]
source=source.replace('build-mlp-prefill-graph-guardfix','build-ndarray-prefill-shape80-guardrepair')
def replace(old,new):
    global source
    assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
replace("native=root/'logs'/native_trial/'native'","native=root/'logs'/native_trial/'jobs'")
replace("expected_graphs=config['num_hidden_layers']*eligible if policy=='candidate' else 0",'expected_graphs=0')
replace("assert row['mlp_prefill_graph_enabled']==(policy=='candidate')","assert not row['mlp_prefill_graph_enabled']")
replace("            assert row['coalesced_block_enabled']","""            assert row['coalesced_block_enabled']
            header=json.loads((root/'checkpoint-headers'/(model+'.json')).read_text())
            expected_tensors=[name for name,item in header.items() if name.startswith(('model.layers.','model.language_model.layers.'))
                and item.get('dtype')=='BF16' and len(item.get('shape',[]))==2
                and not(name.startswith('model.layers.') and name.endswith('.self_attn.k_proj.weight'))]
            assert meta['ndarray_prefill']['tensors']==expected_tensors and len(expected_tensors)<=256
            expected_nd=len(expected_tensors)*sum(batch*c==80 for c in row['prefill_chunk_plan']) if policy=='candidate' else 0
            assert row['ndarray_prefill_available']==(policy=='candidate')
            assert row['ndarray_prefill_counts']==[expected_nd]+[0]*row['decode_steps_after_first']
            assert row['prefill_profile'].get('ndarray_prefill_dispatches',0)==expected_nd""")
replace("    assert meta['mlp_graph']==dict(candidate=True,control=False,fp32_arithmetic=True,original_bf16_values=True,queue='same-native-queue',synchronous=True,lazy_graph_compile=True,encode_ms_includes_graph_gpu_work=True)","""    nd=meta['ndarray_prefill']
    assert {k:v for k,v in nd.items() if k!='tensors'}==dict(physical_rows=80,policy='exact-expanded-F32',fp32_activations=True,
        fp32_results=True,source_count=2,alpha=1,beta=0,cache_limit=256,native_fallback=True,added_weight_storage=True)""")
replace("        assert proof['omitted']['models/'+model+'/'+name]['sha256']==meta['assets'][model][name]","""        asset='models/'+model+'/'+name
        if 'evidence/'+asset in proof['files']:assert digest(root/asset)==meta['assets'][model][name]
        else:assert proof['omitted'][asset]['sha256']==meta['assets'][model][name]""")
replace('source_closure_files=len(identities),actual_graph_routes=True,checks=checks_audit,','source_closure_files=len(identities),external_tool_files=len(external),actual_ndarray_prefill_routes=True,checks=checks_audit,')
replace("scope='Native complete-model comparison only; lazy graph compilation, first-after-init, warm, initialization and RSS distinct; caches unflushed; no fresh vLLM ratio'","scope='Native complete-model row80 NDArray comparison only; first-after-init, warm, initialization and RSS distinct; added FP32 weight storage; caches unflushed; no fresh vLLM ratio'")
exec(compile(source,__file__,'exec'))
archive_record=json.loads((base/'archive-verification.json').read_text());assert archive_record['verified_files']==len(proof['files'])
result['archive_sha256']=archive_record['archive_sha256'];Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
