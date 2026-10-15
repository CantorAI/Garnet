"""Original72 complete-output/clock checks plus independent selected-route counts."""
from pathlib import Path
import hashlib,json,re,statistics,sys
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
native_trial=sys.argv[3] if len(sys.argv)>3 else 'shape-selected-two-native-20261015'
assert native_trial in ('shape-selected-two-native-20261015','shape-selected-two-native-confirm-20261015')
assert proof['jobs'][native_trial]['exit_code']==0
saved=sys.argv[:]
try:
    sys.argv=[saved[0],str(base),str(base/'checks-reverification.json')]
    helper=Path(__file__).with_name('verify_shape_selected_two_checks.py')
    exec(compile(helper.read_text(),str(helper),'exec'),{'__file__':str(helper)})
finally:sys.argv=saved
checks_audit=json.loads((base/'checks-reverification.json').read_text())
jobs=root/'logs'/native_trial/'jobs';audit=json.loads((jobs/'checks-verification.json').read_text())
assert audit['archive_sha256']==proof['checks_archive_sha256']
for key,value in checks_audit.items():
    if key=='verified_files':assert value>=audit[key]
    else:assert value==audit[key],key
identities=json.loads((jobs/'source-identities.json').read_text());assert identities==checks_audit['source_manifest']
for name,sha in identities.items():assert digest(root/'source'/name)==sha and digest(jobs/'source-snapshot'/name)==sha
external=json.loads((jobs/'external-tool-identities.json').read_text())['files']
for name,sha in external.items():assert digest(root/'external-tools'/name)==sha and digest(jobs/'tool-source-snapshot'/name)==sha
template=root/'source/test2026/metal/verify_mlp_prefill_native_profiles.py'
assert digest(template)=='027b85935a861df5040a4a39128cda09f3fe9e814348eb58c4d5e7198eb6ecf8'
source=template.read_text();source=source[source.index("native=root/'logs'/native_trial/'native'"):]
source=source.replace('build-mlp-prefill-graph-guardfix','build-shape-selected-two-body')
def replace(old,new):
    global source
    assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
replace("native=root/'logs'/native_trial/'native'","native=root/'logs'/native_trial/'jobs'")
replace("expected_graphs=config['num_hidden_layers']*eligible if policy=='candidate' else 0",'expected_graphs=0')
replace("assert row['mlp_prefill_graph_enabled']==(policy=='candidate')","assert not row['mlp_prefill_graph_enabled']")
replace("            assert row['coalesced_block_enabled']",'''            assert row['coalesced_block_enabled']
            header=json.loads((root/'checkpoint-headers'/(model+'.json')).read_text())
            table=json.loads((root/'preparation.json').read_text())['selection']['selected']
            expected_tensors={str(b):[name for name,item in header.items() if name.startswith(('model.layers.','model.language_model.layers.'))
                and item.get('dtype')=='BF16' and len(item.get('shape',[]))==2
                and any((b,*item['shape'])==(v['batch'],v['output_rows'],v['inner']) for v in table)
                and not (name.startswith('model.layers.') and name.endswith('.self_attn.k_proj.weight'))] for b in (1,2)}
            assert meta['selected_two_body']['tensors']==expected_tensors
            assert row['two_body_available']==(policy=='candidate')
            expected_two=len(expected_tensors[str(batch)]) if policy=='candidate' else 0
            assert row['two_body_counts']==[0]+[expected_two]*row['decode_steps_after_first']''')
replace("    assert meta['mlp_graph']==dict(candidate=True,control=False,fp32_arithmetic=True,original_bf16_values=True,queue='same-native-queue',synchronous=True,lazy_graph_compile=True,encode_ms_includes_graph_gpu_work=True)",
        "    assert meta['selected_two_body']['table']==[[1,4096,1024,64],[2,896,4864,64],[2,6144,1024,128]]")
replace("        assert proof['omitted']['models/'+model+'/'+name]['sha256']==meta['assets'][model][name]",'''        asset='models/'+model+'/'+name
        if 'evidence/'+asset in proof['files']:assert digest(root/asset)==meta['assets'][model][name]
        else:assert proof['omitted'][asset]['sha256']==meta['assets'][model][name]''')
replace('source_closure_files=len(identities),actual_graph_routes=True,checks=checks_audit,',
        'source_closure_files=len(identities),external_tool_files=len(external),actual_selected_two_routes=True,checks=checks_audit,')
replace("scope='Native complete-model comparison only; lazy graph compilation, first-after-init, warm, initialization and RSS distinct; caches unflushed; no fresh vLLM ratio'", "scope='Native complete-model selected-shape comparison only; first-after-init, warm, initialization and RSS distinct; caches unflushed; no fresh vLLM ratio'")
exec(compile(source,__file__,'exec'))
