"""Audit all72 native outputs, actual vector routes, clocks and source closure."""
from pathlib import Path
import hashlib,json,re,statistics,sys
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
native_trial=sys.argv[3] if len(sys.argv)>3 else 'vector-vision-native-20261016'
assert native_trial in ('vector-vision-native-20261016','vector-vision-native-confirm-20261016') and proof['jobs'][native_trial]['exit_code']==0
saved=sys.argv[:]
try:
    sys.argv=[saved[0],str(base),str(base/'checks-reverification.json')]
    helper=root/'external-tools/verify_vector_vision_native_checks.py'
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
source=source.replace('build-mlp-prefill-graph-guardfix','build-vector-vision-width4')
def replace(old,new):
    global source
    assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
replace("native=root/'logs'/native_trial/'native'","native=root/'logs'/native_trial/'jobs'")
replace("expected_graphs=config['num_hidden_layers']*eligible if policy=='candidate' else 0",'expected_graphs=0')
replace("assert row['mlp_prefill_graph_enabled']==(policy=='candidate')","assert not row['mlp_prefill_graph_enabled']")
replace("            assert row['coalesced_block_enabled']","""            assert row['coalesced_block_enabled']
            assert row['vector_vision_context_enabled']==(policy=='candidate')
            vision_depth=json.loads((root/'models'/model/'config.json').read_text()).get('vision_config',{}).get('depth',0)
            expected_vector=vision_depth if model=='Qwen3.5-0.8B' and policy=='candidate' else 0
            assert row['vision_vector_dispatches']==expected_vector""")
replace("    assert meta['mlp_graph']==dict(candidate=True,control=False,fp32_arithmetic=True,original_bf16_values=True,queue='same-native-queue',synchronous=True,lazy_graph_compile=True,encode_ms_includes_graph_gpu_work=True)","    assert meta['vector_vision']==dict(width=4,tokens=216,hidden=768,heads=12,rows=[216,432],original_fma_order=True,scalar_fallback=True,added_weight_storage=False)")
replace("        assert proof['omitted']['models/'+model+'/'+name]['sha256']==meta['assets'][model][name]","""        asset='models/'+model+'/'+name
        if 'evidence/'+asset in proof['files']:assert digest(root/asset)==meta['assets'][model][name]
        else:assert proof['omitted'][asset]['sha256']==meta['assets'][model][name]""")
replace('source_closure_files=len(identities),actual_graph_routes=True,checks=checks_audit,','source_closure_files=len(identities),external_tool_files=len(external),actual_vector_vision_routes=True,checks=checks_audit,')
replace("scope='Native complete-model comparison only; lazy graph compilation, first-after-init, warm, initialization and RSS distinct; caches unflushed; no fresh vLLM ratio'","scope='Native complete-model ordered vector vision only; first-after-init, warm, initialization and RSS distinct; caches unflushed; no fresh vLLM ratio'")
exec(compile(source,__file__,'exec'))
import runpy,tempfile
helper=root/'external-tools/compare_vector_vision_native.py'
with tempfile.TemporaryDirectory(prefix='garnet-vector-profiles-audit-') as temporary:
    saved=sys.argv[:]
    try:
        sys.argv=['generate','/unused',str(Path(temporary)/'expanded'),'--generate-only']
        runpy.run_path(str(helper),run_name='__main__')
    finally:sys.argv=saved
    expanded=(Path(temporary)/'expanded.executed.py').read_bytes()
for profile in profiles:
    folder=native/Path(profile['evidence']).name;meta=json.loads((folder/'metadata.json').read_text())
    assert (native/(folder.name+'.executed.py')).read_bytes()==expanded
    for name,sha in meta['source_sha256'].items():
        p=(root/'external-tools'/name) if name==helper.name else root/'source/test2026/metal'/name
        assert digest(p)==sha
    assert meta['wide_batch_prefill']==dict(candidate_vl_b2_plan=[80],control_vl_b2_plan=[80],logical_inputs_unchanged=True,physical_total_slots=80,no_fresh_vllm_ratio=True)
result['regenerated_native_comparators']=6
archive_record=json.loads((base/'archive-verification.json').read_text());assert archive_record['verified_files']==len(proof['files'])
result['archive_sha256']=archive_record['archive_sha256'];Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
