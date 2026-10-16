"""Freeze original72-output auditor with selective route and CPU admission provenance."""
from pathlib import Path
import ast,sys
source=Path(__file__).with_name('verify_ndarray_native_profiles.py').read_text()
def replace(old,new):
    global source
    assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
source=source.replace("'ndarray-native-20261015','ndarray-native-repaired-20261015','ndarray-native-confirm-20261015'",
    "'ndarray-selective-native-20261016','ndarray-selective-native-confirm-20261016'")
source=source.replace("else 'ndarray-native-20261015'", "else 'ndarray-selective-native-20261016'")
source=source.replace("root/'external-tools/verify_ndarray_native_checks.py'", "root/'external-tools/verify_ndarray_selective_native_checks.py'")
source=source.replace('build-ndarray-prefill-shape80-guardrepair','build-ndarray-selective-prefill-packedguard')
replace('''            expected_tensors=[name for name,item in header.items() if name.startswith(('model.layers.','model.language_model.layers.'))
                and item.get('dtype')=='BF16' and len(item.get('shape',[]))==2
                and not(name.startswith('model.layers.') and name.endswith('.self_attn.k_proj.weight'))]''',
    '''            body_tensors=[name for name,item in header.items() if name.startswith(('model.layers.','model.language_model.layers.'))
                and item.get('dtype')=='BF16' and len(item.get('shape',[]))==2
                and not(name.startswith('model.layers.') and name.endswith('.self_attn.k_proj.weight'))]
            admission_file=folder/'weight-admission.json'
            assert digest(admission_file)=='f9505c75246b7ac43fbf1c28987104dd1e5d4c7101ae265e984af20ba246df97'
            admission=next(v for v in json.loads(admission_file.read_text())['models'] if v['model']==model)
            assert admission['checkpoint_sha256']==meta['assets'][model]['model.safetensors']
            assert body_tensors==[c['tensor'] for c in admission['cases']]
            assert all(header[c['tensor']]['shape']==[c['rows'],c['inner']] for c in admission['cases'])
            expected_tensors=[c['tensor'] for c in admission['cases'] if not c['exact_half_ready']]
            common_half_count=sum(c['global_half_ready'] for c in admission['cases'])
            assert len(expected_tensors)==(1 if model=='Qwen3.5-0.8B' else 0)
            assert meta['weight_admission']==dict(report_sha256=digest(admission_file),common_half_matrices=common_half_count,
                body_matrices=len(body_tensors),no_exact_half_matrices=len(expected_tensors))''')
replace("sum(batch*c==80 for c in row['prefill_chunk_plan'])", "sum(batch*c in (80,160) for c in row['prefill_chunk_plan'])")
replace('''            if expected_nd:assert legacy_alpha==0
            else:assert legacy_alpha>0''','''            expected_legacy=common_half_count*sum(2<batch*c<=128 or batch*c==160 for c in row['prefill_chunk_plan'])
            assert legacy_alpha==expected_legacy and legacy_alpha>0''')
replace("dict(physical_rows=80,policy='exact-expanded-F32'", "dict(physical_rows=[80,160],policy='no-exact-half-expanded-F32'")
source=source.replace('Native complete-model row80 NDArray comparison only','Native complete-model selective packed row80/160 NDArray comparison only')
needle="archive_record=json.loads((base/'archive-verification.json').read_text())"
replace(needle,'''# Independently regenerate the exact comparator from its frozen external closure.
import runpy,tempfile
helper=root/'external-tools/compare_ndarray_selective_native.py'
with tempfile.TemporaryDirectory(prefix='garnet-selective-profiles-audit-') as temporary:
    saved=sys.argv[:]
    try:
        sys.argv=['generate','/unused',str(Path(temporary)/'expanded'),'--generate-only']
        runpy.run_path(str(helper),run_name='__main__')
    finally:sys.argv=saved
    expanded=(Path(temporary)/'expanded.executed.py').read_bytes()
for profile in profiles:
    folder=native/Path(profile['evidence']).name
    meta=json.loads((folder/'metadata.json').read_text())
    assert (native/(folder.name+'.executed.py')).read_bytes()==expanded
    for name,sha in meta['source_sha256'].items():
        p=(root/'external-tools'/name) if name==helper.name else root/'source/test2026/metal'/name
        assert digest(p)==sha
    assert meta['wide_batch_prefill']==dict(candidate_vl_b2_plan=[80],control_vl_b2_plan=[80],logical_inputs_unchanged=True,
        physical_rows_by_batch={'1':80,'2':160},no_fresh_vllm_ratio=True)
result['regenerated_native_comparators']=6
result['retained_common_half_counters_verified']=72
'''+needle)
ast.parse(source)
Path(sys.argv[1]).write_bytes(source.encode())
