"""Six-profile native ABBA: only unsupported exact-half matrix, retained original routes."""
from pathlib import Path
import ast,sys
p=Path(__file__).with_name('compare_ndarray_native.py');ns={'__file__':str(p)};wrapper=p.read_text()
exec(compile(wrapper[:wrapper.rindex('\ndestination=')],str(p),'exec'),ns);source=ns['source']
def replace(old,new):
    global source
    assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
source=source.replace('build-ndarray-prefill-shape80-guardrepair','build-ndarray-selective-prefill-packedguard')
source=source.replace('ndarray-native-checks-repaired-20261015','ndarray-selective-native-packed-checks-20261016')
source=source.replace('row80 exact-expanded-F32 NDArray projection candidate','packed row80/160 no-exact-half-only F32 NDArray projection candidate')
source=source.replace('physical_total_slots=80',"physical_rows_by_batch={'1':80,'2':160}")
old='''    ndarray_tensors=[name for name,item in tensor_header.items() if name.startswith(('model.layers.','model.language_model.layers.'))
        and item.get('dtype')=='BF16' and len(item.get('shape',[]))==2
        and not(name.startswith('model.layers.') and name.endswith('.self_attn.k_proj.weight'))]
    assert len(ndarray_tensors)<=256'''
replace(old,'''    body_tensors=[name for name,item in tensor_header.items() if name.startswith(('model.layers.','model.language_model.layers.'))
        and item.get('dtype')=='BF16' and len(item.get('shape',[]))==2
        and not(name.startswith('model.layers.') and name.endswith('.self_attn.k_proj.weight'))]
    parent=Path(json.loads((r/'preparation.json').read_text())['parent'])
    admission_bytes=(parent/'logs/ndarray-weight-admission-20261016/report.json').read_bytes()
    admission_sha=hashlib.sha256(admission_bytes).hexdigest()
    assert admission_sha=='f9505c75246b7ac43fbf1c28987104dd1e5d4c7101ae265e984af20ba246df97'
    (output/'weight-admission.json').write_bytes(admission_bytes)
    admission=next(v for v in json.loads(admission_bytes)['models'] if v['model']==model)
    assert admission['checkpoint_sha256']==metadata['assets'][model]['model.safetensors']
    assert body_tensors==[c['tensor'] for c in admission['cases']]
    assert all(tensor_header[c['tensor']]['shape']==[c['rows'],c['inner']] for c in admission['cases'])
    ndarray_tensors=[c['tensor'] for c in admission['cases'] if not c['exact_half_ready']]
    common_half_count=sum(c['global_half_ready'] for c in admission['cases'])
    assert len(ndarray_tensors)==(1 if vl else 0)
    metadata['weight_admission']=dict(report_sha256=admission_sha,common_half_matrices=common_half_count,
        body_matrices=len(body_tensors),no_exact_half_matrices=len(ndarray_tensors))''')
replace("physical_rows=80,policy='exact-expanded-F32'", "physical_rows=[80,160],policy='no-exact-half-expanded-F32'")
replace("expected_nd=len(ndarray_tensors)*sum(batch*c==80 for c in value['prefill_chunk_plan']) if policy=='candidate' else 0",
    "expected_nd=len(ndarray_tensors)*sum(batch*c in (80,160) for c in value['prefill_chunk_plan']) if policy=='candidate' else 0")
replace('''                if expected_nd:assert legacy_alpha==0
                else:assert legacy_alpha>0''','''                expected_legacy=common_half_count*sum(2<batch*c<=128 or batch*c==160 for c in value['prefill_chunk_plan'])
                assert legacy_alpha==expected_legacy and legacy_alpha>0''')
source=repr(__doc__)+source[source.index('\n'):];ast.parse(source)
destination=Path(sys.argv[2]).with_name(Path(sys.argv[2]).name+'.executed.py');destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,str(destination),'exec'))
