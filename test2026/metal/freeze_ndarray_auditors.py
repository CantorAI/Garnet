"""Generate standalone auditors, preserving original bounds and sample assertions."""
from pathlib import Path
s=Path(__file__).resolve().parent
def swap(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)
text=(s/'verify_ndarray_compat.py').read_text()
text=text.replace('make_ndarray_alpha_probe.py','make_ndarray_view_probe.py').replace('ndarray_alpha_gpu.mm','ndarray_view_gpu.mm')
text=swap(text,"    factory=runpy.run_path(str(snap/'test2026/metal/make_ndarray_view_probe.py'))", "    sys.path.insert(0,str(snap/'test2026/metal'))\n    sys.modules.pop('make_ndarray_alpha_probe',None)\n    factory=runpy.run_path(str(snap/'test2026/metal/make_ndarray_view_probe.py'))\n    sys.path.pop(0)")
needle="    ratios=[]"
text=swap(text,needle,"""    views=[v for v in values if v['kind']=='view']
    assert len(views)==8
    for shape in range(4):
        f=fixtures[shape]
        for packed in (True,False):
            v=next(v for v in views if v['shape']==shape and v['packed_requested']==packed)
            assert v['available'] and v['user_buffer_alias']
            assert v['dtype']==(0x10000010 if storage=='mixed' else 0x10000020)
            assert (v['physical_dim0'],v['physical_dim1'])==(f['inner'],f['columns'])
            assert v['logical_shape']==[1,1,f['inner'],f['columns']]
    ratios=[]""")
text=swap(text,"numerical_records=len(quality),checks=ratios)","numerical_records=len(quality),views=views,checks=ratios)")
(s/'verify_ndarray_view_probe.py').write_bytes(text.encode())

text=(s/'verify_packed_body_prefill.py').read_text()
text=text.replace('make_packed_body_prefill.py','make_ndarray_prefill.py').replace('metal_context_packed_prefill.mm','metal_context_ndarray_prefill.mm')
text=swap(text,"harness=out/'packed_body_prefill_gpu.mm'","harness=out/'ndarray_prefill_gpu.mm'")
text=swap(text,"assert harness.read_bytes()==(snap/'test2026/metal/packed_body_prefill_gpu.mm').read_bytes()","assert harness.read_bytes()==factory['harness']((snap/'test2026/metal/packed_body_prefill_gpu.mm').read_text()).encode()")
text=swap(text,"    stages=json.loads((out/'statuses.json').read_text())","    for name,sha in meta['sdk_headers'].items():assert digest(out/name)==sha\n    stages=json.loads((out/'statuses.json').read_text())")
old="""        eligible=sum(256<c['rows']<=8192 and c['inner']%4==0 and c['rows']*c['inner']>=1048576 and not(c['tensor'].startswith('model.layers.') and c['tensor'].endswith(suffix)) for c in assets['cases'])
        assert eligible>0"""
new="""        admissions=proof['global_half_cases'][trial][model]
        assert len(admissions)==len(assets['cases'])
        for c,a in zip(assets['cases'],admissions):
            assert a['name']==c['name'] and a['raw_sha256']==c['sha256']['raw.bin']
        half_eligible=sum(v['global_half_ready'] for v in admissions)
        float_eligible=sum(not(c['tensor'].startswith('model.layers.') and c['tensor'].endswith(suffix)) for c in assets['cases'])
        assert half_eligible>0 and float_eligible>0"""
text=swap(text,old,new)
text=swap(text,"            for policy in (1,2,3):","            for policy in (1,2,3):\n                eligible={1:half_eligible,2:float_eligible,3:0}[policy]")
text=swap(text,"assert check['selected_projections']==eligible and check['values']==rows*sum(c['rows'] for c in assets['cases'])","assert check['selected_projections']==check['actual_ndarray_dispatches']==eligible and check['values']==rows*sum(c['rows'] for c in assets['cases'])")
text=swap(text,"assert check['passed']==(check['finite'] and check['fp64_worst']<=1 and check['comparison_worst']<=1)","assert check['passed']==(check['finite'] and check['fp64_worst']<=1 and check['comparison_worst']<=1 and (policy!=3 or check['bitwise_equal']))\n                    if policy==3:assert check['bitwise_equal'] and check['comparison_worst']==0")
text=text.replace('Audit frozen packed-prefill generation','Audit frozen native NDArray-prefill generation')
text=text.replace('original BF16 values, unchanged numerical bounds.','original BF16 values, unchanged numerical bounds. Mixed three-source alpha/zero addend or exact F32 weight expansion; virtual transpose and typed aliases regenerated. Cache ownership implementation is captured, not a synthetic ownership-suite proof. F32 cache adds storage.')
(s/'verify_ndarray_prefill.py').write_bytes(text.encode())
