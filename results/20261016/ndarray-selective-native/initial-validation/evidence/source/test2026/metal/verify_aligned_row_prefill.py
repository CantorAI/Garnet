"""Independent archive, original-bound, zero-tail and physical-route audit."""
from pathlib import Path
source=Path(__file__).with_name('verify_packed_body_prefill.py').read_text()
for old,new in [('make_packed_body_prefill','make_aligned_row_prefill'),('metal_context_packed_prefill','metal_context_aligned_rows'),('packed_body_prefill_gpu','aligned_row_prefill_gpu')]:source=source.replace(old,new)
start=source.index('        eligible=sum(');end=source.index('        for batch in (1,2):',start)
source=source[:start]+'''        admission=proof['aligned_global_half_cases'][trial][model]
        assert len(admission)==len(assets['cases'])
        for record,case in zip(admission,assets['cases']):
            assert record['name']==case['name'] and record['raw_sha256']==case['sha256']['raw.bin']
        eligible=sum(record['global_half_ready'] for record in admission)
        assert eligible>0
'''+source[end:]
source=source.replace("check['selected_projections']==eligible","check['selected_projections']==(eligible if rows%(16<<policy) else 0)")
source=source.replace("                    assert check['rows']==rows", "                    assert check['padded_rows_zero']\n                    assert check['rows']==rows")
source=source.replace('Private independent first-layer projections; original FP32 activation and accumulation, original BF16 values, unchanged numerical bounds. No synthetic ownership suite, complete-model or vLLM speed claim.','Private zero-backed MPS physical row alignment with logical shapes unchanged. Independent CPU exact-half admission, original numerical bounds and zero padding. Initialization/padded storage outside timing; no full-model or vLLM speed claim.')
exec(compile(source,__file__,'exec'))
