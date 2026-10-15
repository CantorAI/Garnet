"""Independent exact native bits, FP64, safety controls and raw-window audit."""
from pathlib import Path
source=Path(__file__).with_name('verify_packed_body_prefill.py').read_text()
for old,new in [('make_packed_body_prefill','make_wide_body_decode'),('metal_context_packed_prefill','metal_context_wide_body'),('packed_body_prefill_gpu','wide_body_decode_gpu')]:source=source.replace(old,new)
source=source.replace("rows=80*batch if model=='Qwen3.5-0.8B' else (24 if batch==1 else 44) if model=='Qwen3-0.6B' else 26*batch","rows=batch")
source=source.replace("all(v['kind'] in ('check','samples') for v in values)","all(v['kind'] in ('check','samples','controls') for v in values)")
needle="        selected=[v for v in values if v['kind']=='check']";assert source.count(needle)==1
source=source.replace(needle,"        assert [v for v in values if v['kind']=='controls']==[dict(kind='controls',profiles=63,passed=True)]\n"+needle)
source=source.replace("check['passed']==(check['finite'] and check['fp64_worst']<=1 and check['comparison_worst']<=1)","check['passed']==(check['finite'] and check['bitwise_equal'] and check['fp64_worst']<=1 and check['comparison_worst']<=1)\n                    if check['bitwise_equal']:assert check['comparison_worst']==0")
source=source.replace('numeric_checks=len(all_checks),','numeric_checks=len(all_checks),control_profiles=len(proof[\'jobs\'])*189,')
source=source.replace('Private independent first-layer projections; original FP32 activation and accumulation, original BF16 values, unchanged numerical bounds. No synthetic ownership suite, complete-model or vLLM speed claim.','Private eight-output BF16 decode versus current four-output layout. Exact native output bits, original FP64 bound, tail/fallback/prefill/ownership/abort tests.64/128/256-thread variants. No full-model or vLLM speed claim.')
exec(compile(source,__file__,'exec'))
