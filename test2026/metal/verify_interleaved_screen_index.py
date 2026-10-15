"""Audit complete index roundtrip, bit-identical masks/scores and raw ABBA."""
from pathlib import Path
wrapper=Path(__file__).with_name('verify_screened_seed_abba.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace("meta['control_seed_rows']==8192","meta['interleaved_filter'] and meta['control_seed_rows']==1024")
source=source.replace("('make_screened_seed_abba','make_screened_seed_probe','make_guarded_screened_head')","('make_interleaved_screen_index','make_screened_seed_abba','make_screened_seed_probe','make_guarded_screened_head')")
source=source.replace("importlib.import_module('make_screened_seed_abba')","importlib.import_module('make_interleaved_screen_index')")
needle="    for marker in ('Raw kernel differs";assert source.count(needle)==1
source=source.replace(needle,"    for marker in ('Interleaved index changed a value','Interleaved masked output or flags changed bits','pruned?interleaved:approximate'):assert marker in harness\n"+needle)
needle="        controls=[v for v in records if v['kind']=='control_check']";assert source.count(needle)==1
source=source.replace(needle,"        converted=[v for v in records if v['kind']=='index_roundtrip'];assert len(converted)==1 and converted[0]['all_values_exact'] and converted[0]['elements']==assets['rows']*assets['inner'] and converted[0]['storage']=='interleaved-int8' and converted[0]['padded_columns']==(assets['rows']+3)//4*4\n        roundtrips.append(dict(trial=trial,model=model,**converted[0]))\n        layouts=[v for v in records if v['kind']=='layout_check'];assert len(layouts)==17 and all(v['masked_values_and_flags_bit_exact'] for v in layouts) and {v['query'] for v in layouts}=={q['name'] for q in assets['queries']}\n"+needle)
source=source.replace('control_seed_rows=8192,candidate_seed_rows=1024','control_seed_rows=1024,candidate_seed_rows=1024,interleaved_filter=True')
source=source.replace('complete_gpu_index_roundtrips=roundtrips','complete_cpu_index_roundtrips=roundtrips,cpu_index_roundtrip_checks=len(roundtrips)')
source=source.replace("('profiles','unsafe_input_fallback_checks')","('profiles','unsafe_input_fallback_checks','complete_cpu_index_roundtrips')")
source=source.replace('Same-process exact-seed component only; timing control is screened8192, candidate screened1024; not the full raw head.','Same-process lossless INT8-index layout component only; both seeds1024; same arithmetic order and bit-identical masked scores/flags checked.')
exec(compile(source,__file__,'exec'))
