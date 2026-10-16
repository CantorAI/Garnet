"""Same-process INT8 index layouts, both exact seeds 1024 and original refinement."""
from pathlib import Path
import sys
wrapper=Path(__file__).with_name('run_screened_seed_abba.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('\ndestination=')],__file__,'exec'),ns);source=ns['source']
source=source.replace('from make_screened_seed_abba import generate_harness as generate,generate_shader','from make_interleaved_screen_index import generate_harness as generate,generate_shader')
source=source.replace("paths=[native,Path(__file__),","paths=[native,Path(__file__),s/'make_interleaved_screen_index.py',s/'run_screened_seed_abba.py',")
source=source.replace('control_seed_rows=8192,candidate_seed_rows=1024','interleaved_filter=True,control_seed_rows=1024,candidate_seed_rows=1024')
source=source.replace("'exact 8192-row control versus exact 1024-row candidate seed'","'both exact 1024-row seeds; row-major versus four-column interleaved INT8 filter index'")
needle="    controls=[v for v in records if v['kind']=='control_check']";assert source.count(needle)==1
source=source.replace(needle,"    roundtrips=[v for v in records if v['kind']=='index_roundtrip'];assert len(roundtrips)==1 and roundtrips[0]['all_values_exact'] and roundtrips[0]['elements']==info['rows']*info['inner'] and roundtrips[0]['storage']=='interleaved-int8'\n    layouts=[v for v in records if v['kind']=='layout_check'];assert len(layouts)==17 and all(v['masked_values_and_flags_bit_exact'] for v in layouts) and {v['query'] for v in layouts}=={q['name'] for q in info['queries']}\n"+needle)
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
