"""Matched same-process 8192/1024 exact-seed component ABBA on trained queries."""
from pathlib import Path
import sys
wrapper=Path(__file__).with_name('run_guarded_screened_head.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace('from make_guarded_screened_head import generate_harness as generate,generate_shader','from make_screened_seed_abba import generate_harness as generate,generate_shader')
source=source.replace("paths=[native,Path(__file__),","paths=[native,Path(__file__),s/'make_screened_seed_abba.py',s/'make_screened_seed_probe.py',s/'run_guarded_screened_head.py',")
source=source.replace('metadata=dict(strict_screen_math=True,','metadata=dict(control_seed_rows=8192,candidate_seed_rows=1024,strict_screen_math=True,')
source=source.replace("timed_operations=['exact 8192-row seed'","timed_operations=['exact 8192-row control versus exact 1024-row candidate seed'")
needle="    assert len(checks)==17";assert source.count(needle)==1
source=source.replace(needle,"    controls=[v for v in records if v['kind']=='control_check'];assert len(controls)==17 and {v['query'] for v in controls}=={q['name'] for q in info['queries']} and all(v['all_values_checked'] and v['greedy_exact'] for v in controls)\n"+needle)
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
