"""Observe actual interleaved dispatches through trained session recovery paths."""
from pathlib import Path
wrapper=Path(__file__).with_name('run_small_seed_session_reference.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
for name in ('session','plain'):
    for count in (0,1):source=source.replace(name+'.screened_head_dispatches=='+str(count),name+'.screened_head_dispatches=='+str(count)+' and '+name+'.interleaved_screen_dispatches=='+str(count))
source=source.replace("and 'screen_seed_1024' in garnet.backend_info()","and 'screen_seed_1024' in garnet.backend_info() and 'interleaved_screen_index' in garnet.backend_info()")
source=source.replace('actual_seed_rows=1024','actual_seed_rows=1024,actual_interleaved=True')
exec(compile(source,__file__,'exec'))
