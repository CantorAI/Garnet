"""Observe actual 1024-row seeds through reset, fallback, poison, close and rebuild."""
from pathlib import Path
wrapper=Path(__file__).with_name('run_screened_head_session_reference.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
for name in ('session','plain'):
    for count in (0,1):source=source.replace(name+'.screened_head_dispatches=='+str(count),name+'.screened_head_dispatches=='+str(count)+' and '+name+'.screened_head_seed_rows=='+str(1024 if count else 0))
source=source.replace("assert 'screened_head' in garnet.backend_info()","assert 'screened_head' in garnet.backend_info() and 'screen_seed_1024' in garnet.backend_info()")
source=source.replace('actual_screened_head=True','actual_screened_head=True,actual_seed_rows=1024')
exec(compile(source,__file__,'exec'))
