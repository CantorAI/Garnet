"""Preserve completed validation and pre-worker path failure before repair."""
from pathlib import Path
ns={'__file__':__file__};wrapper=Path(__file__).with_name('archive_prefill_stream_native.py').read_text();exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace("assert status['exit_code']==0;jobs[name]=status","assert status['exit_code']==(0 if name=='prefill-stream-checks-20261015' else 1);jobs[name]=status")
exec(compile(source,__file__,'exec'))
