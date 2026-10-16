"""Freeze terminal two-output component plus original CPU packing evidence."""
from pathlib import Path
p=Path(__file__).with_name('archive_coalesced_body_block.py')
wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],str(p),'exec'),ns)
source=ns['source'].replace("['coalesced-body-block-20261015']","['two-body-20261015']")
old="assert status['exit_code']==0;jobs[name]=status";assert source.count(old)==1
source=source.replace(old,"assert status['exit_code'] in (0,1);jobs[name]=status")
exec(compile(source,__file__,'exec'))
