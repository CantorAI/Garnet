"""Archive a terminal compact eight-output probe and original CPU packing evidence."""
from pathlib import Path
import sys
p=Path(__file__).with_name('archive_coalesced_body_block.py');wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],str(p),'exec'),ns)
trial=sys.argv[3] if len(sys.argv)>3 else 'coalesced-eight-body-20261015'
assert trial in ('coalesced-eight-body-20261015','coalesced-eight-body-unrolled-20261015','coalesced-eight-body-unrolled-fixed-20261015')
source=ns['source'].replace("['coalesced-body-block-20261015']",repr([trial]))
needle="assert status['exit_code']==0;jobs[name]=status"
assert source.count(needle)==1
source=source.replace(needle,"assert status['exit_code'] in (0,1);jobs[name]=status")
exec(compile(source,__file__,'exec'))
