"""Archive a terminal compact eight-output probe and original CPU packing evidence."""
from pathlib import Path
p=Path(__file__).with_name('archive_coalesced_body_block.py');wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],str(p),'exec'),ns)
source=ns['source'].replace("['coalesced-body-block-20261015']","['coalesced-eight-body-20261015']")
exec(compile(source,__file__,'exec'))
