"""Preserve streamed runs, pinned source and the earlier admission evidence."""
from pathlib import Path
wrapper=Path(__file__).with_name('archive_direct_mps_body.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source'].replace("['direct-mps-body-20261015']","['streamed-direct-mps-body-20261015']")
needle='proof=dict(files={}' ;assert source.count(needle)==1
source=source.replace(needle,"for name in ('summary.json','metadata.json'):\n    files['evidence/reference-direct-'+name]=root/'logs/direct-mps-body-20261015/jobs'/name\n"+needle)
exec(compile(source,__file__,'exec'))
