"""Archive exact block runtime and independent original-checkpoint CPU admission."""
from pathlib import Path
wrapper=Path(__file__).with_name('archive_interleaved_bf16_body.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace("['interleaved-bf16-body-20261015']","['coalesced-body-block-20261015']")
needle='proof=dict(files={}';assert source.count(needle)==1
source=source.replace(needle,"for name in ('report.json','source.py'):\n    p=root/'logs/coalesced-body-block-admission-20261015'/name;files['evidence/'+str(p.relative_to(root))]=p\n"+needle)
exec(compile(source,__file__,'exec'))
