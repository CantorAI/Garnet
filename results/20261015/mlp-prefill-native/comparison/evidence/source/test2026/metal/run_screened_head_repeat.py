"""Fresh-process repeat reusing byte-verified immutable screening assets."""
from pathlib import Path
source=Path(__file__).with_name('run_screened_head.py').read_text()
source=source.replace("s/'make_screened_head_harness.py'","s/'run_screened_head.py',s/'make_screened_head_harness.py'")
needle="    assets=out/(model+'-assets');run(model+'-prepare',[sys.executable,str(s/'prepare_screened_head.py'),str(root),model,str(assets)])";assert source.count(needle)==1
source=source.replace(needle,'''    assets=out/(model+'-assets');assets.mkdir()
    origin=root/'logs/screened-head-after-vision-20261015/jobs'/(model+'-assets')
    original=json.loads((origin/'manifest.json').read_text())
    for filename in ('raw.bin','approximate.bin','remaining.bin'):
        import os
        os.link(origin/filename,assets/filename)
    for query in original['queries']:
        for suffix in ('.input.bin','.reference.bin'):os.link(origin/(query['name']+suffix),assets/(query['name']+suffix))
    (assets/'manifest.json').write_bytes((origin/'manifest.json').read_bytes())
    (assets/'reuse.json').write_text(json.dumps(dict(origin=str(origin),scope='All bytes verified below before GPU use'))+'\\n')''')
exec(compile(source,__file__,'exec'))
