"""Preserve direct-row generated context and pinned original dependencies."""
from pathlib import Path
source=Path(__file__).with_name('archive_padded_mps_body.py').read_text().replace("['padded-mps-body-20261015']","['direct-mps-body-20261015']")
needle='proof=dict(files={}' ;assert source.count(needle)==1
source=source.replace(needle,"pinned=root/'screened-head-staging/manifest.json';manifest=json.loads(pinned.read_text())\nfiles['evidence/source-manifest.json']=pinned\nfor name,sha in manifest.items():\n    p=root/'source'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha;files['evidence/source/'+name]=p\n"+needle)
exec(compile(source,__file__,'exec'))
