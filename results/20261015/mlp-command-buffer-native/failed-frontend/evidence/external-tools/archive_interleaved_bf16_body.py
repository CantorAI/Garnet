"""Archive the terminal isolated layout component, source and checkpoint identities."""
from pathlib import Path
source=Path(__file__).with_name('archive_padded_mps_body.py').read_text().replace("['padded-mps-body-20261015']","['interleaved-bf16-body-20261015']")
needle='proof=dict(files={}';assert source.count(needle)==1
source=source.replace(needle,'''for trial in jobs:
    meta=json.loads((root/'logs'/trial/'jobs/metadata.json').read_text())
    for model,info in meta['assets'].items():
        p=root/'models'/model/'model.safetensors';assert digest_checkpoint(p)==info['checkpoint_sha256']
        omitted[str(p.relative_to(root))]=dict(bytes=p.stat().st_size,sha256=info['checkpoint_sha256'])
        folder=root/'logs/decoder-codec-gpu-20261013/jobs'/(model+'-assets')
        p=folder/'manifest.json';files['evidence/'+str(p.relative_to(root))]=p
        for case in info['cases']:
            p=folder/case['name']/'raw.bin';sha=digest_checkpoint(p);assert sha==case['sha256']['raw.bin']
            omitted[str(p.relative_to(root))]=dict(bytes=p.stat().st_size,sha256=sha)
'''+needle)
needle='root=Path(sys.argv[1])';assert source.count(needle)==1
source=source.replace(needle,"def digest_checkpoint(p):\n    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()\n"+needle)
exec(compile(source,__file__,'exec'))
