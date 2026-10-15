"""Archive terminal GEMV compatibility/component jobs and canonical matrix proofs."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
r=Path(sys.argv[1]).resolve();destination=Path(sys.argv[2]);files={};proof={}
folders=sorted((r/'logs').glob('mps-gemv-*'));assert folders
for folder in folders:
    assert (folder/'exit.json').is_file(),'Live GEMV job: '+str(folder)
    for path in folder.rglob('*'):
        if path.is_file() and path.suffix in ('.json','.log','.txt','.py','.mm','.metal'):
            files['evidence/logs/'+str(path.relative_to(r/'logs'))]=path
    for manifest_path in (folder/'jobs').glob('*-assets/manifest.json'):
        data=json.loads(manifest_path.read_text());checkpoint=Path(data['checkpoint']).resolve();assert checkpoint.is_relative_to((r/'models').resolve())
        with checkpoint.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==data['checkpoint_sha256']
        for item in data['cases']:
            path=manifest_path.parent/(item['name']+'.bin');raw=path.read_bytes()
            assert len(raw)==item['columns']*item['inner']*2 and hashlib.sha256(raw).hexdigest()==item['sha256']
            proof[str(path.relative_to(r))]=dict(bytes=len(raw),sha256=item['sha256'],checkpoint_sha256=data['checkpoint_sha256'])
for name in ('mps_gemv_compat.mm','run_mps_gemv_compat.py','make_mps_gemv_probe.py','run_mps_gemv_probe.py','archive_mps_gemv.py','background_comparison.py'):
    files['evidence/source/'+name]=r/'source/test2026/metal'/name
manifest={}
with tarfile.open(destination,'w:gz') as archive:
    for name,path in sorted(files.items()):
        raw=path.read_bytes();manifest[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    for name,data in [('evidence/files.json',manifest),('evidence/omitted-matrix-proof.json',proof)]:
        raw=(json.dumps(data,indent=2)+'\n').encode();info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(manifest),omitted_matrices=len(proof),archive_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())))
