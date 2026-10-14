"""Archive terminal reset experiments and explicit sources; omit model arrays."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
r=Path(sys.argv[1]).resolve();destination=Path(sys.argv[2]);files={}
for folder in sorted((r/'logs').glob('direct-state-zero-*')):
    if folder.name=='direct-state-zero-original-source-20261013':
        for p in folder.rglob('*'):
            if p.is_file():files['evidence/original/'+str(p.relative_to(folder))]=p
        continue
    assert (folder/'exit.json').is_file(),'Live/nonterminal job: '+str(folder)
    for p in folder.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.txt','.log','.py','.cpp','.h','.mm','.cmake'):
            files['evidence/logs/'+str(p.relative_to(r/'logs'))]=p
manifest=json.loads((r/'work/direct-state-zero-manifest.json').read_text())
for name in manifest:
    path=r/'source'/name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest[name]['sha256'],name
    files['evidence/source/'+name]=path
for name in ('run_joined_prefill_checks.py','compare_joined_prefill.py','compare_vllm_metal.py',
             'background_comparison.py','run_matched_frame_process.py','test_resident_kv_reference.py',
             'run_resident_kv_reference.py','archive_direct_state_zero.py'):
    files['evidence/source/test2026/metal/'+name]=r/'source/test2026/metal'/name
files['evidence/source-manifest.json']=r/'work/direct-state-zero-manifest.json'
files['evidence/installer.py']=r/'work/install_direct_state_zero.py'
metadata=json.loads((r/'logs/direct-state-zero-checks-20261013/jobs/metadata.json').read_text())
for status in json.loads((r/'logs/direct-state-zero-checks-20261013/jobs/statuses.json').read_text()):
    for argument in status['command']:
        path=Path(argument).resolve()
        if path.is_file() and path.is_relative_to((r/'source').resolve()):
            files['evidence/check-source/'+str(path.relative_to(r/'source'))]=path
for name in ('test2026/metal/reference_dense.py','test2026/pi5b/reference_decoder.py',
             'test2026/metal/test_qwen35_extended_prefill_reference.py'):
    files['evidence/check-source/'+name]=r/'source'/name
for name,expected in metadata['source_sha256'].items():
    path=r/'source'/name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
    files['evidence/check-source/'+name]=path
for target in ('garnet','garnet_metal'):
    files['evidence/build/'+target+'-flags.make']=r/'build-direct-state-zero/CMakeFiles'/(target+'.dir')/'flags.make'
hashes={}
with tarfile.open(destination,'w:gz') as archive:
    for name,path in sorted(files.items()):
        data=path.read_bytes();hashes[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
    raw=(json.dumps(hashes,indent=2)+'\n').encode();info=tarfile.TarInfo('evidence/files.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(hashes),archive_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())))
