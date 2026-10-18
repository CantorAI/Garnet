"""Pin build-declared inputs/host files before timing; verify after workers finish."""
from pathlib import Path
import datetime, hashlib, json, shlex, subprocess, sys

w=Path(__file__).resolve().parent
r=Path('/Users/shawn/CantorAI')
source=r/'out/research/worldsense-gpu-counter-profile/source'
native=r/'out/build/macos-arm64-release/bin'
record=w/'build-host-dependencies-before-dispatch.json'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def dependencies(text):
    return shlex.split(text.replace('\\\n',' ').split(':',1)[1])
def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

if sys.argv[1]=='capture':
    assert not record.exists(), 'Do not overwrite an existing timing preparation'
    paths=set();raw={}
    for mode in ['off','primary','on','quiet','fault']:
        b=r/('out/build/garnet-metal-worldsense-gpu-counter-profile-'+mode)
        for p in sorted(b.rglob('*.o.d')):
            text=p.read_text();raw[str(p)]={'sha256':sha(p),'text':text}
            paths.update(Path(name)for name in dependencies(text))
    assert len(raw)>=30
    paths.update(p for p in (w/'build-profile').glob('*.cmake')if p.is_file())
    paths.update([w/'direct_float_corpus_test.cpp',w/'direct-upload-selection.json',r/'out/build/garnet-metal-worldsense-gpu-counter-profile-on/direct_float_corpus_test'])
    cmd=['/usr/bin/c++','-std=c++17','-I',str(source/'src/backends/metal'),'-I',str(source/'src/model'),'-I',str(source/'third_party'),'-M','-MT','primary_corpus',str(w/'primary_corpus_test.cpp')]
    text=subprocess.check_output(cmd,text=True)
    (w/'corpus-preprocessor-dependencies.txt').write_text(text)
    paths.update(Path(name)for name in dependencies(text))
    for name in ['Cantor','xlang3','libxlang3_runtime.dylib','libgalaxy.dylib','libgalaxy_garnetfilter.dylib','WorldSense/start.sh','WorldSense/stop.sh']:
        p=native/name
        if p.is_file():paths.add(p)
    backend=native/'WorldSense/Backend'
    paths.update(p for p in backend.rglob('*')if p.is_file()and '__pycache__'not in p.parts and p.suffix in ['.py','.yaml','.yml'])
    compiler=Path(subprocess.check_output(['xcrun','--find','clang++'],text=True).strip());paths.add(compiler)
    assert all(p.is_absolute()and p.is_file()for p in paths)
    aliases={str(p):str(p.resolve())for p in sorted(paths)}
    files={str(p.resolve()):{'bytes':p.stat().st_size,'sha256':sha(p)}for p in sorted(paths)}
    assert len(files)>800
    record.write_text(json.dumps({'observed_utc':now(),'scope':'Union of compiler-generated dependency files for four builds, corpus preprocessing dependencies, compiler and known native host/backend sources. Core Garnet source, checkpoint assets, caller cases and SDK/runtime IDs also have separate manifests. Not a claim to hash every OS dyld-cache page. Staged libgarnet intentionally excluded here: each worker separately verifies its selected library SHA.','preprocessor_command':cmd,'dependency_files':raw,'aliases':aliases,'files':files},indent=2))
    print('Pinned build/SDK/host dependency files',len(files),'before timing',flush=True)
else:
    data=json.loads(record.read_text())
    assert all(str(Path(name).resolve())==target for name,target in data['aliases'].items()),'Dependency alias changed'
    for name,info in data['files'].items():
        p=Path(name);assert p.stat().st_size==info['bytes']and sha(p)==info['sha256'],name
    (w/('dependencies-after-'+sys.argv[1]+'.json')).write_text(json.dumps({'observed_utc':now(),'verified_files':len(data['files']),'all_aliases_and_file_bytes_unchanged':True,'scope':'After completed timing workers; no inference or GPU operation.'},indent=2))
    print('Build/SDK/host dependency identity unchanged',len(data['files']),sys.argv[1],flush=True)
