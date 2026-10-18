"""Freeze compiler/SDK/host closure before capture, verify only outside workers."""
from pathlib import Path
import datetime,hashlib,json,shlex,subprocess,sys
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');n=r/'out/build/macos-arm64-release/bin';record=w/'capture/build-host-dependencies-before-dispatch.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def dependencies(text):return shlex.split(text.replace('\\\n',' ').split(':',1)[1])
if sys.argv[1]=='capture':
 assert not record.exists()
 roles=json.loads((w/'protocol.json').read_text())['roles']+list(json.loads((w/'compatibility-proof-protocol.json').read_text())['roles'])
 paths=set();raw={};recipes={}
 for role in roles:
  root=r/('out/build/garnet-metal-worldsense-paired-gpu-profile-'+role)
  for p in sorted(root.rglob('*.o.d')):
   text=p.read_text();raw[str(p)]=dict(sha256=sha(p),text=text);paths.update(Path(v)for v in dependencies(text))
  paths.add(root/'CMakeCache.txt')
 assert len(raw)>=60
 proof=b/'work/worldsense-paired-gpu-profile-current'
 for name in ['corpus-build-identity.json','direct-corpus-build-identity.json']:
  recipe=json.loads((proof/name).read_text());cmd=recipe['command'];source=next(arg for arg in cmd if arg.endswith('_corpus_test.cpp'));index=cmd.index(source)
  pre=[cmd[0]]+cmd[1:index]+['-M','-MT',name.replace('.json',''),source]
  text=subprocess.check_output(pre,text=True);(w/'capture'/(name+'.dependencies.txt')).write_text(text)
  paths.update(Path(v)for v in dependencies(text));recipes[name]=dict(original_command=cmd,preprocessor_command=pre,dependencies=text)
 paths.update((w/'build-profile').glob('*.cmake'))
 for name in ['Cantor','xlang3','libxlang3_runtime.dylib','libgalaxy.dylib','libgalaxy_garnetfilter.dylib','WorldSense/start.sh','WorldSense/stop.sh']:
  p=n/name
  if p.is_file():paths.add(p)
 backend=n/'WorldSense/Backend';paths.update(p for p in backend.rglob('*')if p.is_file()and '__pycache__'not in p.parts and p.suffix in ['.py','.yaml','.yml'])
 paths.add(Path(subprocess.check_output(['xcrun','--find','clang++'],text=True).strip()));paths.add(Path('/usr/bin/c++'))
 assert all(p.is_absolute()and p.is_file()and p.name!='worldsense-http.py'for p in paths)
 aliases={str(p):str(p.resolve())for p in sorted(paths)};files={str(p.resolve()):dict(bytes=p.stat().st_size,sha256=sha(p))for p in sorted(paths)}
 assert len(files)>1000
 record.write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),aliases=aliases,files=files,compiler_dependency_records=raw,corpus_dependency_recipes=recipes,scope='Union of all ten actual CMake-generated compiler dependency files, both exact corpus recipe header closures, compiler/shim and known native host/backend files. Sources/models/drivers have separate complete manifests. Mutable staged libgarnet deliberately excluded; exact worker library hash checked before launch. OS dyld-cache pages are not claimed.'),indent=2)+'\n')
 print('Pinned actual build/SDK/host dependency files',len(files),flush=True)
else:
 data=json.loads(record.read_text());assert all(str(Path(k).resolve())==v for k,v in data['aliases'].items())
 for name,info in data['files'].items():
  p=Path(name);assert p.stat().st_size==info['bytes']and sha(p)==info['sha256'],name
 (w/'capture'/('dependencies-after-'+sys.argv[1]+'.json')).write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),verified_files=len(data['files']),all_aliases_and_file_bytes_unchanged=True,no_inference=True),indent=2)+'\n')
 print('Dependency identities unchanged',len(data['files']),sys.argv[1],flush=True)
