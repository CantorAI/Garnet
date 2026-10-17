from pathlib import Path
import hashlib,json,datetime,subprocess,shlex,sys
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');n=r/'out/build/macos-arm64-release/bin';record=w/'dependencies-before-dispatch.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
if sys.argv[1]=='capture':
 assert not record.exists();paths=set();depfiles={}
 for build in [r/'out/build/garnet-metal-worldsense-prefix-cache',r/'out/build/garnet-metal-worldsense-isolated-primary-off']:
  for p in build.rglob('*.o.d'):
   text=p.read_text();depfiles[str(p)]={'sha256':sha(p),'text':text};paths.update(Path(x)for x in shlex.split(text.replace(chr(92)+chr(10),' ').split(':',1)[1]))
  paths.update([build/'CMakeCache.txt',build/'CMakeFiles/garnet_metal.dir/flags.make',build/'libgarnet.dylib'])
 assert len(depfiles)>20
 for source,name in [(Path(json.loads((w/'environment.json').read_text())['old_source']),'control-source-before-dispatch.json'),(Path(json.loads((w/'environment.json').read_text())['new_source']),'starting-source-manifest.json')]:
  m=json.loads((w/name).read_text());assert all(sha(source/k)==h for k,h in m.items());paths.update(source/k for k in m)
 for name in ['Cantor','xlang3','libxlang3_runtime.dylib','libgalaxy.dylib','libgalaxy_garnetfilter.dylib','WorldSense/start.sh','WorldSense/stop.sh']:paths.add(n/name)
 paths.update(p for p in (n/'WorldSense/Backend').rglob('*')if p.is_file()and '__pycache__'not in p.parts and p.suffix in ['.py','.yaml','.yml'])
 paths.update([w/'before-libgarnet.dylib',r/'WorldSense/tools/benchmark_recorded.py',Path('/usr/bin/pmset'),Path('/usr/bin/c++'),Path(subprocess.check_output(['xcrun','--find','clang++'],text=True).strip())])
 paths.update(p for p in w.glob('*')if p.is_file()and p.suffix in ['.py','.json','.txt']and p.name!=record.name)
 assets=json.loads((w.parent/'worldsense-isolated-primary/model-assets.json').read_text());actual={}
 for model,files in assets.items():
  actual[model]={}
  for name,info in files.items():
   p=r/'out/models'/model/name;v={'bytes':p.stat().st_size,'sha256':sha(p)};assert v==info,(model,name);actual[model][name]=v;paths.add(p)
 (w/'model-assets.json').write_text(json.dumps(actual,indent=2)+'\n');paths.add(w/'model-assets.json')
 assert all(p.is_absolute()and p.is_file()for p in paths)
 aliases={str(p):str(p.resolve())for p in sorted(paths)};files={str(p.resolve()):{'bytes':p.stat().st_size,'sha256':sha(p)}for p in sorted(paths)}
 record.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'aliases':aliases,'dependency_files':depfiles,'scope':'Two existing builds compiler-declared dependencies; both full source manifests; checkpoint assets; known host/SDK/runtime/backend/client/drivers. Current observations, not retroactive old-build environment proof. Auth helper and staged native/libgarnet excluded; exact selected libraries checked per worker. No OS dyld cache identity claim.'},indent=2)+'\n');print('Pinned source/model/build/SDK/host/driver files',len(files),flush=True)
else:
 d=json.loads(record.read_text());assert all(str(Path(k).resolve())==v for k,v in d['aliases'].items())
 for name,v in d['files'].items():
  p=Path(name);assert p.stat().st_size==v['bytes']and sha(p)==v['sha256'],name
 (w/'dependencies-after-diagnostic.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verified_files':len(d['files']),'all_aliases_and_bytes_unchanged':True},indent=2)+'\n');print('Dependencies unchanged',len(d['files']),flush=True)
