from pathlib import Path
import hashlib,json,shutil,datetime,subprocess,ast
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-physical-memory/source';n=r/'out/build/macos-arm64-release/bin'
assert json.loads((w/'correctness-result.json').read_text())['trained_both_fixtures_off_on_global_inventory_read_only_dedup_and_types']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(n/'libgarnet.dylib')==sha(r/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib');shutil.copy2(n/'libgarnet.dylib',w/'before-libgarnet.dylib')
controller=(w.parent/'worldsense-cached-matrix-metadata/controller.py').read_text().replace('worldsense-cached-matrix-metadata','worldsense-physical-memory')
controller=controller.replace('import os,sys,time,json,hashlib,pathlib,subprocess,shutil,runpy,traceback','import os,sys,time,json,hashlib,pathlib,subprocess,shutil,runpy,traceback,re\nfrom memory_observer import Observer')
needle="  env=dict(os.environ,WORLDSENSE_BENCH_TOKEN=token)";assert needle in controller;controller=controller.replace(needle,"  pid=int(re.search(r'^\\s*pid = (\\d+)',state['cantor_before'],re.M).group(1));observer=Observer(w,label,pid);observer.start()\n"+needle)
a=controller.index("  with open(w/(label+'.client.log')");b=controller.index("  state['client_exit']=",a);block=controller[a:b];controller=controller[:a]+"  try:\n"+''.join(' '+line+'\n'for line in block.splitlines())+"  finally:state['memory_observer_records']=len(observer.finish())\n"+controller[b:]
(w/'controller.py').write_text(controller)
for p in w.glob('*.py'):ast.parse(p.read_text())
flags={};libs={}
for mode in ['off','on']:
 b=r/('out/build/garnet-metal-worldsense-physical-memory-'+mode);libs[mode]=sha(b/'libgarnet.dylib');flags[mode]={}
 for line in (b/'CMakeCache.txt').read_text().splitlines():
  if line.startswith('GARNET_')and ':BOOL='in line:k,v=line.split(':BOOL=');flags[mode][k]=v
 assert flags[mode]['GARNET_METAL_MULTI_INVENTORY']=='ON'
 for rel,dest in [('CMakeCache.txt',mode+'-CMakeCache.txt'),('Testing/Temporary/LastTest.log',mode+'-LastTest.log'),('CMakeFiles/garnet_metal.dir/flags.make',mode+'-flags.make')]:shutil.copy2(b/rel,w/dest)
assert {k:(flags['off'][k],flags['on'][k])for k in flags['off']if flags['off'][k]!=flags['on'][k]}=={'GARNET_METAL_LAZY_RESIDENT_DEFAULTS':('OFF','ON'),'GARNET_METAL_CACHED_MATRIX_METADATA':('OFF','ON')}
(w/'build-flags.json').write_text(json.dumps(flags,indent=2))
manifest={str(p.relative_to(s)):sha(p)for p in sorted(s.rglob('*'))if p.is_file() and '__pycache__'not in p.parts};(w/'starting-source-manifest.json').write_text(json.dumps(manifest,indent=2))
previous=json.loads((w.parent/'worldsense-cached-matrix-metadata/model-assets.json').read_text());assets={}
for model,files in previous.items():
 assets[model]={}
 for rel,old in files.items():
  p=r/'out/models'/model/rel;actual={'bytes':p.stat().st_size,'sha256':sha(p)};assert actual==old;assets[model][rel]=actual
(w/'model-assets.json').write_text(json.dumps(assets,indent=2))
(w/'environment.json').write_text(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'libraries':libs,'os':subprocess.check_output(['/usr/bin/sw_vers'],text=True),'source':'Frozen common diagnostic source, only two old policy flags differ. All observed timings excluded.','tools_and_drivers':{str(p):sha(p)for p in list(w.glob('*.py'))+[Path('/usr/bin/vmmap'),Path('/usr/bin/footprint'),n/'xlang3',n/'libxlang3_runtime.dylib',n/'libgalaxy_garnetfilter.dylib',r/'WorldSense/tools/benchmark_recorded.py']},'request_sha256':sha(w/'cases.json'),'git':{name:subprocess.check_output(['git','-C',str(r/name),'rev-parse','HEAD'],text=True).strip()for name in ['Garnet','Galaxy','WorldSense','xlang3']}},indent=2));print('Frozen diagnostic source/libs/models/request/observer/tools and two policy differences; no worker dispatched',flush=True)

sdk=r/'xlang3/sdk'
(w/'sdk-identities.json').write_text(json.dumps({str(p.relative_to(sdk)):sha(p)for p in sorted((sdk/'xlang3').rglob('*'))if p.is_file()},indent=2))
