from pathlib import Path
import hashlib,json,subprocess,datetime,shutil
w=Path(__file__).resolve().parent;root=Path('/Users/shawn/CantorAI');source=root/'out/research/worldsense-latency-trace/source'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def cmd(*args):return subprocess.check_output(args,text=True)
manifest={str(p.relative_to(source)):sha(p)for p in sorted(source.rglob('*'))if p.is_file()and '__pycache__'not in p.parts and '.git'not in p.parts}
(w/'starting-source-manifest.json').write_text(json.dumps(manifest,indent=2));libs={}
flags={}
for mode in ['off','on']:
 b=root/('out/build/garnet-metal-worldsense-latency-trace-'+mode);libs[mode]=sha(b/'libgarnet.dylib');flags[mode]={}
 for line in (b/'CMakeCache.txt').read_text().splitlines():
  if line.startswith('GARNET_')and ':BOOL='in line:k,v=line.split(':BOOL=');flags[mode][k]=v
assert {k:(flags['off'][k],flags['on'][k])for k in flags['off']if flags['off'][k]!=flags['on'][k]}=={'GARNET_METAL_LAZY_RESIDENT_DEFAULTS':('OFF','ON'),'GARNET_METAL_RESIDENT_WORKSPACE':('OFF','ON'),'GARNET_METAL_GRAPH_WORKSPACE':('OFF','ON'),'GARNET_METAL_DENSE_PRIVATE_STORAGE':('OFF','ON'),'GARNET_METAL_VISION_WORKSPACE':('OFF','ON')}
assert flags['off']['GARNET_METAL_LATENCY_TRACE']=='ON'and flags['on']['GARNET_METAL_LATENCY_TRACE']=='ON'
(w/'build-flag-comparison.json').write_text(json.dumps(flags,indent=2))
previous=json.loads((w.parent/'worldsense-request-pool/model-assets.json').read_text());assets={}
for model,files in previous.items():
 assets[model]={}
 for rel,old in files.items():
  p=root/'out/models'/model/rel;actual=dict(bytes=p.stat().st_size,sha256=sha(p));assert actual==old,(model,rel);assets[model][rel]=actual
(w/'model-assets.json').write_text(json.dumps(assets,indent=2))
env=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),os=cmd('/usr/bin/sw_vers'),swap=cmd('/usr/sbin/sysctl','vm.swapusage'),vm_stat=cmd('/usr/bin/vm_stat'),process_rss_comm=cmd('ps','-A','-o','pid,rss,comm'),libraries=libs,source='Frozen common source; only declared deferred-storage/workspace flags differ; test-fault code exists only in separate fault-source/build',git={r:cmd('git','-C',str(root/r),'rev-parse','HEAD').strip()for r in ['Garnet','Galaxy','WorldSense','xlang3']})
(w/'environment.json').write_text(json.dumps(env,indent=2));shutil.copyfile(w.parent/'worldsense-request-pool/cases.json',w/'cases.json')
s=(w.parent/'worldsense-vision-workspace/controller.py').read_text().replace('worldsense-vision-workspace','worldsense-latency-trace');(w/'controller.py').write_text(s)
print('Frozen source, common tracing, policy flags and rehashed model assets PASS',flush=True)
