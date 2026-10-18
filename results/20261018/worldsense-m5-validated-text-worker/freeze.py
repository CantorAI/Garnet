from pathlib import Path
import hashlib,json,subprocess,datetime,shutil
w=Path(__file__).resolve().parent;root=Path('/Users/shawn/CantorAI');source=root/'out/research/worldsense-validated-text-worker/source'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def cmd(*args):return subprocess.check_output(args,text=True)
manifest={str(p.relative_to(source)):sha(p)for p in sorted(source.rglob('*'))if p.is_file()and '__pycache__'not in p.parts and '.git'not in p.parts}
assert manifest==json.loads((w/'gateway-source-manifest.json').read_text()),'Source changed since correctness'
(w/'starting-source-manifest.json').write_text(json.dumps(manifest,indent=2));libs={}
flags={}
for mode in ['off','primary','on']:
 b=root/('out/build/garnet-metal-worldsense-validated-text-worker-'+mode);libs[mode]=sha(b/'libgarnet.dylib');flags[mode]={}
 for line in (b/'CMakeCache.txt').read_text().splitlines():
  if line.startswith('GARNET_')and ':BOOL='in line:k,v=line.split(':BOOL=');flags[mode][k]=v
assert {k:(flags['off'][k],flags['on'][k])for k in flags['off']if flags['off'][k]!=flags['on'][k]}=={'GARNET_METAL_PRIMARY_BF16_BLOCKS':('OFF','ON'),'GARNET_METAL_NATIVE_TEXT_WORKER':('OFF','ON'),'GARNET_METAL_NATIVE_TEXT_WORKER_VALIDATED_RAW':('OFF','ON'),'GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY':('OFF','ON'),'GARNET_METAL_WIDE_PRIMARY_STORAGE':('OFF','ON'),'GARNET_METAL_PRIMARY_VECTOR_PREFILL':('OFF','ON'),'GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY':('OFF','ON'),'GARNET_METAL_DIRECT_FLOAT_WEIGHT_UPLOAD':('OFF','ON'),'GARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY':('OFF','ON'),'GARNET_METAL_LAZY_RESIDENT_DEFAULTS':('OFF','ON'),'GARNET_METAL_CACHED_MATRIX_METADATA':('OFF','ON')}
assert all(flags[m][cap]==('OFF'if m=='off'else'ON')for m in ['off','primary','on']for cap in ['GARNET_METAL_PRIMARY_BF16_BLOCKS','GARNET_METAL_WIDE_PRIMARY_STORAGE','GARNET_METAL_LAZY_RESIDENT_DEFAULTS','GARNET_METAL_NATIVE_TEXT_WORKER','GARNET_METAL_NATIVE_TEXT_WORKER_VALIDATED_RAW'])
assert all(flags[m]['GARNET_METAL_CACHED_MATRIX_METADATA']==('ON'if m=='on'else'OFF')for m in ['off','primary','on'])
assert {k:(flags['primary'][k],flags['on'][k])for k in flags['on']if flags['primary'][k]!=flags['on'][k]}=={k:('OFF','ON')for k in ['GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_DIRECT_FLOAT_WEIGHT_UPLOAD','GARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY','GARNET_METAL_CACHED_MATRIX_METADATA','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY']}

(w/'build-flag-comparison.json').write_text(json.dumps(flags,indent=2))
previous=json.loads((w.parent/'worldsense-request-pool/model-assets.json').read_text());assets={}
for model,files in previous.items():
 assets[model]={}
 for rel,old in files.items():
  p=root/'out/models'/model/rel;actual=dict(bytes=p.stat().st_size,sha256=sha(p));assert actual==old,(model,rel);assets[model][rel]=actual
(w/'model-assets.json').write_text(json.dumps(assets,indent=2))
env=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),os=cmd('/usr/bin/sw_vers'),swap=cmd('/usr/sbin/sysctl','vm.swapusage'),vm_stat=cmd('/usr/bin/vm_stat'),process_rss_comm=cmd('ps','-A','-o','pid,rss,comm'),libraries=libs,source='WS61 packedVLM/nativeRAWtext process isolation; rawcandidate11flags/storagecontrolcandidate6. Pinned original WS10 RAW child with separate native instrumentation included in all candidate process-set memory. Faults only in correctness build.',git={r:cmd('git','-C',str(root/r),'rev-parse','HEAD').strip()for r in ['Garnet','Galaxy','WorldSense','xlang3']})
old_source=root/'out/research/worldsense-prefix-cache/source'
legacy=json.loads((w/'validated-source-manifest.json').read_text())
assert len(legacy)==1706 and all(sha(old_source/k)==v for k,v in legacy.items())
old_library=root/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib'
assert sha(old_library)=='dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72'
env['legacy_backend']={'library':str(old_library),'sha256':sha(old_library),'source':str(old_source),'source_manifest_sha256':sha(w/'validated-source-manifest.json')}
env['native_hosts']={role:sha(root/('out/build/garnet-metal-worldsense-validated-text-worker-'+role)/'libgarnet_worker_host.dylib')for role in ['on','primary','fault']}
(w/'environment.json').write_text(json.dumps(env,indent=2));shutil.copyfile(w.parent/'worldsense-request-pool/cases.json',w/'cases.json')
print('Frozen WS61 source,11flagcontrast,5baseflags/nativeRAWchild and checkpoint identities PASS',flush=True)
