from pathlib import Path
import hashlib,json,subprocess,datetime,shutil
w=Path(__file__).resolve().parent;root=Path('/Users/shawn/CantorAI');source=root/'out/research/worldsense-cooperative-prefill/source'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def cmd(*args):return subprocess.check_output(args,text=True)
manifest={str(p.relative_to(source)):sha(p)for p in sorted(source.rglob('*'))if p.is_file()and '__pycache__'not in p.parts and '.git'not in p.parts}
assert manifest==json.loads((w/'correctness-source-manifest.json').read_text()),'Source changed since correctness'
(w/'starting-source-manifest.json').write_text(json.dumps(manifest,indent=2));libs={}
flags={}
for mode in ['off','primary','on']:
 b=root/('out/build/garnet-metal-worldsense-cooperative-prefill-'+mode);libs[mode]=sha(b/'libgarnet.dylib');flags[mode]={}
 for line in (b/'CMakeCache.txt').read_text().splitlines():
  if line.startswith('GARNET_')and ':BOOL='in line:k,v=line.split(':BOOL=');flags[mode][k]=v
assert {k:(flags['off'][k],flags['on'][k])for k in flags['off']if flags['off'][k]!=flags['on'][k]}=={'GARNET_METAL_PRIMARY_BF16_BLOCKS':('OFF','ON'),'GARNET_METAL_SOLE_INTERLEAVED_MLP':('OFF','ON'),'GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL':('OFF','ON'),'GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE':('OFF','ON'),'GARNET_METAL_SOLE_COOPERATIVE_PREFILL':('OFF','ON'),'GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY':('OFF','ON'),'GARNET_METAL_WIDE_PRIMARY_STORAGE':('OFF','ON'),'GARNET_METAL_PRIMARY_VECTOR_PREFILL':('OFF','ON'),'GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY':('OFF','ON'),'GARNET_METAL_DIRECT_FLOAT_WEIGHT_UPLOAD':('OFF','ON'),'GARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY':('OFF','ON'),'GARNET_METAL_LAZY_RESIDENT_DEFAULTS':('OFF','ON'),'GARNET_METAL_CACHED_MATRIX_METADATA':('OFF','ON')}
assert all(flags[m][cap]==('OFF'if m=='off'else'ON')for m in ['off','primary','on']for cap in ['GARNET_METAL_PRIMARY_BF16_BLOCKS','GARNET_METAL_WIDE_PRIMARY_STORAGE','GARNET_METAL_LAZY_RESIDENT_DEFAULTS'])
assert all(flags[m]['GARNET_METAL_CACHED_MATRIX_METADATA']==('ON'if m=='on'else'OFF')for m in ['off','primary','on'])
assert {k:(flags['primary'][k],flags['on'][k])for k in flags['on']if flags['primary'][k]!=flags['on'][k]}=={k:('OFF','ON')for k in ['GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_DIRECT_FLOAT_WEIGHT_UPLOAD','GARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY','GARNET_METAL_CACHED_MATRIX_METADATA','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY','GARNET_METAL_SOLE_INTERLEAVED_MLP','GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL','GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE','GARNET_METAL_SOLE_COOPERATIVE_PREFILL']}

(w/'build-flag-comparison.json').write_text(json.dumps(flags,indent=2))
previous=json.loads((w.parent/'worldsense-request-pool/model-assets.json').read_text());assets={}
for model,files in previous.items():
 assets[model]={}
 for rel,old in files.items():
  p=root/'out/models'/model/rel;actual=dict(bytes=p.stat().st_size,sha256=sha(p));assert actual==old,(model,rel);assets[model][rel]=actual
(w/'model-assets.json').write_text(json.dumps(assets,indent=2))
env=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),os=cmd('/usr/bin/sw_vers'),swap=cmd('/usr/sbin/sysctl','vm.swapusage'),vm_stat=cmd('/usr/bin/vm_stat'),process_rss_comm=cmd('ps','-A','-o','pid,rss,comm'),libraries=libs,source='WS67 cooperative padded sole prefill. RAWcandidate13flags/PRIMARYcandidate10flags. New explicit integer/sharedlayout normalization preserves old FP32 arithmetic/order; graph/runtime/decode unchanged66, native report-only extension. Faults only separate correctness build',git={r:cmd('git','-C',str(root/r),'rev-parse','HEAD').strip()for r in ['Garnet','Galaxy','WorldSense','xlang3']})
(w/'environment.json').write_text(json.dumps(env,indent=2));shutil.copyfile(w.parent/'worldsense-request-pool/cases.json',w/'cases.json')
print('Frozen WS67 source/thirteenflagRAWcontrast/tenflagPRIMARYcontrast and checkpoint identities PASS',flush=True)
