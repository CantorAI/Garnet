from pathlib import Path
import hashlib,json,subprocess,datetime,shutil
w=Path(__file__).resolve().parent;root=Path('/Users/shawn/CantorAI');source=root/'out/research/worldsense-packed-request-residency/source'
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
 b=root/('out/build/garnet-metal-worldsense-packed-request-residency-'+mode);libs[mode]=sha(b/'libgarnet.dylib');flags[mode]={}
 for line in (b/'CMakeCache.txt').read_text().splitlines():
  if line.startswith('GARNET_')and ':BOOL='in line:k,v=line.split(':BOOL=');flags[mode][k]=v
assert {k:(flags['off'][k],flags['on'][k])for k in flags['off']if flags['off'][k]!=flags['on'][k]}=={k:('OFF','ON')for k in ['GARNET_METAL_PRIMARY_BF16_BLOCKS','GARNET_METAL_WIDE_PRIMARY_STORAGE','GARNET_METAL_LAZY_RESIDENT_DEFAULTS','GARNET_METAL_PACKED_REQUEST_WEIGHT_RESIDENCY']}
assert {k:(flags['primary'][k],flags['on'][k])for k in flags['on']if flags['primary'][k]!=flags['on'][k]}=={'GARNET_METAL_PACKED_REQUEST_WEIGHT_RESIDENCY':('OFF','ON')}
for role in ['off','primary','on']:
 assert all(flags[role][key]=='OFF'for key in ['GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY','GARNET_METAL_CACHED_MATRIX_METADATA','GARNET_METAL_DIRECT_FLOAT_WEIGHT_UPLOAD','GARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY','GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY','GARNET_METAL_REQUEST_RESIDENCY_TEST_FAULTS','GARNET_METAL_PACKED_REQUEST_RESIDENCY_TEST_FAULTS'])
(w/'build-flag-comparison.json').write_text(json.dumps(flags,indent=2))
previous=json.loads((w.parent/'worldsense-request-pool/model-assets.json').read_text());assets={}
for model,files in previous.items():
 assets[model]={}
 for rel,old in files.items():
  p=root/'out/models'/model/rel;actual=dict(bytes=p.stat().st_size,sha256=sha(p));assert actual==old,(model,rel);assets[model][rel]=actual
assert json.loads((w/'model-assets.json').read_text())==assets  # Preserve frozen input bytes
env=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),os=cmd('/usr/bin/sw_vers'),swap=cmd('/usr/sbin/sysctl','vm.swapusage'),vm_stat=cmd('/usr/bin/vm_stat'),process_rss_comm=cmd('ps','-A','-o','pid,rss,comm'),libraries=libs,source='WS69 original PRIMARY/WIDE/LAZY plus bounded packed payload/metadata request hints. RAW4flag and matchedPRIMARY1flag contrast. Shaders/math/weights unchanged; automatic raw handling and no extra companion native calls. Allocations/hints do not prove physical residency or speed.',git={r:cmd('git','-C',str(root/r),'rev-parse','HEAD').strip()for r in ['Garnet','Galaxy','WorldSense','xlang3']})
(w/'environment.json').write_text(json.dumps(env,indent=2));resource_cases=json.loads((w.parent/'worldsense-request-pool/cases.json').read_text());resource_cases['repeats']=1;assert json.loads((w/'cases.json').read_text())==resource_cases  # Preserve frozen input bytes
print('WS69 source/fourflagRAWcontrast/oneflagPRIMARYcontrast/checkpoint identities frozen',flush=True)
