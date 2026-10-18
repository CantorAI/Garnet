from pathlib import Path
import json,hashlib,shutil,datetime
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=root/'work/worldsense-native-text-worker';dest=w/'failed-resource-integration-56157';dest.mkdir(exist_ok=False);s=Path('/Users/shawn/CantorAI/out/research/worldsense-native-text-worker/source');manifest=json.loads((w/'gateway-source-manifest.json').read_text());seed=json.loads((w.parent/'worldsense-isolated-primary/correctness-source-manifest.json').read_text());assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
for k,v in manifest.items():
 if seed.get(k)!=v:p=dest/'source-overlay'/k;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s/k,p)
for p in w.iterdir():
 if p.is_file()and p.suffix in ['.py','.json','.log','.txt','.metal','.cpp']:shutil.copy2(p,dest/p.name)
for folder in ['speed','primary-speed','production-speed','confirmation','build-profile']:
 for p in (w/folder).glob('*'):
  if p.is_file()and p.suffix in ['.py','.cmake']:
   q=dest/p.relative_to(w);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
for p in [root/'work/worldsense-opt'/name for name in ['dispatch_prepared_resource_ws58.py','ws58_native_route.py','prepare_resource_ws58.py','audit_routes_ws58.py','ws58_admission.py','ws58_owner_parent.py','ws58_owner_death.py','ws58_post_owner_recovery.py','ws58_postproof.py','reproof_ws58.py']]:
 q=dest/'orchestrators'/p.name;q.parent.mkdir(exist_ok=True);shutil.copy2(p,q)
for mode in ['off','on','primary','fault']:
 p=Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-native-text-worker-'+mode)/'libgarnet.dylib';shutil.copy2(p,dest/('library-'+mode+'.dylib'))
(dest/'failure-classification.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller_session=56157,terminal_exit=1,control='valid2frames/4modelcalls/123.0684367500071s HTTP wall',candidate='invalid0frames/38.302846167003736s HTTP wall',error='XLangStream IPC marshal does not support this value yet',classification='WorldSense/Galaxy integration failure; no complete resource/performance comparison, not a valid numerical threshold rejection',suspected_boundary='Galaxy sends default-constructed invalid X::Value for absent text image. Existing local InferJson maps non-string/non-bin to empty path; remote branch forwards raw invalid value to IPC. Native script guards used empty strings and missed this C++ boundary.',status='Hypothesis pending native SDK reproducer; no source edit yet',restored='ValidatedWS10/fourfiles/noCam/benchmark404 PID5248'),indent=2)+'\n')
(dest/'SHA256.json').write_text(json.dumps({str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(dest.rglob('*'))if p.is_file()},indent=2)+'\n');print('Preserved failed actual WorldSense cohort/source/4libraries/consumed drivers and prerequisites')
