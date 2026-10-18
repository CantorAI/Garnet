from pathlib import Path
import subprocess,sys,hashlib,json,datetime
w=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(w/'prepare.py')],check=True)
(w/'driver-identities-before-dispatch.json').write_text(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Final generated stage drivers and request captured before dispatch; common source/model/runtime/build identities remain root preparation records','files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(w.rglob('*')) if p.is_file() and p.suffix in ['.py','.json'] and p.name!='driver-identities-before-dispatch.json'}},indent=2))
common_paths=[p for p in w.parent.rglob('*')if p.is_file()and p.suffix in ['.py','.cmake']and '__pycache__'not in p.parts]
common_paths += [Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib')]
common_paths += [Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-validated-text-worker-'+role)/'libgarnet_worker_host.dylib'for role in ['on','primary','fault']]
common={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in common_paths}
(w/'common-drivers-before-dispatch.json').write_text(json.dumps(common,indent=2)+'\n')
subprocess.run([sys.executable,str(w.parent/'dependency_closure.py'),'pre-production-speed'],check=True)
try:
 subprocess.run([sys.executable,str(w/'controller.py'),'control-1','candidate-1'],check=True)
 subprocess.run([sys.executable,str(w.parent/'dependency_closure.py'),'production-speed'],check=True)
 control=Path('/Users/shawn/CantorAI/out/research/worldsense-prefix-cache/source')
 assert all(hashlib.sha256((control/rel).read_bytes()).hexdigest()==h for rel,h in json.loads((w/'control-source-before-dispatch.json').read_text()).items())
 subprocess.run([sys.executable,str(w/'audit_native_cohort.py')],check=True)
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in common.items())
 subprocess.run([sys.executable,str(w/'analyze.py')],check=True)
finally:subprocess.run([sys.executable,str(w.parent/'restore_normal.py')],check=True)
