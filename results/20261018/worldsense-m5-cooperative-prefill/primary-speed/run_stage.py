from pathlib import Path
import subprocess,sys,hashlib,json,datetime
w=Path(__file__).resolve().parent
assert (w/'execution-plan.json').exists(), 'External preparation must complete before dispatch'
(w/'driver-identities-before-dispatch.json').write_text(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Final generated stage drivers and request captured before dispatch; common source/model/runtime/build identities remain root preparation records','files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(w.rglob('*')) if p.is_file() and p.suffix in ['.py','.json'] and p.name!='driver-identities-before-dispatch.json'}},indent=2))
try:
 subprocess.run([sys.executable,str(w/'controller.py'),'control-1','candidate-1'],check=True)
 subprocess.run([sys.executable,str(w.parent/'dependency_closure.py'),'primary-speed'],check=True)
 identity=json.loads((w/'driver-identities-before-dispatch.json').read_text());assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in identity['files'].items())
 assert hashlib.sha256(Path('/usr/bin/pmset').read_bytes()).hexdigest()==json.loads((w/'observation-tool-identity.json').read_text())['pmset_sha256']
 subprocess.run([sys.executable,str(w/'analyze.py')],check=True)
finally:subprocess.run([sys.executable,str(w.parent/'restore_normal.py')],check=True)
