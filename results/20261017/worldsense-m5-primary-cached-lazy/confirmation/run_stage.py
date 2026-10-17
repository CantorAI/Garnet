from pathlib import Path
import subprocess,sys,hashlib,json,datetime
w=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(w/'prepare.py')],check=True)
(w/'driver-identities-before-dispatch.json').write_text(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Final generated stage drivers and request captured before dispatch; common source/model/runtime/build identities remain root preparation records','files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(w.rglob('*')) if p.is_file() and p.suffix in ['.py','.json'] and p.name!='driver-identities-before-dispatch.json'}},indent=2))
try:
 subprocess.run([sys.executable,str(w/'controller.py'),'control-full-1','candidate-full-1','candidate-full-2','control-full-2'],check=True)
 subprocess.run([sys.executable,str(w.parent/'dependency_closure.py'),'confirmation'],check=True)
 for i in [1,2]:
  subprocess.run([sys.executable,str(w/'analyze.py'),str(w),'control-full-'+str(i),'candidate-full-'+str(i)],check=True)
finally:subprocess.run([sys.executable,str(w.parent/'restore_normal.py')],check=True)
