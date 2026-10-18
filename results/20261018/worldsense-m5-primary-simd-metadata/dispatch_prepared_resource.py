from pathlib import Path
import subprocess,sys,json,hashlib
w=Path(__file__).resolve().parent;load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert load(w/'cohort-preparation-complete.json')['all_prerequisites_checked'];identity=load(w/'driver-identities-before-dispatch.json')['files'];assert all(sha(Path(k))==v for k,v in identity.items())
try:
 subprocess.run([sys.executable,str(w/'controller.py'),'control-1','candidate-1'],check=True)
 assert all(sha(Path(k))==v for k,v in identity.items())
 subprocess.run([sys.executable,str(w/'dependency_closure.py'),'resource'],check=True)
 subprocess.run([sys.executable,str(w/'analyze_resource.py')],check=True)
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
