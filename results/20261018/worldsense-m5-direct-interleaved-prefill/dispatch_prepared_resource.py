from pathlib import Path
import subprocess,sys,json,hashlib
w=Path(__file__).resolve().parent
assert json.loads((w/'cohort-preparation-complete.json').read_text())['all_prerequisites_checked']
identity=json.loads((w/'driver-identities-before-dispatch.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in identity['files'].items())
try:
 subprocess.run([sys.executable,str(w/'controller.py'),'control-1','candidate-1'],check=True)
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in identity['files'].items())
 subprocess.run([sys.executable,str(w/'dependency_closure.py'),'resource'],check=True)
 subprocess.run([sys.executable,str(w/'analyze_resource.py')],check=True)
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
