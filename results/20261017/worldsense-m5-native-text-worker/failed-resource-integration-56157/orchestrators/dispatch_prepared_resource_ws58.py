"""Dispatch exactly the prepared first resource cohort; no speed promotion."""
from pathlib import Path
import subprocess,sys,json,hashlib,datetime
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=root/'work/worldsense-native-text-worker';prepared=json.loads((w/'cohort-preparation-complete.json').read_text());assert prepared['all_prerequisites_checked']and not prepared['worker_dispatched'];assert not(w/'cohort-manifest.json').exists(),'Never repeat an existing resource cohort unchanged'
identities=json.loads((w/'driver-identities-before-dispatch.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in identities['files'].items())
identities['files'][str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();(w/'driver-identities-before-dispatch.json').write_text(json.dumps(identities,indent=2)+'\n')
subprocess.run([sys.executable,str(w/'dependency_closure.py'),'pre-resource'],check=True)
(w/'resource-launch.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),planned_labels=['control-1','candidate-1'],planned_model_calls=8,preparation_sha256=hashlib.sha256((w/'cohort-preparation-complete.json').read_bytes()).hexdigest(),scope='First original resource gate, no speed/promotion claim.'),indent=2)+'\n')
try:
 subprocess.run([sys.executable,str(w/'controller.py'),'control-1','candidate-1'],check=True)
 subprocess.run([sys.executable,str(w/'dependency_closure.py'),'resource'],check=True)
 subprocess.run([sys.executable,str(w/'audit_native_cohort.py')],check=True)
 subprocess.run([sys.executable,str(w/'analyze_resource.py')],check=True)
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
