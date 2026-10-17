from pathlib import Path
import subprocess,sys
w=Path(__file__).resolve().parent
assert not (w/'cohort-manifest.json').exists()
try:
 subprocess.run([sys.executable,str(w/'controller.py'),'control-step-1','candidate-step-1','candidate-step-2','control-step-2'],check=True)
 subprocess.run([sys.executable,str(w/'analyze.py')],check=True)
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
