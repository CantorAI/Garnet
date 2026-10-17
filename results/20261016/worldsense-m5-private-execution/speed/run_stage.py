from pathlib import Path
import subprocess,sys
w=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(w/'prepare.py')],check=True)
try:
 subprocess.run([sys.executable,str(w/'controller.py'),'control-1','candidate-1'],check=True)
 subprocess.run([sys.executable,str(w/'analyze.py')],check=True)
finally:subprocess.run([sys.executable,str(w.parent/'restore_normal.py')],check=True)
