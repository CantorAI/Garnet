from pathlib import Path
import subprocess,sys
w=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(w/'prepare.py')],check=True)
try:
 subprocess.run([sys.executable,str(w/'controller.py'),'control-full-1','candidate-full-1','candidate-full-2','control-full-2'],check=True)
 for i in [1,2]:
  subprocess.run([sys.executable,str(w/'analyze.py'),str(w),'control-full-'+str(i),'candidate-full-'+str(i)],check=True)
finally:subprocess.run([sys.executable,str(w.parent/'restore_normal.py')],check=True)
