from pathlib import Path
import subprocess,sys
w=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(w/'prepare_cohort.py')],check=True)
try:
 subprocess.run([sys.executable,str(w/'controller.py'),'control-1','candidate-1'],check=True)
 subprocess.run([sys.executable,str(w/'dependency_closure.py'),'resource'],check=True)
 subprocess.run([sys.executable,str(w/'analyze_resource.py')],check=True)
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
