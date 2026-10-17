from pathlib import Path
import subprocess,sys
w=Path(__file__).resolve().parent
for name in ['admission_checks.py','correctness.py']:
 with (w/(name+'.controller.log')).open('w')as stream:subprocess.run([sys.executable,str(w/name)],stdout=stream,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
