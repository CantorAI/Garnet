from pathlib import Path
import subprocess,sys
w=Path(__file__).resolve().parent
for script,log in [('final_build_and_native.py','recovery-native-controller.log'),('model_validation.py','model-validation-controller.log')]:
 with (w/log).open('w')as f:subprocess.run([sys.executable,str(w/script)],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('PASS',script,flush=True)
