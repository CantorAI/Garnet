from pathlib import Path
import subprocess,json
w=Path(__file__).resolve().parent;result={}
for mode in ['off','on']:
 b='/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-request-residency-'+mode
 with (w/('final-configure-'+mode+'.log')).open('w')as f:subprocess.run(['/opt/homebrew/bin/cmake',b],cwd=b,stdout=f,stderr=subprocess.STDOUT,check=True)
 with (w/('final-build-'+mode+'.log')).open('w')as f:subprocess.run(['/opt/homebrew/bin/cmake','--build',b,'-j','2'],stdout=f,stderr=subprocess.STDOUT,check=True)
 with (w/('native-'+mode+'.log')).open('w')as f:p=subprocess.run(['/opt/homebrew/bin/ctest','--test-dir',b,'--output-on-failure'],stdout=f,stderr=subprocess.STDOUT)
 result[mode]=p.returncode;(w/'native-results.json').write_text(json.dumps(result,indent=2));assert p.returncode==0,mode+' native test failed;original failure retained'
 print('BUILD+NATIVE PASS',mode,flush=True)
