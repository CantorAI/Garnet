from pathlib import Path
import subprocess,json
w=Path(__file__).resolve().parent;results={}
for mode in ['off','on']:
 b='/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-private-execution-'+mode
 with (w/('native-'+mode+'.log')).open('w')as f:subprocess.run(['/opt/homebrew/bin/ctest','--test-dir',b,'--output-on-failure'],stdout=f,stderr=subprocess.STDOUT,check=True)
 with (w/('generic-private-'+mode+'.log')).open('w')as f:subprocess.run(['/Users/shawn/CantorAI/out/build/macos-arm64-release/bin/xlang3','-B',str(w/'generic_private_guard.py'),mode],stdout=f,stderr=subprocess.STDOUT,check=True)
 results[mode]=0;(w/'primitive-results.json').write_text(json.dumps(results,indent=2))
print('Both native9 suites and generic private graph guards PASS',flush=True)
