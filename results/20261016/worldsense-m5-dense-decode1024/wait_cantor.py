"""Wait for launchd-owned Cantor before package launcher; setup only."""
import sys,time,subprocess,re
sys.path.insert(0,'/Users/shawn/CantorAI/out/build/macos-arm64-release/bin/WorldSense/Backend/src')
from cantorproxy import ProbeCantor
for attempt in range(300):
 status=subprocess.run(['launchctl','print','gui/501/com.cantorai.cantor'],capture_output=True,text=True)
 match=re.search(r'\bpid = (\d+)',status.stdout)
 existing=ProbeCantor()
 if match and existing and int(match.group(1))==int(existing[0]):
  print('Launchd Cantor LRPC ready',existing[0],flush=True);break
 time.sleep(.1)
else:raise RuntimeError('Launchd-owned Cantor LRPC did not become ready')
