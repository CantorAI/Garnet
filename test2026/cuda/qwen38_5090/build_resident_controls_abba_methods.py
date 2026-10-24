"""Recreate raw ABBA method members from reviewed base scripts; no GPU work."""
from pathlib import Path
import argparse,hashlib,json,zipfile

def build(output):
 source=Path(__file__).parent;plan=json.loads((source/'resident_controls_abba_plan.json').read_text());files={}
 bases=['run-native-resident-controls-http-server.py','run-native-resident-controls-http-workflow.py','run-native-resident-controls-http-candidate.py','serve-native-resident-controls-worker.bat','stop-native-resident-controls-http-worker.py','archive-native-resident-controls-http.py']
 for leg in plan['order']:
  prefix=leg['prefix'];enabled=leg['resident_controls'];name=leg['leg']
  for base in bases:
   repo_name=base.replace('-','_');s=(source/repo_name).read_text(encoding='utf8')
   s=s.replace('native-resident-controls',prefix).replace('shared-resident-controls-http-diagnostic','shared-controls-abba-'+name+'-http-diagnostic')
   if 'worker.bat' in base:s=s.replace('GARNET_STUDY_RESIDENT_CONTROLS=1','GARNET_STUDY_RESIDENT_CONTROLS='+str(int(enabled)))
   if not enabled:s=s.replace("get('resident_request_controls') is True","get('resident_request_controls') is False").replace("get('resident_request_controls') is not True","get('resident_request_controls') is not False").replace('existing resident_controls=True; NVFP4 selectorOFF','existing resident_controls=False; NVFP4 selectorOFF')
   files[base.replace('native-resident-controls',prefix)]=s.encode('utf8')
 files['controls-abba-experiment-plan.json']=(source/'resident_controls_abba_plan.json').read_bytes()
 assert not output.exists(),'Preserve an existing package'
 with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
  for n,b in files.items():z.writestr(n,b)
  z.writestr('methods-sha256.json',json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in files.items()},indent=2))
 return {n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();print(json.dumps(build(a.output),indent=2))
