from pathlib import Path
import json,subprocess,sys,datetime
w=Path(__file__).resolve().parent;protocol=json.loads((w/'protocol.json').read_text());assert protocol['experiment']=='WS-M5-55'and protocol['kind']=='diagnostic_only'and protocol['promote']is False
assert not any((w/(x+'.json')).exists()for x in protocol['schedule'])
subprocess.run([sys.executable,str(w/'dependencies.py'),'capture'],check=True)
try:
 subprocess.run([sys.executable,'-u',str(w/'controller.py'),*protocol['schedule']],check=True)
 subprocess.run([sys.executable,str(w/'dependencies.py'),'verify'],check=True)
 for labels in [('control-1','candidate-1'),('control-2','candidate-2')]:subprocess.run([sys.executable,str(w/'analyze.py'),*labels],check=True)
 (w/'diagnostic-controller-complete.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'four_fresh_workers_completed':True,'forty_original_outputs_native_exact':True,'promote':False,'diagnostic_only':True},indent=2)+'\n')
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
