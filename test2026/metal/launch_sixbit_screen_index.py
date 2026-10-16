"""Record a separate owned nibble screen index controller, including failures before runner setup."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();tools=Path(__file__).parent
assert out.is_relative_to(root/'logs')
if '--worker' not in sys.argv[3:]:
 out.mkdir(parents=True,exist_ok=False);(out/'launcher-source.py').write_bytes(Path(__file__).read_bytes())
 with (out/'output.log').open('w') as stream:p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),'--worker'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 (out/'pid.json').write_text(json.dumps(dict(pid=p.pid))+'\n');print(json.dumps(dict(pid=p.pid,output=str(out),full_goal_met=False)));raise SystemExit(0)
start=time.monotonic();code=1;command=[sys.executable,str(tools/'run_sixbit_screen_index.py'),str(root),str(out/'jobs')]
(out/'command.json').write_text(json.dumps(dict(command=command,launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope=__doc__,full_goal_met=False),indent=2)+'\n')
try:code=subprocess.call(command)
finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start,full_goal_met=False))+'\n')
sys.exit(code)
