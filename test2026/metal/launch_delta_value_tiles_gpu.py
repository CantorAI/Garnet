"""Launch the isolated value-tile experiment only after the exact native cohort/controller are terminal and archived."""
from pathlib import Path
import hashlib,json,subprocess,sys
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();tools=Path(__file__).parent
assert out.is_relative_to(root/'logs') and not out.exists()
for folder,archive,expected in [
 (root/'logs/paired-norm-rope-native-continuation-rearchive-20261016','paired-norm-rope-native-continuation-rearchive-20261016.tar.gz','3ea1a512c785e20ed1f32a3676b3710f49d833b97630b220377f10f6a3c7bbee'),
 (root/'paired-norm-rope-native-investigation-20261016/logs/paired-norm-rope-native-20261016','paired-norm-rope-native-20261016.tar.gz','651a42cb20dfde3ddeb53056762d25cf146507713a3e1c9f34ecb5f3187f3ea6')]:
 pid=json.loads((folder/'pid.json').read_text())['pid'];state=subprocess.run(['/bin/ps','-p',str(pid),'-o','stat='],capture_output=True,text=True);assert state.returncode in (0,1) and (not state.stdout.strip() or state.stdout.strip().startswith('Z'))
 assert json.loads((folder/'exit.json').read_text())['exit_code']==0
 with (root/archive).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==expected
out.mkdir(parents=True)
with (out/'output.log').open('w') as stream:
 p=subprocess.Popen([sys.executable,str(tools/'run_delta_value_tiles_gpu.py'),str(root),str(out/'jobs')],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
(out/'pid.json').write_text(json.dumps(dict(pid=p.pid,runner=str(tools/'run_delta_value_tiles_gpu.py')))+'\n');print(json.dumps(dict(pid=p.pid,output=str(out),full_goal_met=False)))
