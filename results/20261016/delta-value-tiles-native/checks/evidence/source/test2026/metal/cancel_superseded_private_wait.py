"""Cancel only the owned preflight waiter, never a benchmark or user process."""
import json,os,shlex,signal,subprocess,time
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation').resolve();out=root/'logs/private-bf16-confirm-clean-20261014'
info=json.loads((out/'pid.json').read_text());assert info['pid']==4319
state=json.loads((out/'preflight/wait.json').read_text())
assert state['benchmark_started'] is False and state['quiet_seconds']<1 and not (out/'jobs').exists()
command=subprocess.check_output(['/bin/ps','-p','4319','-o','args='],text=True).strip();parts=shlex.split(command)
assert parts[1:]==[str(root/'source/test2026/metal/launch_private_bf16_quiet.py'),str(root),str(out),'--worker']
os.kill(4319,signal.SIGTERM)
for _ in range(30):
 result=subprocess.run(['/bin/ps','-p','4319','-o','comm='],capture_output=True,text=True)
 if result.returncode or not result.stdout.strip():break
 time.sleep(.1)
else:raise RuntimeError('Owned preflight waiter did not terminate')
(out/'exit.json').write_text(json.dumps(dict(exit_code=-15,reason='Superseded after exact CPU-service fingerprint audit; no benchmark started')))
print('Canceled owned preflight waiter 4319; benchmark never started')
