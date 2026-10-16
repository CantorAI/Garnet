"""Wait for the exact native jobs, then run a guarded isolated body component."""
from pathlib import Path
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('component',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_interleaved_bf16_body.py'")
needle="  worldsense_script=Path(";assert source.count(needle)==1
source=source.replace(needle,'''  dependencies={'wide-batch-prefill-checks-20261015':69725,'wide-batch-prefill-native-20261015':72436}
  for name,pid in dependencies.items():
   folder=root/'logs'/name;assert json.loads((folder/'pid.json').read_text())['pid']==pid
   while not (folder/'exit.json').exists():
    os.kill(pid,0)
    args=subprocess.check_output(['ps','-p',str(pid),'-o','args='],text=True)
    assert str(folder) in args and '--worker' in args
    if time.monotonic()-start>7200:raise TimeoutError('Exact native dependency still live')
    time.sleep(5)
   assert json.loads((folder/'exit.json').read_text())['exit_code']==0,name
  start=time.monotonic()
'''+needle)
exec(compile(source,__file__,'exec'))
