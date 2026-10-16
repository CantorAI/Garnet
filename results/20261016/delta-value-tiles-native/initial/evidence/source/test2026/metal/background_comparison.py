"""Launch a durable comparison command; retain its exact command and exit status."""
import json,os,subprocess,sys
from pathlib import Path

mode,folder,*command=sys.argv[1:];folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True)
if mode=='launch':
    if (folder/'pid').exists() and not (folder/'exit.json').exists():
        try:os.kill(int((folder/'pid').read_text()),0)
        except ProcessLookupError:pass
        else:raise RuntimeError('Existing investigation job is still running')
    (folder/'exit.json').unlink(missing_ok=True)
    with (folder/'output.log').open('w') as log:
        process=subprocess.Popen([sys.executable,__file__,'run',str(folder),*command],stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (folder/'pid').write_text(str(process.pid));print('Started investigation PID',process.pid)
elif mode=='run':
    status={'command':command,'exit_code':1}
    try:status['exit_code']=subprocess.run(command).returncode
    finally:(folder/'exit.json').write_text(json.dumps(status,indent=2)+'\n')
    raise SystemExit(status['exit_code'])
else:raise ValueError('Expected launch or run')
