"""Durable, exclusive validation job for compact joined-half storage."""
import json, subprocess, sys, time
from pathlib import Path

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
mode = 'profiles' if '--profiles' in sys.argv else 'checks'
extra = ['--confirm'] if '--confirm' in sys.argv else []
if '--worker' not in sys.argv:
    out.mkdir(parents=True, exist_ok=False)
    with (out/'output.log').open('w') as stream:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(root), str(out), '--worker', *(['--profiles'] if mode=='profiles' else []), *extra],
                                   stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    (out/'pid.json').write_text(json.dumps({'pid': process.pid}))
    print(json.dumps({'pid': process.pid, 'output': str(out)}))
else:
    begin = time.monotonic()
    code = 1
    try:
        if mode=='profiles':
            assert json.loads((root/'logs/compact-joined-checks-20261013/exit.json').read_text())['exit_code']==0
        script = 'run_compact_joined_profiles.py' if mode=='profiles' else 'run_compact_joined_checks.py'
        code = subprocess.call([sys.executable, str(root/'source/test2026/metal'/script), str(root), str(out/'jobs'), *extra])
    finally:
        (out/'exit.json').write_text(json.dumps({'exit_code': code, 'wall_seconds': time.monotonic()-begin}))
    sys.exit(code)
