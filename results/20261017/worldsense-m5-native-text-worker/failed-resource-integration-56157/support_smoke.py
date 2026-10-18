"""Native support/bootstrap APIs only; no model load/inference."""
import sys,os,time,json
from pathlib import Path
w=Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-native-text-worker');source='/Users/shawn/CantorAI/out/research/worldsense-native-text-worker/source';build='/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-native-text-worker-off';sys.path.insert(0,source);sys.path.insert(0,build)
import garnet
mode=sys.argv[1];before=time.monotonic();info=json.loads(garnet.native_worker_support_json());after=time.monotonic();assert info['pid']==os.getpid()and info['parent_pid']==os.getppid()and before-.0001<=info['steady_seconds']<=after+.0001;assert Path(info['library_path']).resolve()==Path(build+'/libgarnet.dylib').resolve();assert info['source_root']==source and not info['watchdog_active']
rejected=False
try:garnet.native_worker_watch_parent(os.getpid())
except Exception as error:rejected='ownership mismatch'in str(error)
assert rejected and not json.loads(garnet.native_worker_support_json())['watchdog_active']
if mode=='unmarked':
 rejected=False
 try:garnet.native_worker_watch_parent(os.getppid())
 except Exception as error:rejected='marker missing'in str(error)
 assert rejected and not json.loads(garnet.native_worker_support_json())['watchdog_active']
else:
 assert mode=='owned'and garnet.native_worker_watch_parent(os.getppid())and garnet.native_worker_watch_parent(os.getppid());ready=json.loads(garnet.native_worker_support_json());assert ready['watchdog_active']and ready['watch_parent_pid']==os.getppid()
(w/('support-smoke-'+mode+'.json')).write_text(json.dumps(dict(native_API_library_identity_clock_parent_guards=True,mode=mode,watchdog_start_idempotent_and_normal_exit_expected=mode=='owned',model_or_gateway_test=False),indent=2)+'\n');print('Native support identity/clock/ownership/marker guard PASS '+mode,flush=True)
