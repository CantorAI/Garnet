"""Orchestrate native identity/memory/watchdog guards with no model inference."""
from pathlib import Path
import subprocess,json,hashlib,os,time,sys,uuid,datetime,signal
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');n=r/'out/build/macos-arm64-release/bin';s=r/'out/research/worldsense-prefix-cache/source'
proof=json.loads((w/'structural-RAW-counter-source-proof.json').read_text());assert proof['logical_primary_and_lazy_domains_absent']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((w/'feasibility-identities-before.json').read_text())['old_source'];assert all(sha(s/k)==v for k,v in old.items())
paths=[p for p in w.rglob('*')if p.is_file()and p.suffix in ['.py','.cpp','.h','.mm','.dylib']]
paths += [r/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib',n/'xlang3',n/'libxlang3_runtime.dylib']
identities={str(p):sha(p)for p in paths};(w/'native-host-guard-identities-before.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=identities),indent=2)+'\n')
normal=w/'native-host-normal.json';assert not normal.exists();nonce=uuid.uuid4().hex;env=dict(os.environ,GARNET_NATIVE_WORKER_PARENT=str(os.getpid()))
with(w/'native-host-normal.log').open('w')as log:
 p=subprocess.run([str(n/'xlang3'),'-B',str(w/'native_host_guard.py'),'normal',str(normal),nonce,str(os.getpid())],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=30)
assert p.returncode==0,'Native host guard failed; inspect retained log'
data=json.loads(normal.read_text());assert data['nonce']==nonce and data['model_calls']==0
folder=w/'native-host-parent-death';folder.mkdir(exist_ok=False);nonce=uuid.uuid4().hex
with(folder/'owner.log').open('w')as log:
 owner=subprocess.Popen([sys.executable,str(w/'owned_probe_parent.py'),str(folder),nonce],stdout=log,stderr=subprocess.STDOUT)
 childpid=None;birth=None
 def state(pid):
  p=subprocess.run(['/bin/ps','-p',str(pid),'-o','lstart=','-o','stat='],capture_output=True,text=True)
  return p.stdout.strip()if p.returncode==0 else None
 try:
  for _ in range(200):
   assert owner.poll()is None,'Owned launcher failed before readiness'
   if(folder/'parent-ready.json').exists():break
   time.sleep(.05)
  else:raise RuntimeError('Owner readiness timeout')
  parent=json.loads((folder/'parent-ready.json').read_text());ready=json.loads((folder/'native-ready.json').read_text())
  childpid=parent['child_pid'];assert parent['parent_pid']==owner.pid and parent['nonce']==nonce and ready['nonce']==nonce and ready['pid']==childpid and ready['parent_pid']==owner.pid
  birth=state(childpid);assert birth is not None
  (folder/'observed-owned-identity.json').write_text(json.dumps(dict(parent=parent,native_pid=childpid,ps_start_and_state=birth,nonce=nonce),indent=2)+'\n')
  (folder/'exit.request').write_text('exit\n');assert owner.wait(timeout=5)==0
  gone=False
  for _ in range(100):
   current=state(childpid)
   if current is None or current!=birth:
    # ps state can change while the same process lives. Require absence,
    # a different start identity, or zombie state (already terminated).
    if current is None or current[:24]!=birth[:24]or current.split()[-1].startswith('Z'):gone=True;break
   time.sleep(.05)
  assert gone,'Native watchdog did not terminate owned orphan'
  (folder/'result.json').write_text(json.dumps(dict(owned_parent_terminal_exit=0,native_child_terminated_after_parent_exit=True,child_exit_code_unobserved=True,no_model_inference=True,scope='Birth-bound owned no-model child while native VM sleeps; not active model inference lifecycle proof.'),indent=2)+'\n')
 finally:
  if owner.poll()is None:owner.terminate();owner.wait(timeout=5)
  if childpid and birth:
   current=state(childpid)
   if current and current[:24]==birth[:24]and not current.split()[-1].startswith('Z'):os.kill(childpid,signal.SIGTERM)
assert all(sha(Path(p))==v for p,v in identities.items())and all(sha(s/k)==v for k,v in old.items())
(w/'native-host-guards-result.json').write_text(json.dumps(dict(native_identity_symbol_ABI_SHA_unique_image_guards_pass=True,unloaded_backend_and_wrong_parent_rejected=True,ten_actual_common_clock_containment_checks=True,real_Mach_and_Metal_process_snapshots_pass=True,explicit_structural_RAW_logical_zero_contract_pass=True,owned_native_watchdog_parent_death_no_model_pass=True,source_library_runtime_drivers_unchanged=True,no_model_or_performance_admission=True),indent=2)+'\n')
print('Native sidecar identity/clock/real memory/RAW-domain/owned watchdog guards PASS; no model/speed admission',flush=True)
