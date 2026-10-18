"""Frozen original full learned proof and actual core readers/routes, no speed evidence."""
from pathlib import Path
import subprocess,sys,json,hashlib,time,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');n=r/'out/build/macos-arm64-release/bin';s=r/'out/research/worldsense-paired-gpu-profile-current/source';load=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
frozen=load(w/'learned-identities-before.json');manifest=frozen['source'];inputs=frozen['external']
def verify():
 assert len(manifest)==1723 and all(sha(s/k)==v for k,v in manifest.items())and all(sha(Path(k))==v for k,v in inputs.items())
 assert all(str(Path(k).resolve())==v['resolved']and str(Path(k).readlink())==v['target']and sha(Path(k)/'libgarnet.dylib')==v['library_sha256']for k,v in frozen['build_aliases'].items())
 assert len(frozen['legacy_source'])==1708 and all(sha(Path(frozen['legacy_root'])/k)==v for k,v in frozen['legacy_source'].items())
verify();assert load(w.parent/'worldsense-paired-gpu-profile/compatibility-native-independent-audit.json')['combined_native_CTest_count']==79;assert not(w/'learned-controller-terminal.json').exists(),'No silent unchanged retry'
journal=[]
def run(command,name):
 row=dict(command=command,log=name,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());journal.append(row);(w/'learned-controller-journal.json').write_text(json.dumps(journal,indent=2)+'\n');print('START '+name,flush=True)
 with(w/name).open('w')as f:p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=dict(__import__('os').environ,GARNET_METAL_COUNTER_TRACE='1',GARNET_METAL_COUNTER_TRACE_FILE=str(w/(name+'.counters.jsonl'))))
 row.update(returncode=p.returncode,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(w/'learned-controller-journal.json').write_text(json.dumps(journal,indent=2)+'\n')
 if p.returncode:raise RuntimeError('Learned proof failed: '+name+' exit '+str(p.returncode))
 print('PASS '+name,flush=True)
job='gui/501/com.cantorai.cantor';terminal=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),success=False,promotion=False)
try:
 subprocess.run(['sh',str(n/'WorldSense/stop.sh')],check=False);subprocess.run(['launchctl','bootout',job],check=False)
 for _ in range(200):
  if subprocess.run(['launchctl','print',job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
  time.sleep(.1)
 else:raise RuntimeError('Cantor did not unload')
 compat=r/'out/build/garnet-metal-worldsense-paired-gpu-profile-current-compat56';candidate=r/'out/build/garnet-metal-worldsense-paired-gpu-profile-current-on';selection=w.parent/'worldsense-primary-blocks/matrix-selection.json'
 for executable,build,request,output in [('primary_corpus_test',candidate,selection,'corpus-GPU-bits'),('direct_float_corpus_test',compat,w/'direct-upload-selection.json','direct-upload-corpus')]:run([str(build/executable),str(request),str(w/(output+'.json'))],output+'.log')
 run([sys.executable,str(w/'resident_controller.py')],'trained-resident-controller.log')
 for mode in ['off','on','fault','compat56']:run([str(n/'xlang3'),'-B',str(w/'linear_guard.py'),mode],'trained-linear-'+mode+'.log')
 for mode in ['off','on','compat56']:run([str(n/'xlang3'),'-B',str(w/'vision_grid_guard.py'),mode],'vision-grid-'+mode+'.log')
 for mode in ['off','primary','on','quiet','compat56']:run([str(n/'xlang3'),'-B',str(w/'api_state_guard.py'),mode],'api-state-guard-'+mode+'.log')
 for name in ['request_session_guard','request_full_session_guard']:run([str(n/'xlang3'),'-B',str(w/(name+'.py'))],name+'.log')
 for mode in ['quiet','trace','fault']:
  run([str(n/'xlang3'),'-B',str(w.parent/'worldsense-paired-gpu-profile-legacy/api_state_guard.py'),mode],'legacy-api-state-guard-'+mode+'.log')
  run([str(n/'xlang3'),'-B',str(w.parent/'worldsense-paired-gpu-profile-legacy/vision_grid_guard.py'),mode],'legacy-vision-grid-'+mode+'.log')
 for family in ['current','legacy']:run([str(n/'xlang3'),'-B',str(w.parent/('worldsense-paired-gpu-profile-'+family)/'counter_fault_model_guard.py')],family+'-counter-fault-model-recovery.log')
 verify();(w/'api-correctness-result.json').write_text(json.dumps(dict(original32VLM28text_states_tokens_vision_prefix_capacity_isolation_pass=True,original_five_full_readers_in_actual_candidate_raw_direct_pass=True,actual_core_counter_deltas_zero_all_modes=True,old_raw_compatibility_lifecycles_pass=True,source1723_and_explicit_inputs_unchanged=True,no_speed_claim=True),indent=2)+'\n');terminal['success']=True;print('WS76 full learned controller PASS; independent audit required',flush=True)
except BaseException as error:terminal['error']=repr(error);raise
finally:
 try:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True);terminal['normal_restored']=True
 finally:terminal['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(w/'learned-controller-terminal.json').write_text(json.dumps(terminal,indent=2)+'\n')
