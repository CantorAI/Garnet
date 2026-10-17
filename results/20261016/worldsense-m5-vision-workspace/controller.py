import os,sys,time,json,hashlib,pathlib,subprocess,shutil,runpy,traceback
w=pathlib.Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-vision-workspace');root=pathlib.Path('/Users/shawn/CantorAI');native=root/'out/build/macos-arm64-release/bin';job='gui/501/com.cantorai.cantor';plist=pathlib.Path('/Users/shawn/Library/LaunchAgents/com.cantorai.cantor.plist');api=runpy.run_path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-http.py')
labels=sys.argv[1:] or ['control-1','candidate-1','candidate-2','control-2'];records=[]
def command(args,check=True):return subprocess.run(args,check=check,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
def stop():
 command(['sh',str(native/'WorldSense/stop.sh')],False)
 command(['launchctl','bootout',job],False)
 for _ in range(150):
  if command(['launchctl','print',job],False).returncode:return
  time.sleep(.1)
 raise RuntimeError('Cantor bootout did not complete')
def start(label):
 command(['launchctl','bootstrap','gui/501',str(plist)])
 log=open(w/('server-'+label+'.log'),'ab')
 subprocess.Popen(['sh',str(native/'WorldSense/start.sh')],env=dict(os.environ,WORLDSENSE_ENABLE_BENCHMARK='1',WORLDSENSE_STORAGE_DIR=str(root/'out/data/worldsense')),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(60):
  try:return api['login']()
  except Exception:time.sleep(1)
 raise RuntimeError('WorldSense did not become ready')
def stage(source):
 temp=native/'libgarnet.dylib.benchmark-new';shutil.copy2(source,temp);os.replace(temp,native/'libgarnet.dylib')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
 for label in labels:
  print('START',label,time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),flush=True);stop()
  library=root/('out/build/garnet-metal-worldsense-vision-workspace-'+('on'if label.startswith('candidate')else'off'))/'libgarnet.dylib'
  stage(library)
  state={'label':label,'library_sha256':sha(library),'serving_sha256':sha(root/'out/research/worldsense-vision-workspace/source/runtime/metal_serving.py'),'galaxy_library_sha256':sha(native/'libgalaxy_garnetfilter.dylib'),'vm_stat_before':command(['/usr/bin/vm_stat']).stdout,'swap_before':command(['/usr/sbin/sysctl','vm.swapusage']).stdout}
  token=start(label);logfile=root/'out/logs/cantor-service.log';offset=logfile.stat().st_size;errorfile=root/'out/logs/cantor-service-error.log';error_offset=errorfile.stat().st_size
  state['cantor_before']=command(['launchctl','print',job]).stdout
  env=dict(os.environ,WORLDSENSE_BENCH_TOKEN=token)
  with open(w/(label+'.client.log'),'w')as log:
   proc=subprocess.run([str(native/'xlang3'),'-B',str(root/'WorldSense/tools/benchmark_recorded.py'),'http://localhost:9981',str(w/('cases-paired-2.json'if label=='paired-2'else'cases.json')),str(w/(label+'.json')),label],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
  state['client_exit']=proc.returncode;state['vm_stat_after']=command(['/usr/bin/vm_stat']).stdout;state['swap_after']=command(['/usr/sbin/sysctl','vm.swapusage']).stdout
  with open(logfile,'rb')as log:log.seek(offset);(w/(label+'.cantor.log')).write_bytes(log.read())
  with open(errorfile,'rb')as log:log.seek(error_offset);(w/(label+'.cantor-stderr.log')).write_bytes(log.read())
  state['ps_after']=command(['ps','-A','-o','pid,rss,comm']).stdout
  records.append(state);(w/'cohort-manifest.json').write_text(json.dumps(records,indent=2))
  print('FINISH',label,'exit',proc.returncode,flush=True)
  if proc.returncode:raise RuntimeError(label+' failed; raw failure retained')
except Exception:
 traceback.print_exc();raise
finally:
 stop();stage(w/'before-libgarnet.dylib');start('restored-control');print('RESTORED ORIGINAL CONTROL AND LIVE ASSIGNMENTS',flush=True)
