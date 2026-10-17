from pathlib import Path
import subprocess,os,json,traceback,time
w=Path(__file__).resolve().parent;n=Path('/Users/shawn/CantorAI/out/build/macos-arm64-release/bin');job='gui/501/com.cantorai.cantor';plist='/Users/shawn/Library/LaunchAgents/com.cantorai.cantor.plist'
def run(args,log):
 with (w/log).open('w')as stream:subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT,check=True)
subprocess.run(['sh',str(n/'WorldSense/stop.sh')],check=False)
subprocess.run(['launchctl','bootout',job],check=False)
for _ in range(200):
 if subprocess.run(['launchctl','print',job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
 time.sleep(.1)
else:raise RuntimeError('Service unload did not complete')
try:
 for mode in ['off','on']:
  for fixture in ['fixture-1-8','fixture-2-64']:
   run([str(n/'xlang3'),'-B',str(w/'actual_policy_guard.py'),str(w/fixture/'fixture.json'),'/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-private-execution-'+mode,'/Users/shawn/CantorAI/out/research/worldsense-private-execution/source'],'policy-'+mode+'-'+fixture+'.log')
 (w/'actual-policy-results.json').write_text(json.dumps({'off_on_both_fixtures':True,'exact_states_predictions_and_actual_route':True},indent=2))
 print('ALL ACTUAL POLICY GUARDS PASS',flush=True)
finally:
 subprocess.run(['launchctl','bootstrap','gui/501',plist],check=True)
 env=dict(os.environ,WORLDSENSE_STORAGE_DIR='/Users/shawn/CantorAI/out/data/worldsense');env.pop('WORLDSENSE_ENABLE_BENCHMARK',None)
 log=(w/'restored-normal-server.log').open('ab');subprocess.Popen(['sh',str(n/'WorldSense/start.sh')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print('RESTORED VALIDATED PRODUCTION LIBRARY AND NORMAL SERVICE',flush=True)
