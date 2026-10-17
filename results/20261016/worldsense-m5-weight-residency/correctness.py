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
  run(['/opt/homebrew/bin/ctest','--test-dir','/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-weight-residency-'+mode,'--output-on-failure'],'final-tests-'+mode+'.log')
 run(['python3',str(w/'resident_controller.py')],'trained-resident-controller.log')
 for mode in ['off','on']:
  run([str(n/'xlang3'),'-B',str(w/'linear_guard.py'),mode],'trained-linear-'+mode+'.log')
  run([str(n/'xlang3'),'-B',str(w/'vision_grid_guard.py'),mode],'vision-grid-'+mode+'.log')
  run([str(n/'xlang3'),'-B',str(w/'api_state_guard.py'),mode],'api-state-guard-'+mode+'.log')
 (w/'api-correctness-result.json').write_text(json.dumps({'off_and_on_real_serving_boundaries':True,'states_tokens_prefix_and_recovery':True},indent=2));print('BOTH REAL NATIVE API STATE GUARDS PASS',flush=True)
finally:
 subprocess.run(['launchctl','bootstrap','gui/501',plist],check=True)
 subprocess.run([str(n/'xlang3'),'-B',str(w/'wait_cantor.py')],check=True)
 env=dict(os.environ,WORLDSENSE_STORAGE_DIR='/Users/shawn/CantorAI/out/data/worldsense');env.pop('WORLDSENSE_ENABLE_BENCHMARK',None)
 log=(w/'restored-normal-server.log').open('ab');subprocess.Popen(['sh',str(n/'WorldSense/start.sh')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print('RESTORED VALIDATED PRODUCTION LIBRARY AND NORMAL SERVICE',flush=True)
