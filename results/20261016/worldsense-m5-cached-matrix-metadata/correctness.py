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
 results={}
 for mode in ['off','on','fault']:
  run(['/opt/homebrew/bin/ctest','--test-dir','/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-cached-matrix-metadata-'+mode,'--output-on-failure'],'final-tests-'+mode+'.log');results[mode]=0
 (w/'native-results.json').write_text(json.dumps(results,indent=2));print('ALL NATIVE REFERENCE TESTS PASS',flush=True)
 run(['python3',str(w/'resident_controller.py')],'trained-resident-controller.log')
 for mode in ['off','on','fault']:
  run([str(n/'xlang3'),'-B',str(w/'linear_guard.py'),mode],'trained-linear-'+mode+'.log')
 for fixture in ['fixture-1-8','fixture-2-64']:
  run([str(n/'xlang3'),'-B',str(w/'resident_fault_guard.py'),str(w/fixture/'fixture.json'),'/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-cached-matrix-metadata-fault','/Users/shawn/CantorAI/out/research/worldsense-cached-matrix-metadata/source'],'fault-resident-'+fixture+'.log')
 (w/'fault-correctness-result.json').write_text(json.dumps({'partial_allocation_failure_exact_recovery':True,'native_metadata_and_4and6_input_tests_pass':True},indent=2))
 for mode in ['off','on']:
  run([str(n/'xlang3'),'-B',str(w/'vision_grid_guard.py'),mode],'vision-grid-'+mode+'.log')
  run([str(n/'xlang3'),'-B',str(w/'api_state_guard.py'),mode],'api-state-guard-'+mode+'.log')
 (w/'api-correctness-result.json').write_text(json.dumps({'off_and_on_real_serving_boundaries':True,'states_tokens_prefix_and_recovery':True,'four_input_six_input_vision_and_faults_pass':True},indent=2));print('BOTH REAL NATIVE API STATE GUARDS PASS',flush=True)

finally:
 subprocess.run(['launchctl','bootstrap','gui/501',plist],check=True)
 subprocess.run([str(n/'xlang3'),'-B',str(w/'wait_cantor.py')],check=True)
 env=dict(os.environ,WORLDSENSE_STORAGE_DIR='/Users/shawn/CantorAI/out/data/worldsense');env.pop('WORLDSENSE_ENABLE_BENCHMARK',None)
 log=(w/'restored-normal-server.log').open('ab');subprocess.Popen(['sh',str(n/'WorldSense/start.sh')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print('RESTORED VALIDATED PRODUCTION LIBRARY AND NORMAL SERVICE',flush=True)
