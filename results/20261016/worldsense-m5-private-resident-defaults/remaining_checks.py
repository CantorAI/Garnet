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
 run(['python3',str(w/'private_trained_controller.py')],'private-trained-controller.log')
 for mode in ['off','on']:
  run([str(n/'xlang3'),'-B',str(w/'graph_guard.py'),mode],'graph-'+mode+'.log')
  run([str(n/'xlang3'),'-B',str(w/'generic_private_guard.py'),mode],'generic-private-'+mode+'.log')
  for fixture in ['fixture-1-8','fixture-2-64']:
   run([str(n/'xlang3'),'-B',str(w/'actual_policy_guard.py'),str(w/fixture/'fixture.json'),'/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-private-resident-defaults-'+mode,'/Users/shawn/CantorAI/out/research/worldsense-private-resident-defaults/source'],'policy-'+mode+'-'+fixture+'.log')
 run(['/opt/homebrew/bin/ctest','--test-dir','/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-private-resident-defaults-fault','--output-on-failure'],'tests-fault.log')
 for script in ['ordinary_fault_guard.py','weight_fault_guard.py']:
  run([str(n/'xlang3'),'-B',str(w/script)],script+'.log')
 run([str(n/'xlang3'),'-B',str(w/'generic_private_guard.py'),'fault'],'generic-private-fault.log')
 for mode in ['off','on','fault']:
  run([str(n/'xlang3'),'-B',str(w/'linear_deferred_guard.py'),mode],'linear-deferred-'+mode+'.log')
 for fixture in ['fixture-1-8','fixture-2-64']:
  run([str(n/'xlang3'),'-B',str(w/'actual_policy_guard.py'),str(w/fixture/'fixture.json'),'/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-private-resident-defaults-fault','/Users/shawn/CantorAI/out/research/worldsense-private-resident-defaults/fault-source','fault'],'policy-fault-'+fixture+'.log')
 (w/'fault-correctness-result.json').write_text(json.dumps({'test_only_build_native12':True,'ordinary_and_weight_growth_abort':True,'private_graph_and_session_poison_reset_exact_recovery':True},indent=2))
 (w/'private-trained-complete.json').write_text(json.dumps({'private4_and6_off_on':True,'not_speed_evidence':True},indent=2));print('DEDICATED PRIVATE TRAINED CHECKS PASS',flush=True)
finally:
 subprocess.run(['launchctl','bootstrap','gui/501',plist],check=True)
 subprocess.run([str(n/'xlang3'),'-B',str(w/'wait_cantor.py')],check=True)
 env=dict(os.environ,WORLDSENSE_STORAGE_DIR='/Users/shawn/CantorAI/out/data/worldsense');env.pop('WORLDSENSE_ENABLE_BENCHMARK',None)
 log=(w/'restored-normal-server.log').open('ab');subprocess.Popen(['sh',str(n/'WorldSense/start.sh')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print('RESTORED VALIDATED PRODUCTION LIBRARY AND NORMAL SERVICE',flush=True)
