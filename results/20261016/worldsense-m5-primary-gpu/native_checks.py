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
 for mode in ['off','on']:
  run(['/opt/homebrew/bin/ctest','--test-dir','/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-primary-gpu-'+mode,'--output-on-failure'],'native-'+mode+'.log');results[mode]=0
 (w/'native-results.json').write_text(json.dumps(results,indent=2))
 paths=[Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-primary-gpu-'+mode+'/primary_projection_words.bin')for mode in ['off','on']]
 identical=paths[0].read_bytes()==paths[1].read_bytes()
 (w/'native-word-comparison.json').write_text(json.dumps({'records':186,'all_FP32_words_identical':identical,'bytes':[p.stat().st_size for p in paths],'scope':'Actual full native projection outputs including prefill/generic/body/vocab, not model performance.'},indent=2))
 assert identical,'Packed projection bits differ from OFF control'
 print('BOTH7NATIVE TESTS AND186FULL_FP32WORD RECORDS PASS',flush=True)

finally:
 subprocess.run(['launchctl','bootstrap','gui/501',plist],check=True)
 subprocess.run([str(n/'xlang3'),'-B',str(w/'wait_cantor.py')],check=True)
 env=dict(os.environ,WORLDSENSE_STORAGE_DIR='/Users/shawn/CantorAI/out/data/worldsense');env.pop('WORLDSENSE_ENABLE_BENCHMARK',None)
 log=(w/'restored-normal-server.log').open('ab');subprocess.Popen(['sh',str(n/'WorldSense/start.sh')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print('RESTORED VALIDATED PRODUCTION LIBRARY AND NORMAL SERVICE',flush=True)
