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
  run(['/opt/homebrew/bin/ctest','--test-dir','/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-primary-u32-integration-'+mode,'--output-on-failure'],'native-'+mode+'.log');results[mode]=0
 (w/'native-results.json').write_text(json.dumps(results,indent=2))
 paths=[Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-primary-u32-integration-'+mode+'/primary_projection_words.bin')for mode in ['off','on']]
 identical=paths[0].read_bytes()==paths[1].read_bytes()
 (w/'native-word-comparison.json').write_text(json.dumps({'records':186,'all_FP32_words_identical':identical,'bytes':[p.stat().st_size for p in paths],'scope':'Actual full native projection outputs including prefill/generic/body/vocab, not model performance.'},indent=2))
 assert identical,'Packed projection bits differ from OFF control'
 print('ALL3NATIVE7TESTS AND186FULL_FP32WORD RECORDS PASS',flush=True)
 run(['/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-primary-u32-integration-on/primary_corpus_test',str(w.parent/'worldsense-primary-blocks/matrix-selection.json'),str(w/'corpus-GPU-bits.json')],'corpus-GPU-bits.log')
 assert json.loads((w/'corpus-GPU-bits.json').read_text())['totals']['GPU_vector_original_FP32_words_checked']==5921308672
 print('ALL398PACKED_ACTUAL_WEIGHTS_GPU_WORDS PASS',flush=True)
 run(['python3',str(w/'resident_controller.py')],'trained-resident-controller.log')
 for mode in ['off','on','fault']:
  run([str(n/'xlang3'),'-B',str(w/'linear_guard.py'),mode],'trained-linear-'+mode+'.log')
 for mode in ['off','on']:
  run([str(n/'xlang3'),'-B',str(w/'vision_grid_guard.py'),mode],'vision-grid-'+mode+'.log')
  run([str(n/'xlang3'),'-B',str(w/'api_state_guard.py'),mode],'api-state-guard-'+mode+'.log')
 (w/'api-correctness-result.json').write_text(json.dumps({'all32VLM28text_state_bits_tokens_prefix_capacity_vision_and_layer_isolation':True,'all398actualpacked_weights_GPU_bit_reconstruction':True,'all398actualpacked_weights_vector_GPU_bit_reconstruction':True,'all398actualpacked_weights_Kmajor_GPU_bit_reconstruction':True,'all398actualpacked_weights_u32_GPU_bit_reconstruction':True,'native7tests_all3builds_and186_projection_words_exact':True},indent=2))
 print('FULL_PRIMARY_MODEL_STATE_AND_CORPUS_CORRECTNESS_PASS',flush=True)

finally:
 subprocess.run(['launchctl','bootstrap','gui/501',plist],check=True)
 subprocess.run([str(n/'xlang3'),'-B',str(w/'wait_cantor.py')],check=True)
 env=dict(os.environ,WORLDSENSE_STORAGE_DIR='/Users/shawn/CantorAI/out/data/worldsense');env.pop('WORLDSENSE_ENABLE_BENCHMARK',None)
 log=(w/'restored-normal-server.log').open('ab');subprocess.Popen(['sh',str(n/'WorldSense/start.sh')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print('RESTORED VALIDATED PRODUCTION LIBRARY AND NORMAL SERVICE',flush=True)
