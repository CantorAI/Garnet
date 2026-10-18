"""Complete learned corpus and native model proof only, never a speed result."""
from pathlib import Path
import subprocess,sys,json,hashlib,time
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');n=r/'out/build/macos-arm64-release/bin';s=r/'out/research/worldsense-native-text-worker/source';job='gui/501/com.cantorai.cantor'
assert set(json.loads((w/'native-results.json').read_text()))=={'off','primary','on','fault'}
assert all(v==0 for v in json.loads((w/'native-results.json').read_text()).values())
assert json.loads((w/'native-word-comparison.json').read_text())['all_four_builds_FP32_words_identical']
program=json.loads((w/'actual-program-proof.json').read_text());assert program['all_four_actual_main_and_prefill_programs_byte_exact_WS54']
manifest=json.loads((w/'gateway-source-manifest.json').read_text());assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==h for k,h in manifest.items())
(w/'full-correctness-driver-identities.json').write_text(json.dumps(dict(utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),files={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(w.iterdir())if p.is_file()and(p.suffix in ['.py','.cpp']or p.name in ['gateway-source-manifest.json','direct-upload-selection.json'])}),indent=2)+'\n')
def run(args,log):
 with(w/log).open('w')as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run(['sh',str(n/'WorldSense/stop.sh')],check=False);subprocess.run(['launchctl','bootout',job],check=False)
for _ in range(200):
 if subprocess.run(['launchctl','print',job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
 time.sleep(.1)
else:raise RuntimeError('Service unload did not complete')
try:
 run([str(r/'out/build/garnet-metal-worldsense-native-text-worker-on/primary_corpus_test'),str(w.parent/'worldsense-primary-blocks/matrix-selection.json'),str(w/'corpus-GPU-bits.json')],'corpus-GPU-bits.log')
 c=json.loads((w/'corpus-GPU-bits.json').read_text())['totals'];assert c['packed_matrices']==201 and c['raw_small_matrices']==245
 assert c['HOST_raw_FP32_words_checked']==1724383232
 assert c['retained_primary_and_small_raw_bytes']==9850930176
 assert c['GPU_original_FP32_words_checked']==c['GPU_vector_original_FP32_words_checked']==4200857600
 assert set(c['GPU_prefill_tile_FP32_words_checked'])=={'32x32','64x32','64x64'}and all(v==4200857600 for v in c['GPU_prefill_tile_FP32_words_checked'].values())
 print('ALL201ACTUAL_PACKED_WEIGHTS_ORIGINAL2_AND_NEW3_TILE_READERS_FP32_BITS_EXACT',flush=True)
 run([str(r/'out/build/garnet-metal-worldsense-native-text-worker-on/direct_float_corpus_test'),str(w/'direct-upload-selection.json'),str(w/'direct-upload-corpus.json')],'direct-upload-corpus.log')
 print('DIRECT_NATIVE_LEARNED_RAW_WEIGHTS_FP32_BITS_EXACT',flush=True)
 run([sys.executable,str(w/'resident_controller.py')],'trained-resident-controller.log')
 for mode in ['off','on','fault']:run([str(n/'xlang3'),'-B',str(w/'linear_guard.py'),mode],'trained-linear-'+mode+'.log')
 for mode in ['off','on']:run([str(n/'xlang3'),'-B',str(w/'vision_grid_guard.py'),mode],'vision-grid-'+mode+'.log')
 for mode in ['off','primary','on']:run([str(n/'xlang3'),'-B',str(w/'api_state_guard.py'),mode],'api-state-guard-'+mode+'.log')
 assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==h for k,h in manifest.items())
 (w/'api-correctness-result.json').write_text(json.dumps({'all32VLM28text_state_bits_tokens_prefix_capacity_vision_and_layer_isolation':True,'all201actualpacked_weights_original_two_and_three_tile_GPU_bit_readers':True,'native7tests_all4builds_and186_projection_words_exact':True,'actual_priorprimary_and_vector_API_states_and_tokens_match':True,'source1723_unchanged':True},indent=2)+'\n')
 print('FULL_VECTOR_PREFILL_MODEL_STATE_AND_FIVE_GPU_READERS_CORRECTNESS_PASS',flush=True)
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
