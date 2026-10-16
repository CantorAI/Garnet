"""Freeze the accepted d52 source/library, original assets and two-cohort matched policy."""
from pathlib import Path
import hashlib,json,platform,subprocess,sys
root=Path(sys.argv[1]).resolve();tools=Path(__file__).parent;native=root/'vector-vision-native-investigation-20261016';source=native/'source';proofs=tools/'proofs'

def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

for name in ('nibble-screen-index-20261016','sixbit-screen-index-20261016','sixbit-screen-index-clean-initial-20261016'):
 folder=root/'logs'/name;pid=json.loads((folder/'pid.json').read_text())['pid'];state=subprocess.run(['/bin/ps','-p',str(pid),'-o','stat='],capture_output=True,text=True);assert state.returncode in (0,1) and (not state.stdout.strip() or state.stdout.strip().startswith('Z')) and json.loads((folder/'exit.json').read_text())['exit_code']==0 and (root/(name+'.tar.gz')).is_file()
decision=json.loads((proofs/'native-decision-repeat.json').read_text());assert decision['native_confirmation_pass'] and decision['candidate_library_sha256']=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e' and len(decision['trials'])==2
for label in ('initial','repeat'):
 audit=json.loads((proofs/('native-'+label+'-verification.json')).read_text());assert audit['full_output_requests']==72 and audit['actual_vector_vision_routes']
for trial in decision['trials']:
 assert len(trial['profiles'])==6 and all(p['decode_ratio']>=.98 for p in trial['profiles']) and all(p['ttft_fraction']<=.98 for p in trial['profiles'] if p['model']=='Qwen3.5-0.8B')
prep=json.loads((native/'preparation.json').read_text());source_manifest={}
for name,sha in prep['source_sha256'].items():
 path=source/name;assert digest(path)==sha,name;source_manifest[path.relative_to(root).as_posix()]=sha
assert len(prep['source_sha256'])==1406
for p in [*tools.glob('*.py'),*proofs.glob('*.json'),root/'source/test2026/metal/compare_vllm_metal.py']:
 source_manifest[p.relative_to(root).as_posix()]=digest(p)
library=native/'build-vector-vision-width4/libgarnet.dylib';assert digest(library)==decision['candidate_library_sha256']
bridge=root/'logs/webp-thread-probe-confirm-20261013/jobs/libwebp_thread.dylib';private_bridge=native/'logs/webp-thread-probe-confirm-20261013/jobs/libwebp_thread.dylib'
assert digest(bridge)==digest(private_bridge)=='3c86ac196227b3729477cd24704ea4cea4e796fe5f9933c7a5a20361a32804a3'
source_manifest[bridge.relative_to(root).as_posix()]=digest(bridge)
runtime=Path('/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3');hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize','hw.model')}
assert hardware['machdep.cpu.brand_string']=='Apple M4' and int(hardware['hw.memsize'])==16*2**30 and platform.machine()=='arm64'
assets={}
models=('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B')
for tree in ('models','models-exact-f32'):
 for model in models:
  for p in (root/tree/model).iterdir():
   if p.is_file() and (p.suffix in ('.json','.safetensors','.txt','.model')):assets[p.relative_to(root).as_posix()]=dict(bytes=p.stat().st_size,sha256=digest(p))
selection=dict(scope='Fresh repeated matched FP32 cohorts after accepted vector-vision native confirmation. Original learned values, input/caps/stops/preprocessing, sequential G/V/V/G; no cache/speculation. No stock BF16/equal-storage claim.',models={m:{str(b):'vector_vision' for b in (1,2)} for m in models},builds=dict(vector_vision=dict(directory=library.parent.relative_to(root).as_posix(),sha256=digest(library),attention_prefix_enabled=True,global_half_alpha_enabled=True,screened_head_enabled=True,vector_vision_width=4)),native_source_root=source.relative_to(root).as_posix(),runtime_sha256=digest(runtime),guard_source_sha256=digest(root/'source/test2026/metal/compare_vllm_metal.py'),source_manifest=source_manifest,asset_manifest=assets,hardware=hardware,common_frontend='combined threaded decode and exact-coordinate strip resize',cohort_design=dict(complete_cohorts=2,repeat_after_complete_output_dtype_and_archive_audit=True,repeat_even_if_initial_performance_targets_unmet=True,no_favorable_third_repeat=True,worker_order=['garnet','vllm-metal','vllm-metal','garnet'],fresh_workers_per_policy=2,requests_per_worker=3,warm_requests_per_policy=4,separate_initialization_first_after_init_and_warm=True,hardware_caches_unflushed=True,archive_transfers_only_after_gpu_cohorts=True),vl_b2_internal_prefill_plan=[80],full_goal_met=False)
path=root/'work/vector-vision-next-comparator-selection.json';raw=(json.dumps(selection,indent=2)+'\n').encode()
if path.exists():assert path.read_bytes()==raw
else:path.write_bytes(raw)
print(json.dumps(dict(selection=str(path),selection_sha256=digest(path),source_files=len(source_manifest),assets=len(assets),native_library_sha256=digest(library),hardware=hardware,full_goal_met=False)))
