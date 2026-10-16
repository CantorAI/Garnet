"""Exact failing trained fixture on accepted d52 control; correctness diagnostic, not speed comparison."""
from pathlib import Path
import hashlib,json,sys,tarfile
diag_root=Path(sys.argv[1]).resolve();diag_output=Path(sys.argv[2]).resolve();diag_parent=Path(json.loads((diag_root/'preparation.json').read_text())['parent'])
def diag_digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
diag_archive=diag_parent/'prepared-qkv-native-checks-countfix-20261016.tar.gz';assert diag_digest(diag_archive)=='544e11d2c2f18582ceffb6a12404fb3113d4f03d5b382b55944076894e849c1a'
with tarfile.open(diag_archive) as diag_tar:diag_proof=json.load(diag_tar.extractfile('evidence/manifest.json'))
diag_fixture=diag_root/'logs/prepared-qkv-native-checks-countfix-20261016/jobs/trained-qkv-reference/layer0-b2-chunk22/fixture.json'
diag_name='evidence/'+diag_fixture.relative_to(diag_root).as_posix();assert diag_digest(diag_fixture)==diag_proof['files'][diag_name]['sha256'];diag_case=json.loads(diag_fixture.read_text());assert diag_case['batch']==2 and diag_case['chunk']==22 and diag_case['layer']==0
diag_inputs={}
for diag_step in diag_case['steps']:
 for diag_name in diag_step['inputs']+[diag_step['expected']]:
  diag_path=Path(diag_name);diag_relative=diag_path.relative_to(diag_root).as_posix();diag_sha=diag_digest(diag_path);assert diag_sha==diag_proof['omitted'][diag_relative]['sha256'];diag_inputs[diag_relative]=diag_sha
diag_control=diag_parent/'vector-vision-native-investigation-20261016';diag_library=diag_control/'build-vector-vision-width4/libgarnet.dylib';assert diag_digest(diag_library)=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'
diag_checkpoint=Path(diag_case['checkpoint']);assert diag_digest(diag_checkpoint)==diag_proof['omitted']['models/Qwen3-0.6B/model.safetensors']['sha256']
diag_driver=diag_control/'source/test2026/metal/run_dense_reference.py';diag_prep=json.loads((diag_root/'preparation.json').read_text());assert diag_digest(diag_driver)==diag_prep['changes']['test2026/metal/run_dense_reference.py']['before']
diag_archived_driver=diag_root/'logs/prepared-qkv-native-checks-20261016/jobs.prepared_qkv.complete33.executed.py';diag_raw=diag_archived_driver.read_text();diag_prefix=diag_raw[:diag_raw.index('\nexternal=')]
exec(compile(diag_prefix,'archived_original_exclusive_diagnostic_functions','exec'))
(diag_output/'metadata.json').write_bytes((json.dumps(dict(failed_archive_sha256=diag_digest(diag_archive),control_library_sha256=diag_digest(diag_library),control_driver_sha256=diag_digest(diag_driver),checkpoint_sha256=diag_digest(diag_checkpoint),fixture_sha256=diag_digest(diag_fixture),input_sha256=diag_inputs,scope=__doc__,full_goal_met=False),indent=2)+'\n').encode())
run('accepted-d52-exact-trained-fixture',['/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3',str(diag_driver),str(diag_fixture),str(diag_control/'build-vector-vision-width4'),str(diag_control/'source'),'bf16'])
print('Accepted d52 exact saved fixture completed under original bounds; diagnostic only, no model speed claim.',flush=True)
