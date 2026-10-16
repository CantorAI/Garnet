"""Reuse archived passing25 stages unchanged; execute only8 missing original-bound checks."""
from pathlib import Path
import hashlib,json,shutil,sys,tarfile
resume_root=Path(sys.argv[1]).resolve();resume_output=Path(sys.argv[2]).resolve();resume_parent=Path(json.loads((resume_root/'preparation.json').read_text())['parent'])
resume_prior=resume_root/'logs/prepared-qkv-native-checks-20261016/jobs';resume_archive=resume_parent/'prepared-qkv-native-checks-20261016.tar.gz'
def resume_digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
assert resume_digest(resume_archive)=='f1c7ba015ebaf86c6982bd0284c03d2f6116fb2d6e12fd05a3cdcc31d5c735eb'
assert json.loads((resume_prior.parent/'exit.json').read_text())['exit_code']==1
resume_statuses=json.loads((resume_prior/'statuses.json').read_text());assert len(resume_statuses)==25 and all(v['exit_code']==0 and v['interruption'] is None for v in resume_statuses)
resume_meta=json.loads((resume_prior/'metadata.json').read_text());assert resume_meta['candidate_sha256']=='e34fc562b4840af94eed5e4a6043cdf912a4beb15a5c48d75d74ef7946012f77'
assert resume_digest(resume_root/'build-prepared-qkv-text/libgarnet.dylib')==resume_meta['candidate_sha256']
resume_preparation=json.loads((resume_root/'preparation.json').read_text())
for resume_name,resume_sha in resume_preparation['source_sha256'].items():assert resume_digest(resume_root/'source'/resume_name)==resume_sha,resume_name
resume_original=resume_prior.parent/'jobs.prepared_qkv.complete33.executed.py';resume_raw=resume_original.read_text();assert resume_raw.count('assert len(statuses)==23')==resume_raw.count('assert len(statuses)==29')==1
resume_prefix=resume_raw[:resume_raw.index('\nexternal=')]
resume_tail=resume_raw[resume_raw.index("for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):",resume_raw.index('assert len(statuses)==23')):resume_raw.index('\nfor name,sha in external_hashes.items():')]
resume_tail=resume_tail.replace('assert len(statuses)==29','assert len(statuses)==31');assert resume_tail.count('run(')==3 and 'assert len(statuses)==33' in resume_tail
# The exact original exclusive run/guard functions are retained. No configure,
# build, primitive or completed numerical stage is executed again.
exec(compile(resume_prefix,'archived_original_exclusive_validation_functions','exec'))
shutil.copytree(resume_prior,resume_output,dirs_exist_ok=True)
(resume_output.parent/'jobs.prepared_qkv.complete33.executed.py').write_bytes(resume_original.read_bytes())
(resume_output/'resumed-tail.executed.py').write_bytes(resume_tail.encode())
with tarfile.open(resume_archive) as resume_tar:resume_proof=json.load(resume_tar.extractfile('evidence/manifest.json'))
resume_prefix_name='evidence/logs/prepared-qkv-native-checks-20261016/jobs/'
for resume_name,resume_info in resume_proof['files'].items():
 if resume_name.startswith(resume_prefix_name):assert resume_digest(resume_prior/resume_name[len(resume_prefix_name):])==resume_info['sha256']
assert resume_digest(resume_original)==resume_proof['files']['evidence/logs/prepared-qkv-native-checks-20261016/jobs.prepared_qkv.complete33.executed.py']['sha256']
(resume_output/'prior-manifest.json').write_bytes((json.dumps(resume_proof,indent=2)+'\n').encode())
resume_copied={}
for resume_path in resume_prior.rglob('*'):
 if not resume_path.is_file():continue
 resume_relative=resume_path.relative_to(resume_prior).as_posix();resume_sha=resume_digest(resume_path);assert resume_digest(resume_output/resume_relative)==resume_sha;resume_copied[resume_relative]=resume_sha
resume_paths=[Path(__file__),Path(__file__).with_name('launch_resume_prepared_qkv_native_checks.py')];resume_hashes={p.as_posix():resume_digest(p) for p in resume_paths}
for resume_path in resume_paths:
 resume_target=resume_output/'resume-driver-snapshot'/resume_path.name;resume_target.parent.mkdir(exist_ok=True);resume_target.write_bytes(resume_path.read_bytes())
resume_record=dict(prior_archive_sha256=resume_digest(resume_archive),prior_status=json.loads((resume_prior.parent/'exit.json').read_text()),original_driver_sha256=resume_digest(resume_original),candidate_sha256=resume_meta['candidate_sha256'],source_manifest=resume_preparation['source_sha256'],reused_stages=[v['name'] for v in resume_statuses],fresh_stages=[],reused_file_sha256=resume_copied,resume_source_sha256=resume_hashes,
 original_count_assertions=[23,29,33],corrected_counts=[25,31,33],scope='Only stage-count bookkeeping corrected; unchanged library/source/bounds, archived passing25 reused, missing8 execute fresh.',full_goal_met=False)
(resume_output/'resume-provenance.json').write_bytes((json.dumps(resume_record,indent=2)+'\n').encode());statuses=list(resume_statuses)
runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3';py=sys.executable;models=resume_root/'models'
exec(compile(resume_tail,str(resume_output/'resumed-tail.executed.py'),'exec'))
assert len(statuses)==33 and statuses[:25]==resume_statuses
resume_record['fresh_stages']=[v['name'] for v in statuses[25:]];assert len(resume_record['fresh_stages'])==8
(resume_output/'resume-provenance.json').write_bytes((json.dumps(resume_record,indent=2)+'\n').encode())
for resume_name,resume_sha in resume_preparation['source_sha256'].items():assert resume_digest(resume_root/'source'/resume_name)==resume_sha
for resume_name,resume_sha in resume_hashes.items():assert resume_digest(Path(resume_name))==resume_sha
assert resume_digest(resume_root/'build-prepared-qkv-text/libgarnet.dylib')==resume_meta['candidate_sha256']
print('Original25 archived checks reused unchanged; all8 missing checks completed. No new build or model performance claim.',flush=True)
