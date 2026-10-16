"""All original33 gates with archived passing25 reuse and explicit unchanged-candidate provenance."""
from pathlib import Path
import hashlib,json,sys
count_base=Path(sys.argv[1]).resolve();count_root=count_base/'evidence';count_proof=json.loads((count_root/'manifest.json').read_text());count_trial='prepared-qkv-native-checks-countfix-20261016';count_jobs=count_root/'logs'/count_trial/'jobs'
assert count_proof['prior_archive_sha256']=='f1c7ba015ebaf86c6982bd0284c03d2f6116fb2d6e12fd05a3cdcc31d5c735eb'
assert count_proof['jobs']['prepared-qkv-native-checks-20261016']['exit_code']==1 and count_proof['jobs'][count_trial]['exit_code']==0
count_prior=count_root/'logs/prepared-qkv-native-checks-20261016/jobs';count_old=json.loads((count_prior/'statuses.json').read_text());count_new=json.loads((count_jobs/'statuses.json').read_text());assert len(count_old)==25 and len(count_new)==33 and count_new[:25]==count_old and all(v['exit_code']==0 for v in count_new)
count_resume=json.loads((count_jobs/'resume-provenance.json').read_text());assert count_resume['prior_archive_sha256']==count_proof['prior_archive_sha256'] and count_resume['candidate_sha256']=='e34fc562b4840af94eed5e4a6043cdf912a4beb15a5c48d75d74ef7946012f77'
assert count_resume['reused_stages']==[v['name'] for v in count_old] and count_resume['fresh_stages']==[v['name'] for v in count_new[25:]] and len(count_resume['fresh_stages'])==8
assert count_resume['original_count_assertions']==[23,29,33] and count_resume['corrected_counts']==[25,31,33]
assert count_resume['source_manifest']==json.loads((count_root/'preparation.json').read_text())['source_sha256']
def count_digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,sha in count_resume['resume_source_sha256'].items():assert count_digest(count_jobs/'resume-driver-snapshot'/Path(name).name)==sha
for name,sha in count_resume['reused_file_sha256'].items():
 assert count_digest(count_prior/name)==sha
 if name not in ('statuses.json','background_service_checks.json'):assert count_digest(count_jobs/name)==sha
count_original=count_prior.parent/'jobs.prepared_qkv.complete33.executed.py';assert count_digest(count_original)==count_resume['original_driver_sha256']
count_raw=count_original.read_text();count_tail=count_raw[count_raw.index("for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):",count_raw.index('assert len(statuses)==23')):count_raw.index('\nfor name,sha in external_hashes.items():')].replace('assert len(statuses)==29','assert len(statuses)==31')
assert (count_jobs/'resumed-tail.executed.py').read_bytes()==count_tail.encode() and count_original.read_bytes()==(count_jobs.parent/'jobs.prepared_qkv.complete33.executed.py').read_bytes()
source=Path(__file__).with_name('verify_prepared_qkv_native_checks.py').read_text().replace("trial='prepared-qkv-native-checks-20261016'","trial='prepared-qkv-native-checks-countfix-20261016'")
needle="replace(needle,extra+needle);exec(compile(source,__file__,'exec'))";assert source.count(needle)==1
source=source.replace(needle,"""replace(needle,extra+needle)
replace("reused_validation_stages=[],reused_compile_stages=[],fresh_validation_stages=[s['name'] for s in stages],","reused_validation_stages=[s['name'] for s in stages[:25]],reused_compile_stages=['configure','build','vector-vision-primitive-compile','prepared-qkv-compile','ctest'],fresh_validation_stages=[s['name'] for s in stages[25:]],")
replace('regenerated_guard_and_full33_runner=True','regenerated_original_failed_full33_driver=True,regenerated_resumed_tail=True')
exec(compile(source,__file__,'exec'))""")
exec(compile(source,__file__,'exec'))
