"""Audit all completed 19 gates and the preserved XLang-only test failure."""
from pathlib import Path
source=Path(__file__).with_name('verify_fused_head_argmax_inference.py').read_text()
source=source[:source.index("trial=root/'logs'/native_trial")]
source=source.replace("assert proof['jobs'][trial]['exit_code']==0", "assert proof['jobs'][trial]['exit_code']==(0 if trial=='fused-head-argmax-checks-20261015' else 1)")
exec(compile(source,__file__,'exec'))
failed=root/'logs/fused-head-argmax-native-20261015';assert json.loads((failed/'recovery-exit.json').read_text())['exit_code']==1
log=(failed/'recovery/Qwen3-0.6B-session.log').read_text();assert 'NameError' in log and "name 'row' is not defined" in log
assert not (failed/'native').exists()
identities=json.loads((failed/'helper-identities.json').read_text())
for name,sha in identities.items():assert digest(failed/'helper-snapshot'/name)==sha
assert "key=lambda i:scores[row*width+i]" in (failed/'helper-snapshot/run_fused_head_session_reference.py').read_text()
assert proof['library_sha256']['build-fused-head-argmax']=='8cfcfb392d19fad68695ad7d80c8c3b5989f01ca956fa50370c7b2f960230177'
result=dict(verified_files=len(proof['files']),original_validation_stages=19,original_gates_all_passed=True,native_benchmark_started=False,failure='XLang NameError in reference argmax lambda closure; compiled candidate untouched',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
