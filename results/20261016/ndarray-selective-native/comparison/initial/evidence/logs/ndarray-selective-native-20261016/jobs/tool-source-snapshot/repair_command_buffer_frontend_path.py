"""After frozen failure evidence, link the existing isolated CPU image frontend."""
from pathlib import Path
import hashlib,json,subprocess,sys

parent=Path(sys.argv[1]).resolve();root=parent/'mlp-command-buffer-investigation-20261015'
native=root/'logs/mlp-command-buffer-native-20261015'
assert json.loads((native/'exit.json').read_text())['exit_code']==1
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
expected='cc2e35df829edd4e9d2539b4a095e63f29ca32e7d5b7570e7a4af2838ae11f89'
assert digest(root/'command-buffer-native-20261015.tar.gz')==expected
assert digest(root/'build-mlp-command-buffer/libgarnet.dylib')=='612af1059c3f86c5d7d5affe3223065b3457c1036bf10464ea3a763672a7e549'
prep=json.loads((root/'preparation.json').read_text())
text={'.json','.py','.h','.cpp','.mm','.metal','.cmake','.log','.txt','.make'}
for name,sha in prep['source_sha256'].items():
    if Path(name).suffix in text:assert digest(root/'source'/name)==sha,name
alias=root/'comparators';assert not alias.exists() and not alias.is_symlink()
interpreter=parent/'comparators/vllm/venv/bin/python';assert interpreter.is_file()
alias.symlink_to(parent/'comparators',target_is_directory=True)
assert (alias/'vllm/venv/bin/python').resolve()==interpreter.resolve()
command=[str(alias/'vllm/venv/bin/python'),'-c','import sys,PIL,numpy; print(sys.version.split()[0]); print(PIL.__version__); print(numpy.__version__)']
result=subprocess.run(command,capture_output=True,text=True);assert result.returncode==0,result.stderr
report=dict(scope='Frontend executable path repair only; no source/library/model changes or package installations.',
    original_failure_archive_sha256=expected,linked_existing_investigation_frontend=True,
    copied_source_hashes_unchanged=True,interpreter_sha256=digest(interpreter),
    command=command,exit_code=result.returncode,versions=result.stdout.splitlines(),full_goal_met=False)
(root/'frontend-path-repair.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
print(json.dumps(report))
