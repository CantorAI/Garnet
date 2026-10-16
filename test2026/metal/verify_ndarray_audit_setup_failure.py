"""Reproduce the frozen silent-compiler assertion locally; no inference or tolerance change."""
from pathlib import Path
import hashlib,json,runpy,sys
base=Path(sys.argv[1]).resolve();validation=base.parent/'validation';proof=json.loads((base/'evidence/manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
assert proof['validation_archive_sha256']==json.loads((validation/'archive-verification.json').read_text())['archive_sha256']=='f60abf2cfd3a3b1a335e096a291b5343118772a50c9ee8d5f68c37db7fd0e388'
folder=base/'evidence/logs/ndarray-native-finalizer-20261015'
assert json.loads((folder/'exit.json').read_text())['exit_code']==1
commands=json.loads((folder/'commands.json').read_text());assert [(v['label'],v['exit_code']) for v in commands]==[('archive',0),('extract',0),('audit',1)]
helper=base/'evidence/consumed-tools/verify_ndarray_native_checks.py'
assert digest(helper)=='e1fe8dbaf9bd254a07e94e530f74c55834bf3feb55037204316a4040fcebf444'
saved=sys.argv;observed=None
try:
    sys.argv=[saved[0],str(validation),str(base/'unexpected-old-auditor-success.json')]
    runpy.run_path(str(helper),run_name='__main__')
except AssertionError as error:
    tb=error.__traceback__
    while tb.tb_next:tb=tb.tb_next
    stage=tb.tb_frame.f_locals.get('stage');casefolder=tb.tb_frame.f_locals.get('folder')
    assert tb.tb_lineno==17 and stage['name']=='ndarray-cache-compile' and stage['exit_code']==0
    assert (casefolder/'ndarray-cache-compile.log').is_file() and (casefolder/'ndarray-cache-compile.log').stat().st_size==0
    observed=dict(kind='AssertionError',expanded_line=17,stage=stage['name'],successful_compile_exit=stage['exit_code'],compiler_log_bytes=0)
finally:sys.argv=saved
assert observed is not None
result=dict(verified_files=len(proof['files']),reproduced_failure=observed,
    scope='Frozen CPU audit-setup failure reproduced; no inference, numerical gate change, performance or goal claim.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
