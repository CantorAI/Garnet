"""Wait on the exact live validation handle before the six-profile native ABBA."""
from pathlib import Path
source=Path(__file__).with_name('queue_screened_prefill_reset_profiles.py').read_text()
for old,new in [('screened-prefill-reset-checks-20261015','wide-batch-prefill-checks-20261015'),('pid=29706',"pid=json.loads((dependency/'pid.json').read_text())['pid']"),('len(stages)==26','len(stages)==28'),('screened-head-staging/manifest.json','wide-batch-prefill-staging/manifest.json'),('queue_screened_prefill_reset_profiles.py','queue_wide_batch_prefill_profiles.py'),('run_screened_prefill_reset_profiles.py','run_wide_batch_prefill_profiles.py'),('compare_screened_prefill_reset.py','compare_wide_batch_prefill.py'),('run_guarded_screened_prefill_reset_checks.py','run_guarded_wide_batch_prefill_checks.py'),('launch_screened_prefill_reset_checks.py','launch_wide_batch_prefill_checks.py')]:source=source.replace(old,new)
needle="        command=[sys.executable,str(root/'source/test2026/metal/run_wide_batch_prefill_profiles.py')";assert source.count(needle)==1
source=source.replace(needle,'''        baseline=root/'source';closure={}
        paths=set()
        for folder in ('src','include','cmake','xModel'):
            paths.update(p for p in (baseline/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.h','.cpp','.mm','.metal','.cmake'))
        paths.update(p for p in (baseline/'test2026/metal').glob('*') if p.is_file() and p.suffix in ('.py','.h','.cpp','.mm','.metal'))
        paths.update([baseline/'CMakeLists.txt',baseline/'test2026/pi5b/reference_decoder.py'])
        for p in sorted(paths):
            name=str(p.relative_to(baseline));raw=p.read_bytes();closure[name]=hashlib.sha256(raw).hexdigest();target=out/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        (out/'source-identities.json').write_bytes((json.dumps(closure,indent=2)+'\\n').encode())
'''+needle)
needle='        code=subprocess.call(command)';assert source.count(needle)==1
source=source.replace(needle,needle+"\n        assert all(hashlib.sha256((baseline/name).read_bytes()).hexdigest()==sha for name,sha in closure.items()),'Source changed during comparison'")
exec(compile(source,__file__,'exec'))
