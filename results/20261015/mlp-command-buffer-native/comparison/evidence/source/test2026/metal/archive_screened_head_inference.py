"""Preserve screened-head native gates, recovery and full-output ABBA evidence."""
from pathlib import Path
source=Path(__file__).with_name('archive_fused_head_argmax_inference.py').read_text()
source=source.replace('fused-head-argmax','screened-head')
for old,new in [('archive_fused_head_argmax_inference.py','archive_screened_head_inference.py'),('run_guarded_fused_head_argmax_checks.py','run_guarded_screened_head_checks.py'),('launch_fused_head_argmax_checks.py','launch_screened_head_checks.py'),('run_fused_head_session_checks.py','run_screened_head_session_checks.py'),('test_fused_head_session_reference.py','test_screened_head_session_reference.py'),('run_fused_head_session_reference.py','run_screened_head_session_reference.py'),('compare_fused_head_argmax.py','compare_screened_head.py'),('run_fused_head_argmax_profiles.py','run_screened_head_profiles.py')]:source=source.replace(old,new)
needle='proof=dict(files={}' ;assert source.count(needle)==1
source=source.replace(needle,'''for name in ('garnet_screened_head_test','garnet_screened_head_real_test'):
    p=root/'build-screened-head'/name;omitted[str(p.relative_to(root))]=dict(bytes=p.stat().st_size,sha256=digest(p))
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    assets=root/'logs/screened-head-after-vision-20261015/jobs'/(model+'-assets')
    p=assets/'manifest.json';entries[str(p.relative_to(root))]=p
    meta=json.loads(p.read_text())
    for name in ['raw.bin']+[q['name']+suffix for q in meta['queries'] for suffix in ('.input.bin','.reference.bin')]:
        p=assets/name;omitted[str(p.relative_to(root))]=dict(bytes=p.stat().st_size,sha256=digest(p))
'''+needle)
exec(compile(source,__file__,'exec'))
