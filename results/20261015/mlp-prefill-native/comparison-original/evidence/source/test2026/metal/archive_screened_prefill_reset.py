"""Preserve terminal combined gates and screened-control native measurements."""
from pathlib import Path
wrapper=Path(__file__).with_name('archive_screened_head_inference.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
for old,new in [('screened-head-checks-20261015','screened-prefill-reset-checks-20261015'),('screened-head-native-20261015','screened-prefill-reset-native-20261015'),('build-screened-head','build-screened-prefill-reset'),('build-global-half-alpha','build-screened-head'),('887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f','2b44d93ab1e1980845be89ea59cbaedbebcb66ad385ba24df3d52af2b6bacc23')]:source=source.replace(old,new)
for old,new in [('archive_screened_head_inference.py','archive_screened_prefill_reset.py'),('run_guarded_screened_head_checks.py','run_guarded_screened_prefill_reset_checks.py'),('launch_screened_head_checks.py','launch_screened_prefill_reset_checks.py'),('compare_screened_head.py','compare_screened_prefill_reset.py'),('run_screened_head_profiles.py','run_screened_prefill_reset_profiles.py')]:source=source.replace(old,new)
needle='proof=dict(files={}' ;assert source.count(needle)==1
source=source.replace(needle,"for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):\n    p=root/'models'/model/'config.json';entries[str(p.relative_to(root))]=p\n"+needle)
exec(compile(source,__file__,'exec'))
