"""Strict-math bounded screening, original raw control, and full-row unsafe-input tests."""
from pathlib import Path
source=Path(__file__).with_name('run_screened_head_repeat.py').read_text();ns={'__file__':__file__}
exec(compile(source[:source.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace('from make_screened_head_harness import generate','from make_guarded_screened_head import generate_harness as generate,generate_shader')
source=source.replace("s/'run_screened_head.py'","s/'run_screened_head_repeat.py',s/'make_guarded_screened_head.py',s/'run_screened_head.py'")
source=source.replace("+raw+argmax+(s/'screened_head.metal').read_text()","+raw+argmax+generate_shader((s/'screened_head.metal').read_text())")
source=source.replace("metadata=dict(scope=__doc__,","metadata=dict(strict_screen_math=True,query_admission='finite abs<=1e6 and norm in [1e-10,1e10]; otherwise full original refinement',scope=__doc__,")
exec(compile(source,__file__,'exec'))
