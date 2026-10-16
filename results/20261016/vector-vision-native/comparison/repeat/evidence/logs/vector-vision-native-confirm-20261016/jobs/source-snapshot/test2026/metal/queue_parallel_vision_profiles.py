"""Wait for exact owned twenty-stage validation, then run complete native ABBA."""
from pathlib import Path
source=Path(__file__).with_name('queue_fused_head_argmax_profiles.py').read_text()
source=source.replace('fused-head-argmax-checks-20261015','parallel-vision-checks-20261015').replace('pid=68935','pid=61469').replace('launch_fused_head_argmax_checks.py','launch_parallel_vision_checks.py').replace('fused-head-argmax-staging/manifest.json','parallel-vision-staging/manifest.json').replace('len(stages)==19','len(stages)==20')
begin=source.index("        helpers=(");end=source.index("        snapshots=",begin)
source=source[:begin]+"        helpers=('queue_parallel_vision_profiles.py','run_parallel_vision_profiles.py','compare_parallel_vision.py','test_parallel_vision_grid.py','run_parallel_vision_grid.py')\n"+source[end:]
source=source.replace("[('recovery','run_fused_head_session_checks.py'),('native','run_fused_head_argmax_profiles.py')]","[('native','run_parallel_vision_profiles.py')]")
exec(compile(source,__file__,'exec'))
