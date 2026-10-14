"""Exclusive extra session gates before native performance profiles."""
from pathlib import Path
import sys
base=Path(__file__).with_name('run_joined_prefill_checks.py').read_text()
exec(compile(base[:base.index("assert not build.exists()")],str(Path(__file__)),'exec'))
build=r/'build-dirty-prefix-reset';runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3'
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
 run(model+'-session-recovery',[sys.executable,str(s/'test_prefix_reset_session_reference.py'),runtime,str(build),str(r/'source'),str(r/'models'/model),str(output/model)])
assert len(statuses)==3 and all(v['exit_code']==0 for v in statuses)
