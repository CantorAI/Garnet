"""Guard every trained session/fallback/recovery suite at 0.5 seconds."""
from pathlib import Path
base=Path(__file__).with_name('run_joined_prefill_checks.py').read_text()
exec(compile(base[:base.index("assert not build.exists()")],__file__,'exec'))
build=r/'build-fused-head-argmax';runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3'
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    run(model+'-session',[sys.executable,str(s/'test_fused_head_session_reference.py'),runtime,str(build),str(r/'source'),str(r/'models'/model),str(output/model)])
assert len(statuses)==3 and all(v['exit_code']==0 for v in statuses)
print('All three fused-head trained session/fallback/recovery suites PASS',flush=True)
