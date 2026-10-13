import subprocess
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation');s=root/'source/test2026/metal';py='/opt/homebrew/bin/python3'
build=root/'logs/exact-half-prefill-build-20261013'
subprocess.run([py,str(s/'background_comparison.py'),'launch',str(build),py,str(root/'build_exact_half_prefill.py')],check=True)
def launch(folder,prior,command):
 subprocess.run([py,str(s/'background_comparison.py'),'launch',str(folder),py,str(s/'run_after_owned_job.py'),str(prior),*command],check=True)
checks=root/'logs/exact-half-prefill-checks-20261013'
runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3'
launch(checks,build,[py,str(s/'three_model_suite.py'),runtime,str(root/'build-exact-half-prefill'),str(root/'source'),'handoff-checks',str(checks/'jobs')])
text=root/'logs/native-ab-exact-half-prefill-20261013'
base=[py,str(s/'compare_vllm_metal.py'),runtime,str(root/'build-exact-half-prefill'),str(root/'source'),str(root/'comparators/vllm/venv/bin/python')]
launch(text,checks,base+[str(text/'jobs'),'shared',str(root/'build-delta-split')])
ts=root/'logs/exact-half-text-summary-20261013'
launch(ts,text,[py,str(s/'summarize_native_builds.py'),str(text/'jobs')])
frame=root/'logs/native-frame-ab-exact-half-prefill-20261013'
launch(frame,ts,base+[str(frame/'jobs'),'shared',str(root/'build-delta-split'),'vl'])
launch(root/'logs/exact-half-frame-summary-20261013',frame,[py,str(s/'summarize_native_frames.py'),str(frame/'jobs')])
