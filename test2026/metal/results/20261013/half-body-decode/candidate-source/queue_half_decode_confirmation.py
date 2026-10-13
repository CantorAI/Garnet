import subprocess
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation');s=root/'source/test2026/metal';py='/opt/homebrew/bin/python3'
text=root/'logs/native-ab-half-decode-confirm2-20261013'
base=[py,str(s/'compare_vllm_metal.py'),'/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3',str(root/'build-exact-half-decode'),str(root/'source'),str(root/'comparators/vllm/venv/bin/python')]
subprocess.run([py,str(s/'background_comparison.py'),'launch',str(text),*base,str(text/'jobs'),'shared',str(root/'build-kv-chunk-vector')],check=True)
def launch(folder,prior,command):
 subprocess.run([py,str(s/'background_comparison.py'),'launch',str(folder),py,str(s/'run_after_owned_job.py'),str(prior),*command],check=True)
ts=root/'logs/half-decode-confirm2-text-summary-20261013'
launch(ts,text,[py,str(s/'summarize_native_builds.py'),str(text/'jobs')])
frame=root/'logs/native-frame-ab-half-decode-confirm2-20261013'
launch(frame,ts,base+[str(frame/'jobs'),'shared',str(root/'build-kv-chunk-vector'),'vl'])
launch(root/'logs/half-decode-confirm2-frame-summary-20261013',frame,[py,str(s/'summarize_native_frames.py'),str(frame/'jobs')])
