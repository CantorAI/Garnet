import subprocess
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation');s=root/'source/test2026/metal';py='/opt/homebrew/bin/python3'
def launch(folder,prior,command):
 subprocess.run([py,str(s/'background_comparison.py'),'launch',str(folder),py,str(s/'run_after_owned_job.py'),str(prior),*command],check=True)
text=root/'logs/vllm-matched-text-vector-slice-20261013'
base=[py,str(s/'compare_vllm_metal.py'),'/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3',str(root/'build-vector-slice'),str(root/'source'),str(root/'comparators/vllm/venv/bin/python')]
launch(text,root/'logs/vector-slice-boundary-checks-20261013',base+[str(text/'jobs'),'shared'])
ts=root/'logs/vllm-text-vector-summary-20261013'
launch(ts,text,[py,str(s/'summarize_vllm_comparison.py'),str(text/'jobs')])
frame=root/'logs/vllm-matched-vl-vector-slice-20261013'
launch(frame,ts,base+[str(frame/'jobs'),'shared','-','vl'])
launch(root/'logs/vllm-frame-vector-summary-20261013',frame,[py,str(s/'summarize_frame_comparison.py'),str(frame/'jobs')])
