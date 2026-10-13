import subprocess
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation');s=root/'source/test2026/metal';py='/opt/homebrew/bin/python3'
text=root/'logs/native-ab-vector-slice-20261013'
def launch(folder,prior,command):
 subprocess.run([py,str(s/'background_comparison.py'),'launch',str(folder),py,str(s/'run_after_owned_job.py'),str(prior),*command],check=True)
launch(root/'logs/vector-slice-text-summary-20261013',text,[py,str(s/'summarize_native_builds.py'),str(text/'jobs')])
frame=root/'logs/native-frame-ab-vector-slice-20261013'
launch(frame,text,[py,str(s/'compare_vllm_metal.py'),'/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3',str(root/'build-vector-slice'),str(root/'source'),str(root/'comparators/vllm/venv/bin/python'),str(frame/'jobs'),'shared',str(root/'build-prefill-wide'),'vl'])
launch(root/'logs/vector-slice-frame-summary-20261013',frame,[py,str(s/'summarize_native_frames.py'),str(frame/'jobs')])
