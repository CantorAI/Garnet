import subprocess
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation');s=root/'source/test2026/metal';py='/opt/homebrew/bin/python3'
runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3'
build=str(root/'build-mps-vision-scores');source=str(root/'source');frontend=str(root/'comparators/vllm/venv/bin/python')
def launch(name,prior,command):
 folder=root/'logs'/name
 subprocess.run([py,str(s/'background_comparison.py'),'launch',str(folder),py,str(s/'run_after_owned_job.py'),str(prior),*command],check=True)
 return folder
prior=root/'logs/decode-occupancy-confirm-20261013'
frame=launch('native-frame-ab-mps-vision-confirm-20261013',prior,[py,str(s/'compare_vllm_metal.py'),runtime,build,source,frontend,str(root/'logs/native-frame-ab-mps-vision-confirm-20261013/jobs'),'shared',str(root/'build-kv-chunk-vector'),'vl'])
launch('mps-vision-confirm-frame-summary-20261013',frame,[py,str(s/'summarize_native_frames.py'),str(frame/'jobs')])
text=launch('vllm-matched-mps-vision-text-20261013',frame,[py,str(s/'compare_vllm_metal.py'),runtime,build,source,frontend,str(root/'logs/vllm-matched-mps-vision-text-20261013/jobs'),'shared'])
launch('mps-vision-text-target-summary-20261013',text,[py,str(s/'summarize_vllm_comparison.py'),str(text/'jobs')])
vl=launch('vllm-matched-mps-vision-vl-20261013',text,[py,str(s/'compare_vllm_metal.py'),runtime,build,source,frontend,str(root/'logs/vllm-matched-mps-vision-vl-20261013/jobs'),'shared','-','vl'])
launch('mps-vision-vl-target-summary-20261013',vl,[py,str(s/'summarize_frame_comparison.py'),str(vl/'jobs')])
