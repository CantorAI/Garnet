import subprocess
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation');s=root/'source/test2026/metal';py='/opt/homebrew/bin/python3'
build=root/'logs/mps-vision-context-build-20261013'
subprocess.run([py,str(s/'background_comparison.py'),'launch',str(build),py,str(root/'build_mps_vision_context.py')],check=True)
def launch(folder,prior,command):
 subprocess.run([py,str(s/'background_comparison.py'),'launch',str(folder),py,str(s/'run_after_owned_job.py'),str(prior),*command],check=True)
checks=root/'logs/mps-vision-context-checks-20261013';runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3'
launch(checks,build,[py,str(s/'three_model_suite.py'),runtime,str(root/'build-mps-vision-context'),str(root/'source'),'handoff-checks',str(checks/'jobs')])
frame=root/'logs/native-frame-ab-mps-vision-context-20261013'
launch(frame,checks,[py,str(s/'compare_vllm_metal.py'),runtime,str(root/'build-mps-vision-context'),str(root/'source'),str(root/'comparators/vllm/venv/bin/python'),str(frame/'jobs'),'shared',str(root/'build-mps-vision-scores'),'vl'])
launch(root/'logs/mps-vision-context-frame-summary-20261013',frame,[py,str(s/'summarize_native_frames.py'),str(frame/'jobs')])
