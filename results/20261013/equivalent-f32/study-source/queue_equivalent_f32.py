import subprocess
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation');s=root/'source/test2026/metal';py='/opt/homebrew/bin/python3'
runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3'
build=str(root/'build-mps-vision-scores');source=str(root/'source');frontend=str(root/'comparators/vllm/venv/bin/python')
audit=root/'logs/exact-f32-actual-dtypes-fit-20261013'
subprocess.run([py,str(s/'background_comparison.py'),'launch',str(audit),py,str(s/'run_exact_f32_dtype_audit.py'),str(root),str(audit/'jobs')],check=True)
def launch(name,prior,command):
 folder=root/'logs'/name
 subprocess.run([py,str(s/'background_comparison.py'),'launch',str(folder),py,str(s/'run_after_owned_job.py'),str(prior),*command],check=True)
 return folder
text=launch('equivalent-f32-text-20261013',audit,[py,str(root/'compare_equivalent_f32.py'),runtime,build,source,frontend,str(root/'logs/equivalent-f32-text-20261013/jobs'),'shared'])
launch('equivalent-f32-text-summary-20261013',text,[py,str(root/'equivalent_f32_summarize_vllm_comparison.py'),str(text/'jobs')])
vl=launch('equivalent-f32-vl-20261013',text,[py,str(root/'compare_equivalent_f32.py'),runtime,build,source,frontend,str(root/'logs/equivalent-f32-vl-20261013/jobs'),'shared','-','vl'])
launch('equivalent-f32-vl-summary-20261013',vl,[py,str(root/'equivalent_f32_summarize_frame_comparison.py'),str(vl/'jobs')])
