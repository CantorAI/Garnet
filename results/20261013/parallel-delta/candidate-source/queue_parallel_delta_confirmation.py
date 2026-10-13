import subprocess
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation');s=root/'source/test2026/metal';py='/opt/homebrew/bin/python3'
runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3'
build=str(root/'build-parallel-delta');source=str(root/'source');frontend=str(root/'comparators/vllm/venv/bin/python')
def launch(name,prior,command):
 folder=root/'logs'/name
 subprocess.run([py,str(s/'background_comparison.py'),'launch',str(folder),py,str(s/'run_after_owned_job.py'),str(prior),*command],check=True)
 return folder
prior=root/'logs/native-frame-ab-parallel-delta-retry2-20261013'
frame=launch('native-frame-ab-parallel-delta-confirm-20261013',prior,[py,str(s/'compare_vllm_metal.py'),runtime,build,source,frontend,str(root/'logs/native-frame-ab-parallel-delta-confirm-20261013/jobs'),'shared',str(root/'build-mps-vision-scores'),'vl'])
launch('parallel-delta-confirm-frame-summary-20261013',frame,[py,str(s/'summarize_native_frames.py'),str(frame/'jobs')])
text=launch('vllm-matched-parallel-delta-text-20261013',frame,[py,str(s/'compare_vllm_metal.py'),runtime,build,source,frontend,str(root/'logs/vllm-matched-parallel-delta-text-20261013/jobs'),'shared'])
launch('parallel-delta-text-target-summary-20261013',text,[py,str(s/'summarize_vllm_comparison.py'),str(text/'jobs')])
vl=launch('vllm-matched-parallel-delta-vl-20261013',text,[py,str(s/'compare_vllm_metal.py'),runtime,build,source,frontend,str(root/'logs/vllm-matched-parallel-delta-vl-20261013/jobs'),'shared','-','vl'])
launch('parallel-delta-vl-target-summary-20261013',vl,[py,str(s/'summarize_frame_comparison.py'),str(vl/'jobs')])
fptext=launch('equivalent-f32-parallel-delta-text-20261013',vl,[py,str(root/'compare_equivalent_f32.py'),runtime,build,source,frontend,str(root/'logs/equivalent-f32-parallel-delta-text-20261013/jobs'),'shared'])
launch('equivalent-f32-parallel-delta-text-summary-20261013',fptext,[py,str(root/'equivalent_f32_summarize_vllm_comparison.py'),str(fptext/'jobs')])
fpvl=launch('equivalent-f32-parallel-delta-vl-20261013',fptext,[py,str(root/'compare_equivalent_f32.py'),runtime,build,source,frontend,str(root/'logs/equivalent-f32-parallel-delta-vl-20261013/jobs'),'shared','-','vl'])
launch('equivalent-f32-parallel-delta-vl-summary-20261013',fpvl,[py,str(root/'equivalent_f32_summarize_frame_comparison.py'),str(fpvl/'jobs')])
launch('parallel-delta-default-fallback-20261013',fpvl,[py,str(root/'build_parallel_delta_fallback.py')])
