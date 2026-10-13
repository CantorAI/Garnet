from pathlib import Path
import json,subprocess
r=Path('/Users/zsdx/garnet-metal-investigation');s=r/'source/test2026/metal';py='/opt/homebrew/bin/python3'
for prior in ('mps-half-decode-build-20261013','mps-half-decode-exclusivity-wait-20261013'):
 subprocess.run([py,str(s/'run_after_owned_job.py'),str(r/'logs'/prior),py,'-c','pass'],check=True)
subprocess.run([py,str(r/'three_model_suite_guarded_mps_decode.py'),'/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3',str(r/'build-mps-half-decode'),str(r/'source'),'handoff-checks',str(r/'logs/mps-half-decode-checks-20261013/jobs')],check=True)
