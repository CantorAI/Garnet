import hashlib,json,subprocess
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation');folder=root/'logs/xlang-image-preprocessing-20261012'
folder.mkdir(parents=True,exist_ok=True);output=folder/'patches.f32';assert not output.exists()
metadata=root/'images/RealWorld-04.batch1.json'
command=['/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3',str(root/'source/test2026/metal/probe_xlang_image_preprocessing.py'),str(root/'source'),str(root/'models/Qwen3.5-0.8B'),str(metadata),str(output)]
print('Native preprocessing probe command',json.dumps(command),flush=True)
result=subprocess.run(command,capture_output=True,text=True)
print(result.stdout,flush=True);print(result.stderr,flush=True)
assert result.returncode==0 and 'xlang_image_preparation_complete' in result.stdout
expected=json.loads(metadata.read_text())['patch_sha256'];actual=hashlib.sha256(output.read_bytes()).hexdigest()
assert actual==expected,(actual,expected)
print('Native CLI preprocessing produced byte-identical canonical patches',flush=True)
