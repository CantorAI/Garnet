"""Reuse immutable passed references; independently repeat only chunk64 versus32."""
import hashlib,json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(exist_ok=False)
origin=root/'logs/replanned-vl-prefill-20261015';assert json.loads((origin/'exit.json').read_text())['exit_code']==0
meta=json.loads((origin/'jobs/metadata.json').read_text())
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
assert digest(root/'build-interleaved-screen-index/libgarnet.dylib')==meta['library_sha256']
assert digest(root/'models/Qwen3.5-0.8B/model.safetensors')==meta['checkpoint_sha256']
assert digest(root/'models/Qwen3.5-0.8B/config.json')==meta['config_sha256']
assert all(digest(root/'source'/name)==sha for name,sha in meta['source_sha256'].items())
assert digest(Path('/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3'))==meta['runtime_sha256']
(out/'dependency.json').write_bytes((json.dumps(dict(metadata=meta,reused_independent_profiles=12,reused_carried_steps=36),indent=2)+'\n').encode())
(out/'run-confirm-source.py').write_bytes(Path(__file__).read_bytes())
code=subprocess.call([sys.executable,str(root/'source/test2026/metal/compare_replanned_vl_prefill.py'),str(root),str(out/'chunk64'),'64']);assert code==0
profiles=json.loads((out/'chunk64/summary.json').read_text())['profiles'];assert len(profiles)==1
(out/'summary.json').write_bytes((json.dumps(dict(profiles=[dict(maximum=64,**profiles[0])]),indent=2)+'\n').encode())
