"""Preserve completed retuned-profile logs, references, source and exact identities."""
from pathlib import Path
source=Path(__file__).with_name('archive_padded_mps_body.py').read_text().replace("['padded-mps-body-20261015']","['replanned-vl-prefill-20261015']")
source=source.replace("'.metal','.log','.txt')","'.metal','.log','.txt','.cmake')")
needle='proof=dict(files={}';assert source.count(needle)==1
source=source.replace(needle,'''meta=json.loads((root/'logs/replanned-vl-prefill-20261015/jobs/metadata.json').read_text())
for name,key in [('models/Qwen3.5-0.8B/model.safetensors','checkpoint_sha256'),('build-interleaved-screen-index/libgarnet.dylib','library_sha256')]:
    p=root/name
    with p.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    assert sha==meta[key];omitted[name]=dict(bytes=p.stat().st_size,sha256=sha)
p=root/'models/Qwen3.5-0.8B/config.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['config_sha256'];files['evidence/models/Qwen3.5-0.8B/config.json']=p
p=Path('/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3')
with p.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
assert sha==meta['runtime_sha256'];omitted['external/xlang3']=dict(bytes=p.stat().st_size,sha256=sha)
'''+needle)
exec(compile(source,__file__,'exec'))
