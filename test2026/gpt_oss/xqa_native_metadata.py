"""Execute the real wrapper metadata block with CPU mock build payloads only."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

repo=Path(__file__).resolve().parents[2];sys.path.insert(0,str(repo/'tools/gpt_oss'))
from prepare_xqa_sources import generate
script=(repo/'tools/gpt_oss/verify_xqa_native.sh').read_text()
marker='"$python" - "$repo" "$build" "$binary" "$directory" <<\'PY\'\n'
assert script.count(marker)==1
program=script.split(marker,1)[1].split('\nPY\n',1)[0]
with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary);build=root/'build';(build/'bin').mkdir(parents=True)
    for name in ('libgarnet.so','libgarnet_gpt_oss.so','garnet_gpt_oss_kernel_parity','xqa-parity'):
        (build/'bin'/name).write_bytes(b'CPU MOCK ONLY '+name.encode())
    vendor=repo/'plugins/gpt_oss/third_party/flashinfer_xqa'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    key=hashlib.sha256((sha(repo/'tools/gpt_oss/prepare_xqa_sources.py')+':'+sha(vendor/'SHA256SUMS')).encode()).hexdigest()
    generated=build/'Garnet/plugins/gpt_oss'/('xqa-'+key);generate(vendor,generated)
    def run(label):
        out=root/label;out.mkdir()
        result=subprocess.run([sys.executable,'-',str(repo),str(build),str(build/'bin/xqa-parity'),str(out)],
            input=program,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        return result,out
    result,out=run('metadata')
    assert result.returncode==0,result.stdout
    data=json.loads((out/'gate-metadata.json').read_text())
    assert len(data['source_sha256'])==9 and len(data['generated_sha256'])==18
    assert len(list((out/'vendor').iterdir()))==20
    assert data['private_binary_sha256']==sha(build/'bin/xqa-parity')
    assert all(sha(out/'source'/name)==digest for name,digest in data['source_sha256'].items())
    assert all(sha(out/'generated'/name)==digest for name,digest in data['generated_sha256'].items())
    # More than one current content-addressed build closure must not be guessed.
    other=build/('xqa-'+key);generate(vendor,other)
    result,out=run('ambiguous')
    assert result.returncode!=0 and 'Ambiguous generated build closure' in result.stdout
    assert not (out/'gate-metadata.json').exists()
print('Actual XQA wrapper metadata CPU mock: nested target closure/snapshot hashes and ambiguous closure rejection PASS; no native execution/GPU proof')
