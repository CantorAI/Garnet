"""CPU mock-artifact negative tests for the independent native-only raw audit."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

repo=Path(__file__).resolve().parents[2];sys.path.insert(0,str(repo/'tools/gpt_oss'))
from audit_xqa_native import audit,coverage,SPECS
from prepare_xqa_sources import generate

def save(p,d):p.write_text(json.dumps(d))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary);(root/'native-payload').mkdir()
    vendor=repo/'plugins/gpt_oss/third_party/flashinfer_xqa'
    shutil.copytree(vendor,root/'vendor');generate(vendor,root/'generated')
    sources=['test2026/gpt_oss/xqa_attention_parity.cu','plugins/gpt_oss/cuda/gpt_oss_xqa_attention.cu',
        'plugins/gpt_oss/cuda/gpt_oss_xqa_bridge.cu','plugins/gpt_oss/include/gpt_oss_xqa_attention.h',
        'plugins/gpt_oss/include/gpt_oss_xqa_layout.h','plugins/gpt_oss/cuda/gpt_oss_xqa_bridge.h',
        'plugins/gpt_oss/include/gpt_oss_kernels.h','plugins/gpt_oss/CMakeLists.txt','tools/gpt_oss/prepare_xqa_sources.py']
    for n in sources:
        p=root/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/n,p)
    inference={}
    for name in ('libgarnet.so','libgarnet_gpt_oss.so','garnet_gpt_oss_kernel_parity'):
        (root/'native-payload'/name).write_bytes(b'CPU MOCK ONLY '+name.encode());inference[name]=sha(root/'native-payload'/name)
    (root/'native-payload/xqa-parity').write_bytes(b'CPU MOCK ONLY native executable')
    metadata=dict(source_commit='a'*40,inference_binaries=inference,
        private_binary_sha256=sha(root/'native-payload/xqa-parity'),source_sha256={n:sha(repo/n) for n in sources},
        generated_sha256={p.name:sha(p) for p in (root/'generated').iterdir()})
    save(root/'gate-metadata.json',metadata);(root/'source.txt').write_text('a'*40+'\n')
    save(root/'terminal.json',dict(terminal=True,actual_exit=0,phase='complete-native-only'))
    (root/'hardware.csv').write_text('index, name, uuid, driver_version\n0, CPU MOCK, MOCK0, MOCK\n1, CPU MOCK, MOCK1, MOCK\n')
    lines=[]
    for d in (0,1):
        lines.append(f'DEVICE {d} sequential independent native tests')
        for i,(b,p,w,h,l,pattern,unique,fixed) in enumerate(SPECS):
            lines.append(f'CASE {i} B={b} KVH={h} pages={p} physical={b*p if unique else 9} window={w} layer={l} pattern={pattern} ordinary/captured PASS')
    totals=coverage()
    lines.append('XQA_NATIVE_PARITY_ACTUAL_PASS devices=2 '+' '.join(f'{n}={v}' for n,v in totals.items())+' max_abs=0.001 bound=.006*(1+abs)')
    lines.append('SCOPE native sampled FP64/full safety metadata/read-only KV/ordinary+changed graph; NOT compiled/plugin/pretrained/performance or all-four goal proof')
    text='\n'.join(lines)+'\n'
    for name in ('native','native-mem','native-trace'):(root/(name+'.log')).write_text(text+('ERROR SUMMARY: 0 errors\n' if name=='native-mem' else ''))
    (root/'native-mem.xml').write_text('<ComputeSanitizerOutput/>')
    (root/'native-kernels.csv').write_text('Name,Instances\nGarnetGptOssXqaKernel,144\ngptOssXqaPrepare,144\ngptOssXqaFinalize,144\n')
    assert audit(root,root/'positive.json')['sanitizer']['records']==0
    rejected=0
    def reject(name,change):
        global rejected
        p=root/name;original=p.read_bytes();p.write_bytes(change(original));out=root/f'negative-{rejected}.json'
        try:
            try:audit(root,out)
            except (ValueError,KeyError,FileNotFoundError):pass
            else:raise AssertionError('Invalid native artifact accepted: '+name)
            assert not out.exists();rejected+=1
        finally:p.write_bytes(original)
    def mutate_json(change):
        def update(raw):
            d=json.loads(raw);change(d);return json.dumps(d).encode()
        return update
    for change in (lambda d:d.update(actual_exit=1),lambda d:d.update(terminal=False),lambda d:d.update(phase='ordinary-native')):
        reject('terminal.json',mutate_json(change))
    for original,replacement in [('devices=2','devices=1'),('CASE 17 B=512','CASE 16 B=512'),
        ('ordinary/captured PASS','ordinary/captured FAIL'),('invocations=144','invocations=143'),
        ('guards=1116','guards=1115'),('max_abs=0.001','max_abs=nan'),('max_abs=0.001','max_abs=0.008'),
        ('bound=.006','bound=.06'),('SCOPE native sampled','SCOPE full pretrained')]:
        reject('native.log',lambda raw,o=original,n=replacement:raw.replace(o.encode(),n.encode()))
    reject('native-mem.log',lambda raw:raw.replace(b'ERROR SUMMARY: 0',b'ERROR SUMMARY: 1'))
    reject('native-mem.log',lambda raw:raw+b'No attachable process found\n')
    for kind in ('Memory','Api'):
        reject('native-mem.xml',lambda raw,k=kind:f'<ComputeSanitizerOutput><record><kind>{k}</kind></record></ComputeSanitizerOutput>'.encode())
    reject('native-kernels.csv',lambda raw:raw.replace(b'gptOssXqaFinalize,144',b'gptOssXqaFinalize,143'))
    reject('hardware.csv',lambda raw:raw.replace(b'MOCK1',b'MOCK0'))
    reject('source.txt',lambda raw:b'b'*40)
    for n in ('native-payload/xqa-parity','native-payload/libgarnet.so','source/plugins/gpt_oss/include/gpt_oss_xqa_layout.h',
        'vendor/NOTICE','vendor/SHA256SUMS','generated/mha.cu','generated/NOTICE'):
        reject(n,lambda raw:raw+b'\nCPU intentional corruption\n')
    reject('gate-metadata.json',mutate_json(lambda d:d['source_sha256'].pop('plugins/gpt_oss/include/gpt_oss_kernels.h')))
    reject('generated/garnet-adaptation.json',mutate_json(lambda d:d.update(upstream_commit='b'*40)))
    (root/'generated/extra').write_text('unexpected')
    try:audit(root,root/'bad-extra.json')
    except ValueError:rejected+=1
    else:raise AssertionError('Extra generated dependency accepted')
    (root/'generated/extra').unlink()
    try:audit(root,root/'positive.json')
    except ValueError:rejected+=1
    else:raise AssertionError('Existing audit overwritten')
print('CPU MOCK XQA native audit rejects',rejected,'coverage/status/identity/source/numerical-summary/memory/dispatch hazards; no GPU/native execution proof')
