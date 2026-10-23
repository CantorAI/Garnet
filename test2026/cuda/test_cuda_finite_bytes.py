"""Exact bounded diagnostic bytes, finite checks and storage lifetime."""
import hashlib,json,os,struct,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
import garnet,tensor
output=sys.argv[1];io=garnet.cuda_tensor_io();checks=[]
def reject(name,action):
    rejected=False
    try:action()
    except BaseException:rejected=True
    assert rejected,name
    checks.append(name)
try:
    patterns=[0,0x80000000,1,0x80000001,0x007fffff,0x00800000,
        0x3f800000,0xbf800000,0x3eaaaaab,0x7f7fffff,0xff7fffff,0x00800001]
    expected=struct.pack('<12I',*patterns)
    values=list(struct.unpack('<12f',expected));source=io.from_host(values,[3,4])
    actual=io.finite_f32_bytes(source)
    legacy=struct.pack('<12f',*garnet.tensor_to_cpu(source).tolist())
    assert actual==expected==legacy
    assert hashlib.sha256(actual).hexdigest()==hashlib.sha256(legacy).hexdigest()
    checks.append('finite_exact_bits_and_legacy_sha256')
    for _ in range(3):assert io.finite_f32_bytes(source)==expected
    checks.append('repeated_read_preserves_source')
    view=io.storage_view(source,[2,4]);del source
    assert io.finite_f32_bytes(view)==expected[:32]
    checks.append('logical_view_span_and_retained_owner')
    for name,value in [('nan',float('nan')),('positive_inf',float('inf')),('negative_inf',-float('inf'))]:
        bad=io.from_host([0.,value,1.,-0.],[1,4])
        reject(name,lambda:io.finite_f32_bytes(bad))
    reject('cpu_source',lambda:io.finite_f32_bytes(garnet.tensor_to_cpu(view)))
    reject('symbolic_source',lambda:io.finite_f32_bytes(tensor.input('unmaterialized',shape=[1,4],dtype=tensor.float32)))
    zeros=io.zeros([8,131072]);blob=io.finite_f32_bytes(zeros)
    assert len(blob)==4194304
    assert hashlib.sha256(blob).hexdigest()==hashlib.sha256(b'\x00'*4194304).hexdigest()
    checks.append('maximum_bounded_copy_exact')
    reject('oversize_source',lambda:io.finite_f32_bytes(io.zeros([1048577])))
    result={'checks':checks,'passed':True,'max_copy_bytes':4194304,
        'scope':'Diagnostic exact IEEE754 bytes/finite rejection; no model quality or timing assertion'}
    Path(output).write_text(json.dumps(result,indent=2));print('FINITE_F32_BYTES_PASS',len(checks),flush=True)
finally:io.close()
