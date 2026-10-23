"""Compare PE CUDA device fatbins with an archived native binary."""
import hashlib,json,struct,sys,zipfile
from pathlib import Path
archive,current,output=map(Path,sys.argv[1:4])
assert not output.exists(),'Preserve device comparison evidence'
with zipfile.ZipFile(archive) as z:prior=z.read('runtime/garnet.dll')
present=current.read_bytes()
def sections(data):
    pe=struct.unpack_from('<I',data,0x3c)[0];assert data[pe:pe+4]==b'PE\0\0'
    count=struct.unpack_from('<H',data,pe+6)[0];size=struct.unpack_from('<H',data,pe+20)[0]
    result={}
    for i in range(count):
        start=pe+24+size+40*i;name=data[start:start+8].rstrip(b'\0').decode('ascii')
        length,offset=struct.unpack_from('<II',data,start+16)
        if name.startswith('.nv'):
            assert offset+length<=len(data)
            result[name]={'bytes':length,'sha256':hashlib.sha256(data[offset:offset+length]).hexdigest()}
    assert '.nv_fatb' in result
    return result
a,b=sections(prior),sections(present)
result={'prior_archive':str(archive),'current_dll':str(current),
    'prior_dll_sha256':hashlib.sha256(prior).hexdigest(),'current_dll_sha256':hashlib.sha256(present).hexdigest(),
    'prior_sections':a,'current_sections':b,'device_fatbin_equal':a['.nv_fatb']==b['.nv_fatb'],
    'scope':'PE device-fatbin equality only; host registration pointers may differ. Not a correctness or speed gate.'}
output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
assert result['device_fatbin_equal'],'Device code differs'
