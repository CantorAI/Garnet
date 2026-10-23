"""Verify actual compressed-tensors0.18.0 packing on a CPU fixture."""
import argparse,hashlib,importlib.metadata,inspect,json
from pathlib import Path
import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32,unpack_from_int32
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    version=importlib.metadata.version('compressed-tensors')
    if version!='0.18.0':raise ValueError('Expected the candidate recipe version0.18.0')
    values=torch.arange(-8,8,dtype=torch.int8,device='cpu').reshape(1,16)
    packed=pack_to_int32(values,4);unsigned=[int(v)&0xffffffff for v in packed.reshape(-1).tolist()]
    if unsigned!=[0x76543210,0xfedcba98]:raise ValueError('Unexpected INT4 packing')
    recovered=unpack_from_int32(packed,4,values.shape)
    if not torch.equal(values,recovered):raise ValueError('INT4 roundtrip failed')
    source=Path(inspect.getsourcefile(pack_to_int32)).read_bytes();source_path=a.output.with_suffix('.helpers.py');source_path.write_bytes(source)
    result={'package':'compressed-tensors','version':version,'source_sha256':hashlib.sha256(source).hexdigest(),
        'values':values.tolist(),'packed_i32':packed.tolist(),'packed_u32':unsigned,'roundtrip':True,
        'layout':'Eight offset-binary INT4 lanes per int32, low lane first; integer is nibble minus8',
        'scope':'CPU format fixture and installed primary implementation only; no actual-weight kernel or GPU benchmark'}
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
