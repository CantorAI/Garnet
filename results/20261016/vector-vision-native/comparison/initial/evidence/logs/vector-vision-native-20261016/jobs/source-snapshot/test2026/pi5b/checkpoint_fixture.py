import json,struct
import numpy as np
def write_checkpoint(path,weights,suffixes,prefix,fp32_names=()):
    header={};payload=bytearray();decoded={}
    for name,weight in weights.items():
        dtype='F32' if name in fp32_names else 'BF16'
        if dtype=='BF16':
            bits=weight.astype(np.float32).view(np.uint32)
            bf16=((bits+0x7fff+((bits>>16)&1))>>16).astype('<u2')
            decoded[name]=(bf16.astype(np.uint32)<<16).view(np.float32)
            raw=bf16.tobytes()
        else:
            decoded[name]=weight.astype(np.float32);raw=decoded[name].astype('<f4').tobytes()
        header[prefix+'.'+suffixes[name]]=dict(dtype=dtype,shape=list(weight.shape),data_offsets=[len(payload),len(payload)+len(raw)])
        payload.extend(raw)
    encoded=json.dumps(header,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%8)
    path.write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)
    return decoded

def read_checkpoint(path,suffixes,prefix):
    raw=path.read_bytes();length=struct.unpack('<Q',raw[:8])[0]
    header=json.loads(raw[8:8+length]);data_start=8+length;weights={}
    for name,suffix in suffixes.items():
        entry=header[prefix+'.'+suffix];start,end=entry['data_offsets'];data=raw[data_start+start:data_start+end]
        if entry['dtype']=='BF16':array=(np.frombuffer(data,dtype='<u2').astype(np.uint32)<<16).view(np.float32)
        elif entry['dtype']=='F32':array=np.frombuffer(data,dtype='<f4')
        else:raise ValueError('unexpected checkpoint dtype')
        weights[name]=array.reshape(entry['shape'])
    return weights
