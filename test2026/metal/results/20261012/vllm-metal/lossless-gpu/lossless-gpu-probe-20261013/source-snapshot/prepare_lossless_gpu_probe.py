"""Prepare exact BF16 files for the isolated Metal codec/matvec experiment.

No checkpoint is changed. Packing and CPU bit checks are outside GPU timing.
Usage: Python script model.safetensors output_directory [--small]
"""
import argparse,hashlib,json,mmap,struct
from pathlib import Path
import numpy as np
from probe_lossless_bf16_codec import encode,decode

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def save_case(folder,name,bits,rows,inner,finite=True):
    target=folder/name;target.mkdir(parents=True,exist_ok=False)
    names=('payload','packed','metadata','exceptions')
    files={key:(target/(key+'.bin')).open('wb') for key in names}
    raw=(target/'raw.bin').open('wb');exception_offset=0
    try:
        # Keep host memory bounded even for the complete vocabulary head.
        for start in range(0,bits.size,262144):
            part=np.array(bits[start:start+262144],dtype='<u2',copy=True)
            packed=encode(part,128)
            np.testing.assert_array_equal(decode(packed),part)
            raw.write(part.tobytes())
            packed['metadata']+=np.uint64(exception_offset)<<np.uint64(8)
            exception_offset+=packed['exceptions'].size
            for key in names:files[key].write(packed[key].tobytes())
    finally:
        raw.close()
        for stream in files.values():stream.close()
    packed_bytes=sum((target/(key+'.bin')).stat().st_size for key in names)
    case=dict(name=name,rows=rows,inner=inner,count=int(bits.size),finite=finite,
        block=128,raw_bytes=(target/'raw.bin').stat().st_size,packed_bytes=packed_bytes,
        storage_ratio=(target/'raw.bin').stat().st_size/packed_bytes,
        cpu_full_bit_roundtrip=True,
        sha256={path.name:digest(path) for path in target.glob('*.bin')})
    (target/'case.json').write_text(json.dumps(case,indent=2)+'\n')
    print(json.dumps({k:v for k,v in case.items() if k!='sha256'}),flush=True)
    return case

def main():
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint',type=Path)
    parser.add_argument('output',type=Path);parser.add_argument('--small',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);cases=[]
    patterns=np.arange(65536,dtype=np.uint16)
    cases.append(save_case(args.output,'all-bit-patterns',patterns,512,128,False))
    rng=np.random.default_rng(5178)
    cases.append(save_case(args.output,'random-tail',rng.integers(0,65536,4097,dtype=np.uint16),1,4097,False))
    with args.checkpoint.open('rb') as stream:
        length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length));base=8+length
        with mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
            names=[name for name,info in header.items() if name!='__metadata__' and
                info['dtype']=='BF16' and len(info['shape'])==2 and
                ('layers.0.mlp.down_proj.weight' in name and not name.startswith('mtp.'))]
            head='lm_head.weight' if 'lm_head.weight' in header else next(name for name in header if name.endswith('embed_tokens.weight'))
            names.append(head)
            for index,name in enumerate(names):
                info=header[name];rows,inner=info['shape'];assert inner%128==0
                if args.small:rows=min(rows,4096)
                bits=np.frombuffer(mapping,dtype='<u2',count=rows*inner,offset=base+info['data_offsets'][0])
                case=save_case(args.output,'matrix-'+str(index),bits,rows,inner)
                case['tensor']=name;cases.append(case);del bits
    (args.output/'manifest.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint),
        checkpoint_sha256=digest(args.checkpoint),small=args.small,cases=cases),indent=2)+'\n')

if __name__=='__main__':main()
