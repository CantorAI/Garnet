"""Exact BF16 packing feasibility; no quantization or speed claim.

Run in CPython with a checkpoint path and output JSON path. The prototype
stores 8 sign/mantissa bits, 3 exponent tags, full exponent escapes and
64-bit block metadata. Every sampled reconstructed bit must match.
"""
import json,mmap,struct,sys
from pathlib import Path
import numpy as np
BLOCK=256

def encode(bits):
    original_count=bits.size
    bits=np.pad(bits.reshape(-1), (0,(-original_count)%BLOCK)).reshape(-1,BLOCK)
    exponents=((bits>>7)&255).astype(np.int16)
    hist=np.zeros((len(bits),256),np.int32)
    for i in range(BLOCK):hist[np.arange(len(bits)),exponents[:,i]]+=1
    cumulative=np.pad(hist.cumsum(axis=1),((0,0),(1,0)))
    ends=np.arange(1,257);starts=np.maximum(0,ends-7)
    bases=(cumulative[:,ends]-cumulative[:,starts]).argmax(axis=1)
    delta=bases[:,None]-exponents;escape=(delta<0)|(delta>6)
    tags=np.where(escape,7,delta).astype(np.uint32).reshape(-1,32)
    packed=np.zeros((len(tags),3),np.uint32)
    for i in range(32):
        word,shift=divmod(i*3,32)
        packed[:,word]|=tags[:,i]<<np.uint32(shift)
        if shift>29:packed[:,word+1]|=tags[:,i]>>np.uint32(32-shift)
    counts=escape.sum(axis=1).astype(np.uint64)
    offsets=np.cumsum(counts)-counts
    metadata=(offsets<<np.uint64(8))|bases.astype(np.uint64)
    payload=((bits&127)|((bits>>8)&128)).astype(np.uint8)
    return dict(count=original_count,payload=payload,packed=packed,metadata=metadata,
        exceptions=exponents[escape].astype(np.uint8))

def decode(encoded):
    packed=encoded['packed'];tags=np.zeros((len(packed),32),np.uint32)
    for i in range(32):
        word,shift=divmod(i*3,32);value=packed[:,word]>>np.uint32(shift)
        if shift>29:value|=packed[:,word+1]<<np.uint32(32-shift)
        tags[:,i]=value&7
    tags=tags.reshape(-1,BLOCK);base=(encoded['metadata']&255).astype(np.int16)
    exponents=base[:,None]-tags.astype(np.int16);mask=tags==7
    for row in range(len(tags)):
        start=int(encoded['metadata'][row]>>np.uint64(8));count=int(mask[row].sum())
        exponents[row,mask[row]]=encoded['exceptions'][start:start+count]
    payload=encoded['payload'].astype(np.uint16)
    values=(payload&127)|((payload&128)<<8)|(exponents.astype(np.uint16)<<7)
    return values.reshape(-1)[:encoded['count']]

def check(bits):
    encoded=encode(bits);actual=decode(encoded);np.testing.assert_array_equal(bits.reshape(-1),actual)
    return sum(encoded[name].nbytes for name in ('payload','packed','metadata','exceptions'))

def main():
    # Includes both signs, zeros, subnormals, infinities and NaN payload bits.
    all_patterns=np.arange(65536,dtype=np.uint16);check(all_patterns)
    rng=np.random.default_rng(5082)
    for count in (1,31,32,33,255,256,257,4097):check(rng.integers(0,65536,count,dtype=np.uint16))
    checkpoint=Path(sys.argv[1]);results=[]
    with checkpoint.open('rb') as stream:
        length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length));base=8+length
        with mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
            for name,info in header.items():
                if name=='__metadata__' or info['dtype']!='BF16' or len(info['shape'])!=2:continue
                count=int(np.prod(info['shape']))
                if count<BLOCK:continue
                starts=rng.integers(0,count//BLOCK,size=min(128,count//BLOCK))*BLOCK
                source=np.frombuffer(mapping,dtype='<u2',count=count,offset=base+info['data_offsets'][0])
                sample=np.array(source[starts[:,None]+np.arange(BLOCK)[None,:]],copy=True).reshape(-1);del source
                packed_bytes=check(sample)
                results.append(dict(tensor=name,elements=count,sampled_weights=sample.size,
                    original_bytes=sample.nbytes,packed_bytes=packed_bytes,ratio=sample.nbytes/packed_bytes))
    estimated_bits=sum(r['elements']*16/r['ratio'] for r in results)/sum(r['elements'] for r in results)
    report=dict(scope='Exact CPU roundtrip feasibility only; no GPU codec or inference speed demonstrated',
        all_65536_bit_patterns_passed=True,tail_cases_passed=True,checkpoint=str(checkpoint),
        weighted_estimated_bits_per_weight=estimated_bits,weighted_storage_ratio=16/estimated_bits,tensors=results)
    Path(sys.argv[2]).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in report if key!='tensors'}))

if __name__=='__main__':main()
