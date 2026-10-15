"""Exact BF16 codec with four-bit exponent tags and an escape-free fast path.

This experimental representation preserves every BF16 bit pattern. It does
not quantize weights. GPU integration and model speed are separate checks.
"""
import numpy as np
FLAG=np.uint64(1)<<np.uint64(63)

def encode(bits,block=128):
    if block!=128:raise ValueError('Nibble probe requires 128-value blocks')
    count=bits.size
    bits=np.pad(bits.reshape(-1),(0,(-count)%block)).reshape(-1,block)
    exponents=((bits>>7)&255).astype(np.int16)
    hist=np.zeros((len(bits),256),np.int32)
    for i in range(block):hist[np.arange(len(bits)),exponents[:,i]]+=1
    cumulative=np.pad(hist.cumsum(axis=1),((0,0),(1,0)))
    ends=np.arange(1,257);starts=np.maximum(0,ends-15)
    bases=(cumulative[:,ends]-cumulative[:,starts]).argmax(axis=1)
    delta=bases[:,None]-exponents;escaped=(delta<0)|(delta>14)
    tags=np.where(escaped,15,delta).astype(np.uint16).reshape(-1,4)
    packed=tags[:,0]|(tags[:,1]<<4)|(tags[:,2]<<8)|(tags[:,3]<<12)
    counts=escaped.sum(axis=1).astype(np.uint64);offsets=np.cumsum(counts)-counts
    metadata=(offsets<<np.uint64(8))|bases.astype(np.uint64)|np.where(counts==0,FLAG,np.uint64(0))
    payload=((bits&127)|((bits>>8)&128)).astype(np.uint8)
    return dict(count=count,payload=payload,packed=packed,metadata=metadata,
        exceptions=exponents[escaped].astype(np.uint8))

def decode(encoded):
    packed=encoded['packed'];tags=np.stack([(packed>>shift)&15 for shift in (0,4,8,12)],axis=1).reshape(encoded['payload'].shape)
    base=(encoded['metadata']&255).astype(np.int16);exponents=base[:,None]-tags.astype(np.int16)
    mask=tags==15
    for row in range(len(tags)):
        start=int((encoded['metadata'][row]&~FLAG)>>np.uint64(8));count=int(mask[row].sum())
        exponents[row,mask[row]]=encoded['exceptions'][start:start+count]
    payload=encoded['payload'].astype(np.uint16)
    return ((payload&127)|((payload&128)<<8)|(exponents.astype(np.uint16)<<7)).reshape(-1)[:encoded['count']]
