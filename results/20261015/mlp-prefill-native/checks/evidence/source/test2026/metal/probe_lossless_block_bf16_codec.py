"""Exact 12-bit blocks with whole-block raw fallback; no per-value escapes."""
import numpy as np
FLAG=np.uint64(1)<<np.uint64(63)

def encode(bits,block=128):
    if block!=128:raise ValueError('128-value blocks required')
    count=bits.size
    values=np.pad(np.asarray(bits,dtype='<u2').reshape(-1),(0,(-count)%128)).reshape(-1,128)
    exp=((values>>7)&255).astype(np.uint16)
    base=exp.max(axis=1);compact=base-exp.min(axis=1)<=15
    sizes=np.where(compact,192,256).astype(np.uint64)
    offsets=np.cumsum(sizes)-sizes;payload=np.empty(int(sizes.sum()),np.uint8)
    idx=np.flatnonzero(compact)
    if len(idx):
        low=((values[idx]&127)|((values[idx]>>8)&128)).astype(np.uint8)
        delta=(base[idx,None]-exp[idx]).reshape(-1,32,4)
        tags=(delta[:,:,0]|(delta[:,:,1]<<4)|(delta[:,:,2]<<8)|(delta[:,:,3]<<12)).astype('<u2')
        packed=np.concatenate((low,tags.view(np.uint8).reshape(-1,64)),axis=1)
        payload[offsets[idx,None]+np.arange(192,dtype=np.uint64)]=packed
    idx=np.flatnonzero(~compact)
    if len(idx):payload[offsets[idx,None]+np.arange(256,dtype=np.uint64)]=values[idx].view(np.uint8).reshape(-1,256)
    metadata=(offsets<<np.uint64(8))|base.astype(np.uint64)|np.where(compact,FLAG,np.uint64(0))
    return dict(count=count,payload=payload,metadata=metadata,packed=np.empty(0,'<u2'),exceptions=np.empty(0,np.uint8))

def decode(encoded):
    meta=encoded['metadata'];offsets=(meta&~FLAG)>>np.uint64(8)
    compact=(meta&FLAG)!=0;values=np.empty((len(meta),128),'<u2');payload=encoded['payload']
    idx=np.flatnonzero(compact)
    if len(idx):
        low=payload[offsets[idx,None]+np.arange(128,dtype=np.uint64)].astype(np.uint16)
        raw_tags=payload[offsets[idx,None]+128+np.arange(64,dtype=np.uint64)]
        tags=raw_tags.copy().view('<u2').reshape(-1,32)
        delta=np.stack([(tags>>shift)&15 for shift in (0,4,8,12)],axis=2).reshape(-1,128)
        exponents=(meta[idx,None]&255).astype(np.uint16)-delta
        values[idx]=(low&127)|((low&128)<<8)|(exponents<<7)
    idx=np.flatnonzero(~compact)
    if len(idx):values[idx]=payload[offsets[idx,None]+np.arange(256,dtype=np.uint64)].copy().view('<u2').reshape(-1,128)
    return values.reshape(-1)[:encoded['count']]
