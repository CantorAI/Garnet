"""Six-bit screening index; keep original model bytes and captured native references."""
from pathlib import Path
source=Path(__file__).with_name('prepare_screened_head.py').read_text()
source=source.replace("dtype='i1',mode='w+',shape=(n,k)","dtype='u1',mode='w+',shape=(n,k*3//4)")
source=source.replace('maxima/127','maxima/31').replace('<=127','<=31')
needle='    index[start:start+len(values)]=q.astype(np.int8);meta[start:start+len(values),:3]=upper;meta[start:start+len(values),3]=scale.astype(np.float32)';assert source.count(needle)==1
source=source.replace(needle,'''    encoded=(q.astype(np.int32)+32).astype(np.uint32).reshape(len(values),k//4,4)
    words=encoded[:,:,0]|(encoded[:,:,1]<<6)|(encoded[:,:,2]<<12)|(encoded[:,:,3]<<18)
    packed=np.stack([words&255,(words>>8)&255,(words>>16)&255],axis=2).astype(np.uint8)
    index[start:start+len(values)]=packed.reshape(len(values),k*3//4)
    meta[start:start+len(values),:3]=upper;meta[start:start+len(values),3]=scale.astype(np.float32)''')
source=source.replace('INT8','INT6').replace('index_bytes=n*k,','index_bytes=n*k*3//4,')
exec(compile(source,__file__,'exec'))
