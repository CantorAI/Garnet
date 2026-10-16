"""Derived four-bit screening index, original head values retained for exact refinement."""
from pathlib import Path
source=Path(__file__).with_name('prepare_screened_head.py').read_text()
source=source.replace("dtype='i1',mode='w+',shape=(n,k)","dtype='u1',mode='w+',shape=(n,k//2)")
source=source.replace('maxima/127','maxima/7').replace('<=127','<=7')
needle='    index[start:start+len(values)]=q.astype(np.int8);meta[start:start+len(values),:3]=upper;meta[start:start+len(values),3]=scale.astype(np.float32)';assert source.count(needle)==1
source=source.replace(needle,'''    encoded=(q.astype(np.int8)+8).astype(np.uint8)
    index[start:start+len(values)]=encoded[:,0::2]|(encoded[:,1::2]<<4)
    meta[start:start+len(values),:3]=upper;meta[start:start+len(values),3]=scale.astype(np.float32)''')
source=source.replace('INT8','INT4').replace('index_bytes=n*k,','index_bytes=n*k//2,')
exec(compile(source,__file__,'exec'))
