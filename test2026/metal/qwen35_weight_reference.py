"""Read selected reference weights from single-file or indexed HF checkpoints."""
import json,mmap,struct
from pathlib import Path
import numpy as np

def read_checkpoint_weights(folder,names):
    folder=Path(folder);index=folder/'model.safetensors.index.json'
    weight_map=json.loads(index.read_text())['weight_map'] if index.exists() else None
    grouped={};result={}
    for alias,name in names.items():
        filename=weight_map[name] if weight_map is not None else 'model.safetensors'
        grouped.setdefault(filename,[]).append((alias,name))
    for filename,items in grouped.items():
        with (folder/filename).open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
            length=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+length]);start=8+length
            for alias,name in items:
                info=header[name];a,b=info['data_offsets'];count=int(np.prod(info['shape']))
                if info['dtype']=='BF16':
                    data=(np.frombuffer(mapping,dtype='<u2',count=count,offset=start+a).astype(np.uint32)<<16).view(np.float32)
                elif info['dtype']=='F32':data=np.frombuffer(mapping,dtype='<f4',count=count,offset=start+a).copy()
                else:raise ValueError('Unexpected pretrained weight dtype: '+info['dtype'])
                result[alias]=data.reshape(info['shape'])
    return result
