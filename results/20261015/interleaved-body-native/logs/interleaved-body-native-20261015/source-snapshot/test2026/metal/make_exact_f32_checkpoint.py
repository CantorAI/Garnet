"""Expand finite BF16 weights to numerically identical F32 in a new folder.

This changes storage/dtype, never learned values. Original checkpoints are
read-only. Every destination tensor is verified bit-for-bit against its
source's exact F32 values before the checkpoint is published in that folder.
"""
import hashlib,json,mmap,shutil,struct,sys
from pathlib import Path
import numpy as np

source=Path(sys.argv[1]).resolve();destination=Path(sys.argv[2]).resolve()
checkpoint=source/'model.safetensors';chunk=4*1024*1024
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as original:
    size=struct.unpack_from('<Q',original)[0];header=json.loads(original[8:8+size]);base=8+size
    tensors=sorted((name for name in header if name!='__metadata__'),key=lambda name:header[name]['data_offsets'][0])
    expanded_header={};offset=0;source_end=0
    if '__metadata__' in header:expanded_header['__metadata__']=header['__metadata__']
    for name in tensors:
        info=header[name];assert info['dtype'] in ('BF16','F32'),(name,info['dtype'])
        count=1
        for dim in info['shape']:assert dim>0;count*=dim
        start,end=info['data_offsets'];width=2 if info['dtype']=='BF16' else 4
        assert start>=source_end and end-start==count*width and base+end<=len(original),name
        source_end=end
        expanded_header[name]=dict(dtype='F32',shape=info['shape'],data_offsets=[offset,offset+4*count]);offset+=4*count
    assert shutil.disk_usage(destination.parent if destination.parent.exists() else source.parent).free>offset+2*1024**3,'Insufficient free space'
    destination.mkdir(parents=True,exist_ok=False)
    encoded=json.dumps(expanded_header,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%8)
    partial=destination/'model.safetensors.incomplete';proof=[]
    with partial.open('xb') as target:
        target.write(struct.pack('<Q',len(encoded)));target.write(encoded)
        for name in tensors:
            info=header[name];start,end=info['data_offsets'];width=2 if info['dtype']=='BF16' else 4
            count=(end-start)//width;expected_hash=hashlib.sha256()
            for first in range(0,count,chunk):
                n=min(chunk,count-first);raw=original[base+start+first*width:base+start+(first+n)*width]
                if width==2:
                    values=np.frombuffer(raw,dtype='<u2')
                    assert not np.any((values&0x7f80)==0x7f80),name
                    bits=values.astype('<u4')<<16;data=bits.tobytes()
                else:
                    bits=np.frombuffer(raw,dtype='<u4')
                    assert not np.any((bits&0x7f800000)==0x7f800000),name
                    data=raw
                expected_hash.update(data);target.write(data)
            proof.append(dict(name=name,shape=info['shape'],source_dtype=info['dtype'],elements=count,
                expected_f32_values_sha256=expected_hash.hexdigest()))
    with partial.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as converted:
        converted_base=8+struct.unpack_from('<Q',converted)[0]
        for entry in proof:
            name=entry['name'];info=header[name];start=info['data_offsets'][0];width=2 if info['dtype']=='BF16' else 4
            target_start=expanded_header[name]['data_offsets'][0];actual_hash=hashlib.sha256()
            for first in range(0,entry['elements'],chunk):
                n=min(chunk,entry['elements']-first)
                a=np.frombuffer(original[base+start+first*width:base+start+(first+n)*width],dtype='<u2' if width==2 else '<u4')
                if width==2:a=a.astype('<u4')<<16
                raw=converted[converted_base+target_start+first*4:converted_base+target_start+(first+n)*4]
                b=np.frombuffer(raw,dtype='<u4');assert np.array_equal(a,b),name
                actual_hash.update(raw)
            entry['actual_f32_values_sha256']=actual_hash.hexdigest()
            assert entry['expected_f32_values_sha256']==entry['actual_f32_values_sha256'],name
            entry['all_values_bit_exact']=True
    partial.rename(destination/'model.safetensors')
for name in ('tokenizer.json','tokenizer_config.json','generation_config.json','preprocessor_config.json','special_tokens_map.json','vocab.json','merges.txt'):
    if (source/name).is_file():shutil.copyfile(source/name,destination/name)
config=json.loads((source/'config.json').read_text());config['dtype']='float32'
if 'torch_dtype' in config:config['torch_dtype']='float32'
for key in ('text_config','vision_config'):
    if isinstance(config.get(key),dict):
        config[key]['dtype']='float32'
        if 'torch_dtype' in config[key]:config[key]['torch_dtype']='float32'
(destination/'config.json').write_text(json.dumps(config,indent=2)+'\n')
report=dict(scope='Lossless BF16-to-F32 storage expansion; no quantization or learned-value changes',
    source_checkpoint=str(checkpoint),destination_checkpoint=str(destination/'model.safetensors'),
    source_sha256=digest(checkpoint),destination_sha256=digest(destination/'model.safetensors'),
    converter_sha256=digest(Path(__file__)),tensor_count=len(proof),elements=sum(p['elements'] for p in proof),
    all_values_bit_exact=True,tensors=proof)
(destination/'exact-value-equivalence.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('tensor_count','elements','all_values_bit_exact','source_sha256','destination_sha256')}),flush=True)
