"""Bounded primary-header audit of the pinned compressed-tensors W4A16 candidate."""
import argparse,hashlib,json,math,struct
from pathlib import Path
from urllib.request import urlopen
from inspect_nvfp4 import read_range,projection_shapes
MODEL='pearsonkyle/Qwen3.8-27B-GPTQ-W4A16'
REVISION='4e2c2d8f592d5641beefa6a75357f43cf3756392'
BYTES={'BF16':2,'F16':2,'F32':4,'I32':4,'I64':8}

def validate(header,name,shape,dtypes):
    value=header[name];offsets=value['data_offsets']
    if value['shape']!=list(shape) or value['dtype'] not in dtypes or offsets[1]-offsets[0]!=math.prod(shape)*BYTES[value['dtype']]:
        raise ValueError('Wrong INT4 tensor layout: '+name)

def audit(config,index,headers):
    q=config['quantization_config'];group=q['config_groups']['group_0'];scheme=group['weights']
    if q['quant_method']!='compressed-tensors' or group['format']!='pack-quantized' or scheme['num_bits']!=4 or scheme['group_size']!=128 or not scheme['symmetric'] or group['input_activations'] is not None:raise ValueError('Wrong candidate recipe')
    records=[]
    for prefix,(n,k) in projection_shapes(config).items():
        key=prefix+'.weight_packed'
        if key in index['weight_map']:
            file=index['weight_map'][key];h=headers[file]
            if k%128:raise ValueError('Unaligned group128 projection')
            validate(h,key,(n,k//8),{'I32'});validate(h,prefix+'.weight_scale',(n,k//128),{'BF16','F16','F32'});validate(h,prefix+'.weight_shape',(2,),{'I64'})
            records.append({'prefix':prefix,'logical_shape':[n,k],'algorithm':'INT4_W4A16_G128','file':file,'scale_dtype':h[prefix+'.weight_scale']['dtype']})
        else:
            key=prefix+'.weight';file=index['weight_map'][key];validate(headers[file],key,(n,k),{'BF16'})
            records.append({'prefix':prefix,'logical_shape':[n,k],'algorithm':'BF16','file':file})
    return records

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    base='https://huggingface.co/'+MODEL+'/resolve/'+REVISION+'/'
    metadata={}
    for name in ('config.json','model.safetensors.index.json'):
        with urlopen(base+name,timeout=60) as r:data=r.read(4*1024*1024+1)
        if len(data)>4*1024*1024:raise ValueError('Oversized metadata')
        (a.output/name).write_bytes(data);metadata[name]=json.loads(data)
    expected=next(x for x in json.loads((Path(__file__).parent/'evidence/checkpoints.json').read_text())['checkpoints'] if x['id']==MODEL)
    if hashlib.sha256((a.output/'config.json').read_bytes()).hexdigest()!=expected['sha256']:raise ValueError('Pinned config differs')
    index=metadata['model.safetensors.index.json'];headers={};sources=[]
    for file in sorted(set(index['weight_map'].values())):
        if Path(file).name!=file or not file.endswith('.safetensors'):raise ValueError('Unsafe shard filename')
        prefix,total=read_range(base+file,0,8);size=struct.unpack('<Q',prefix)[0]
        if not 2<=size<=4*1024*1024 or size+8>total:raise ValueError('Invalid header size')
        raw,total2=read_range(base+file+'?header_audit=1',8,size)
        if total!=total2:raise ValueError('Shard changed')
        header=json.loads(raw)
        for name,value in header.items():
            if name=='__metadata__':continue
            begin,end=value['data_offsets']
            if not 0<=begin<=end<=total-size-8:raise ValueError('Out-of-range tensor')
            if value['dtype'] not in BYTES or end-begin!=math.prod(value['shape'])*BYTES[value['dtype']]:raise ValueError('Tensor byte size mismatch')
            if index['weight_map'].get(name)!=file:raise ValueError('Header/index mismatch')
        (a.output/(file+'.header.json')).write_bytes(raw);headers[file]=header
        shape_entries={name:v for name,v in header.items() if name.endswith('.weight_shape')}
        shape_proof=None
        if shape_entries:
            begin=min(v['data_offsets'][0] for v in shape_entries.values());end=max(v['data_offsets'][1] for v in shape_entries.values())
            if end-begin>4*1024*1024:raise ValueError('Shape metadata proof exceeds bounded budget')
            payload,total3=read_range(base+file+'?shape_audit=1',size+8+begin,end-begin)
            if total3!=total:raise ValueError('Shard changed')
            values={name:list(struct.unpack_from('<qq',payload,v['data_offsets'][0]-begin)) for name,v in shape_entries.items()}
            (a.output/(file+'.shapes.json')).write_text(json.dumps(values,indent=2));shape_proof={'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}
        sources.append({'file':file,'file_bytes':total,'header_bytes':size,'header_sha256':hashlib.sha256(raw).hexdigest(),'shape_metadata_proof':shape_proof})
    records=audit(metadata['config.json'],index,headers)
    for record in records:
        if record['algorithm'].startswith('INT4'):
            shapes=json.loads((a.output/(record['file']+'.shapes.json')).read_text())
            if shapes[record['prefix']+'.weight_shape']!=record['logical_shape']:raise ValueError('Stored logical shape mismatch')
    report={'model':MODEL,'revision':REVISION,'projections':records,'sources':sources,
        'algorithms':{kind:sum(r['algorithm']==kind for r in records) for kind in ('INT4_W4A16_G128','BF16')},
        'extra_quantized_metadata_names':[name for h in headers.values() for name in h if 'g_idx' in name or 'zero_point' in name],
        'scope':'Primary headers/shape metadata only; packed values, scales, numerical kernels, GPU fit and inference unproven'}
    (a.output/'header-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({'algorithms':report['algorithms'],'extra_metadata':report['extra_quantized_metadata_names'],'scope':report['scope']}))
