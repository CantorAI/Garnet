"""Audit pinned ModelOpt safetensors headers with bounded HTTP range requests.

No weight payloads are downloaded. Header verification does not establish scale
values, numerical correctness, engine support or measured memory consumption.
"""
import argparse, hashlib, json, math, re, struct, sys
from pathlib import Path
from urllib.request import Request, urlopen
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from xModel.qwen38.vl_27b.model import projection_shapes, validate_config

MODEL = 'nvidia/Qwen3.8-27B-NVFP4'
REVISION = '482ca0f3832238542f8f5295dde86b5f22711d80'
DTYPE_BYTES = {'BF16':2,'F32':4,'F8_E4M3':1,'F8_E4M3FN':1,'U8':1}


def read_range(url, start, size):
    request = Request(url,headers={'Range':'bytes='+str(start)+'-'+str(start+size-1)})
    with urlopen(request,timeout=30) as response:
        content_range = response.headers.get('Content-Range','')
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',content_range)
        if response.status != 206 or not match or int(match[1]) != start or int(match[2]) != start+size-1:
            raise ValueError('Server did not honor bounded range: '+content_range)
        data = response.read(size+1)
        if len(data) != size: raise ValueError('Truncated/oversized range response')
        return data, int(match[3])


def validate_tensor(header, name, shape, dtypes):
    value = header.get(name)
    if not isinstance(value,dict) or value.get('shape') != list(shape) or value.get('dtype') not in dtypes:
        raise ValueError('Tensor dtype/shape mismatch: '+name)
    offsets = value.get('data_offsets',[])
    size = math.prod(shape)*DTYPE_BYTES[value['dtype']]
    if len(offsets) != 2 or any(isinstance(v,bool) or not isinstance(v,int) or v < 0 for v in offsets) or offsets[1]-offsets[0] != size:
        raise ValueError('Tensor byte span mismatch: '+name)
    return size


def validate_projection(header, prefix, shape, algorithm):
    rows, columns = shape
    if algorithm == 'NVFP4':
        if columns%16: raise ValueError('NVFP4 columns must be block16 aligned')
        validate_tensor(header,prefix+'.weight',(rows,columns//2),{'U8'})
        validate_tensor(header,prefix+'.weight_scale',(rows,columns//16),{'F8_E4M3','F8_E4M3FN'})
        validate_tensor(header,prefix+'.weight_scale_2',(),{'F32'})
        validate_tensor(header,prefix+'.input_scale',(),{'F32'})
    elif algorithm == 'FP8':
        validate_tensor(header,prefix+'.weight',shape,{'F8_E4M3','F8_E4M3FN'})
        validate_tensor(header,prefix+'.weight_scale',(),{'F32'})
        validate_tensor(header,prefix+'.input_scale',(),{'F32'})
    elif algorithm == 'BF16': validate_tensor(header,prefix+'.weight',shape,{'BF16'})
    else: raise ValueError('Unverified projection algorithm: '+str(algorithm))
    return {'name':prefix,'algorithm':algorithm,'logical_shape':list(shape)}


def audit_headers(config, index, headers):
    validate_config(config)
    quant = config.get('quantization_config',{})
    if quant.get('quant_method') != 'modelopt' or quant.get('quant_algo') != 'MIXED_PRECISION':
        raise ValueError('Expected pinned mixed ModelOpt recipe')
    layers = quant['quantized_layers']
    records = []
    for prefix, shape in projection_shapes(config).items():
        shard = index['weight_map'][prefix+'.weight']
        algorithm = layers.get(prefix,{}).get('quant_algo','BF16')
        records.append(validate_projection(headers[shard],prefix,shape,algorithm))
    return {'projection_count':len(records), 'algorithms':{a:sum(r['algorithm']==a for r in records) for a in ('NVFP4','FP8','BF16')},
            'indexed_tensor_bytes':index['metadata']['total_size'],'projections':records,
            'scope':'Tensor headers only; scale payloads, kernel support, GPU fit and inference unverified'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True,exist_ok=True)
    base = 'https://huggingface.co/'+MODEL+'/resolve/'+REVISION+'/'
    metadata = {}
    for name in ('config.json','model.safetensors.index.json'):
        with urlopen(base+name,timeout=30) as r: data = r.read(4*1024*1024+1)
        if len(data) > 4*1024*1024: raise ValueError('Metadata too large')
        (out/name).write_bytes(data)
        metadata[name] = json.loads(data)
    index = metadata['model.safetensors.index.json']
    headers, sources = {}, []
    for shard in sorted(set(index['weight_map'].values())):
        if Path(shard).name != shard or not shard.endswith('.safetensors'): raise ValueError('Unsafe shard path')
        size_data, total = read_range(base+shard,0,8)
        size = struct.unpack('<Q',size_data)[0]
        if not 2 <= size <= 4*1024*1024 or size+8 > total: raise ValueError('Invalid header size')
        data, total2 = read_range(base+shard+'?header_only=1',8,size)
        if total2 != total: raise ValueError('Shard changed during inspection')
        header = json.loads(data)
        for key, value in header.items():
            if key == '__metadata__': continue
            offsets = value.get('data_offsets',[])
            if len(offsets) != 2 or not 0 <= offsets[0] <= offsets[1] <= total-size-8:
                raise ValueError('Tensor exceeds shard: '+key)
        headers[shard] = header
        filename = shard+'.header.json'
        (out/filename).write_bytes(data)
        sources.append({'file':filename,'url':base+shard,'header_sha256':hashlib.sha256(data).hexdigest(),'header_bytes':size,'shard_bytes':total})
    report = audit_headers(metadata['config.json'],index,headers)
    report.update(model=MODEL,revision=REVISION,sources=sources)
    (out/'header-audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('projections','sources')},indent=2))


if __name__ == '__main__': main()
