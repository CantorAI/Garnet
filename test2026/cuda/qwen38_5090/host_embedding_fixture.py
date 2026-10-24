"""Prepare independent embedding-row bytes; CPU only, no model/GPU timing.

Usage: python host_embedding_fixture.py <checkpoint-directory> <new-output-directory>
"""
from pathlib import Path
import hashlib,json,math,random,struct,sys

NAME='model.language_model.embed_tokens.weight'

def prepare(checkpoint,output):
    checkpoint=Path(checkpoint).resolve(strict=True);output=Path(output)
    index=checkpoint/'model.safetensors.index.json'
    if index.exists():
        index_raw=index.read_bytes();filename=json.loads(index_raw)['weight_map'][NAME]
    else:index_raw=None;filename='model.safetensors'
    shard=(checkpoint/filename).resolve(strict=True)
    if not shard.is_relative_to(checkpoint):raise ValueError('Embedding shard escapes checkpoint directory')
    with shard.open('rb') as source:
        prefix=source.read(8)
        if len(prefix)!=8:raise ValueError('Missing safetensors header length')
        size=struct.unpack('<Q',prefix)[0]
        if not 2<=size<=32*1024**2:raise ValueError('Invalid/beyond-bound safetensors header')
        header_raw=source.read(size)
        if len(header_raw)!=size:raise ValueError('Truncated safetensors header')
        entry=json.loads(header_raw)[NAME]
        shape=entry['shape'];offsets=entry['data_offsets']
        if (entry['dtype']!='BF16' or len(shape)!=2 or any(type(x) is not int for x in shape)
                or not 2<=shape[0]<=2**31-1 or not 1<=shape[1]<=16384):
            raise ValueError('Embedding needs bounded BF16 [vocabulary, hidden]')
        vocab,width=shape
        if (len(offsets)!=2 or any(type(x) is not int for x in offsets) or offsets[0]<0
                or offsets[1]-offsets[0]!=vocab*width*2 or 8+size+offsets[1]>shard.stat().st_size):
            raise ValueError('Embedding byte geometry exceeds shard')
        rng=random.Random(5090);ids=[0,vocab-1,1,0]+[rng.randrange(vocab) for _ in range(12)]
        captured={}
        for token in dict.fromkeys(ids):
            source.seek(8+size+offsets[0]+token*width*2);raw=source.read(width*2)
            if len(raw)!=width*2:raise ValueError('Embedding row truncated')
            # Independent byte placement oracle: BF16 occupies high half of F32.
            f32=b''.join(b'\x00\x00'+raw[i:i+2] for i in range(0,len(raw),2))
            if not all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',f32)):
                raise ValueError('Embedding oracle requires finite checkpoint rows')
            captured[token]=(raw,f32)
    output.mkdir(parents=True,exist_ok=False)
    rows={}
    for token,(raw,f32) in captured.items():
        a=f'row-{token}.bf16';b=f'row-{token}.f32'
        (output/a).write_bytes(raw);(output/b).write_bytes(f32)
        rows[str(token)]={'bf16':a,'f32':b,'bf16_sha256':hashlib.sha256(raw).hexdigest(),'f32_sha256':hashlib.sha256(f32).hexdigest()}
    report={'status':'checkpoint_embedding_rows_prepared_cpu_only','tensor':NAME,'shape':shape,'ids':ids,'rows':rows,
        'checkpoint':str(checkpoint),'shard':filename,'shard_bytes':shard.stat().st_size,
        'header_sha256':hashlib.sha256(prefix+header_raw).hexdigest(),
        'index_sha256':hashlib.sha256(index_raw).hexdigest() if index_raw is not None else None,
        'entry':entry,'gpu_qualified':False,'scope':'Selected boundary, duplicate and seeded rows only; no whole-checkpoint hash or model/numerical/performance admission.'}
    (output/'fixture.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    if len(sys.argv)!=3:raise SystemExit(__doc__)
    print(json.dumps(prepare(sys.argv[1],sys.argv[2]),indent=2))
