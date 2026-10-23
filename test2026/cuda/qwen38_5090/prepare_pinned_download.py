"""Prepare verified public metadata and optional ephemeral CDN transport for study pins."""
import argparse,hashlib,json,zipfile
from pathlib import Path
from urllib.request import urlopen
from download_pinned import PINS

def prepare(model,prefix,transport=False):
    revision,budget=PINS[model]
    api='https://huggingface.co/api/models/'+model+'/revision/'+revision+'?blobs=true'
    with urlopen(api,timeout=60) as response:raw=response.read(4*1024*1024+1)
    if len(raw)>4*1024*1024:raise ValueError('Oversized model metadata')
    metadata=json.loads(raw)
    if metadata['sha']!=revision:raise ValueError('Wrong checkpoint revision')
    expected=next(x for x in json.loads((Path(__file__).parent/'evidence/checkpoints.json').read_text())['checkpoints'] if x['id']==model)
    tasks=[];small=[];skipped=[];base='https://huggingface.co/'+model+'/resolve/'+revision+'/'
    with zipfile.ZipFile(str(prefix)+'-metadata.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for item in metadata['siblings']:
            name=item['rfilename']
            if Path(name).name!=name:
                if name.endswith('.safetensors'):raise ValueError('Nested checkpoint shards need an explicit layout')
                skipped.append(name);continue
            if item.get('lfs'):
                info={'name':name,'sha256':item['lfs']['sha256'],'bytes':item['size']}
                if transport:
                    with urlopen(base+name,timeout=60) as response:info['url']=response.geturl()
                tasks.append(info)
            else:
                with urlopen(base+name,timeout=60) as response:data=response.read(8*1024*1024+1)
                if len(data)!=item['size'] or len(data)>8*1024*1024:raise ValueError('Metadata byte count mismatch')
                sha=hashlib.sha256(data).hexdigest()
                if name=='config.json' and sha!=expected['sha256']:raise ValueError('Pinned configuration differs from audited evidence')
                archive.writestr(name,data);small.append({'name':name,'sha256':sha,'bytes':len(data)})
        archive.writestr('metadata-provenance.json',json.dumps({'model':model,'revision':revision,'files':small,'skipped_nonruntime_nested_files':skipped},indent=2))
    total=sum(x['bytes'] for x in tasks)
    if total>budget:raise ValueError('Checkpoint exceeds its download budget')
    public={'model':model,'revision':revision,'lfs_files':[{k:v for k,v in x.items() if k!='url'} for x in tasks],
        'metadata_files':small,'api_metadata_sha256':hashlib.sha256(raw).hexdigest(),'total_lfs_bytes':total,
        'scope':'Primary repository bytes/metadata only; runtime, memory admission and performance unproven'}
    Path(str(prefix)+'-artifact-audit.json').write_text(json.dumps(public,indent=2))
    if transport:Path(str(prefix)+'-transport.json').write_text(json.dumps({'model':model,'revision':revision,'files':tasks}))
    print(json.dumps({'model':model,'revision':revision,'lfs_files':len(tasks),'lfs_bytes':total,'metadata_bytes':sum(x['bytes'] for x in small),'transport_prepared':transport}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',choices=sorted(PINS));p.add_argument('--output-prefix',type=Path,required=True);p.add_argument('--transport',action='store_true');a=p.parse_args()
    prepare(a.model,a.output_prefix,a.transport)
