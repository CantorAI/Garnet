"""Verify extracted metadata separately from already verified checkpoint payloads."""
import argparse,hashlib,json
from pathlib import Path
from download_pinned import PINS

def verify(root):
    payload=json.loads((root/'download-provenance.json').read_text())
    metadata=json.loads((root/'metadata-provenance.json').read_text())
    pin=PINS.get(payload['model'])
    if pin is None or payload['revision']!=pin[0]:raise ValueError('Unknown checkpoint pin')
    if any(metadata.get(k)!=payload[k] for k in ['model','revision']):raise ValueError('Metadata/payload identity mismatch')
    files=metadata['files']
    for item in files:
        name=item['name']
        if Path(name).name!=name or name in ['.','..']:raise ValueError('Unsafe metadata path')
        raw=(root/name).read_bytes()
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Metadata mismatch: '+name)
    return {'model':payload['model'],'revision':payload['revision'],'verified_metadata_files':len(files),
        'metadata_provenance_sha256':hashlib.sha256((root/'metadata-provenance.json').read_bytes()).hexdigest(),
        'download_provenance_sha256':hashlib.sha256((root/'download-provenance.json').read_bytes()).hexdigest(),
        'scope':'Extracted small metadata integrity; payload verification is recorded by the downloader, not rerun here'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('models',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    result=[verify(root) for root in a.models];a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
