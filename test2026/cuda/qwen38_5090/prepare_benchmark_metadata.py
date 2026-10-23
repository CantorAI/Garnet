"""Capture installed reference/configuration provenance before real measurements."""
import argparse,datetime,hashlib,json,subprocess
from pathlib import Path
from run_logged import source_identity

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('engine',choices=['vllm','sglang']);p.add_argument('--label',required=True);a=p.parse_args()
    study=Path('D:/CantorAI/5090-study-20261023');refs=study/'results/references'/a.engine
    model=study/'models/Qwen3.8-27B-NVFP4';source=Path(__file__).parent
    paths=[refs/'freeze-cuda13.0.txt',refs/'nvcc-cuda13.0.txt',source/('serve_'+a.engine+'.sh'),model/'config.json',model/'model.safetensors.index.json',model/'download-provenance.json']
    records={}
    for path in paths:
        data=path.read_bytes();records[str(path)]={'sha256':hashlib.sha256(data).hexdigest(),'text':data.decode('utf-8')}
    for name in ('cuda-layout.json','cuda-cublas-layout.json'):
        path=refs/name
        if path.is_file():records[str(path)]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'json':json.loads(path.read_text())}
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'engine':a.engine,'configuration_label':a.label,
        'source':source_identity(),'files':records,'nvidia_smi_xml':subprocess.check_output(['nvidia-smi','-q','-x'],text=True),
        'scope':'Installed source/configuration provenance. Does not prove server health, optimized configuration or equivalence with Garnet.'}
    output=refs/'benchmark-metadata.json'
    if output.exists():raise ValueError('Preserve previous metadata before writing a new configuration')
    output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(str(output))
