"""Capture/compare same-engine greedy IDs before speculative reference timing."""
import argparse, hashlib, json
from pathlib import Path
from urllib.request import Request, urlopen
from benchmark_stream import payload
from prepare_vlm_reference import reference_tokens


def compare_pair(left, right):
    if left['engine'] != right['engine']:
        raise ValueError('Speculation must be checked against the same reference engine')
    for field in ('key', 'semantic_request_sha256'):
        if left[field] != right[field]:
            raise ValueError('Reference requests differ: '+field)
    return bool(left['generated_ids']) and left['prompt_ids'] == right['prompt_ids'] and left['generated_ids'] == right['generated_ids']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--engine', choices=['vllm','sglang'], required=True)
    p.add_argument('--url', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ordinary-oracle', type=Path)
    a=p.parse_args();out=a.output
    if out.exists() and any(out.iterdir()):raise FileExistsError('Preserve previous reference attempt')
    out.mkdir(parents=True,exist_ok=True)
    prior=None
    if a.ordinary_oracle:
        prior=json.loads(a.ordinary_oracle.read_text(encoding='utf8'))
        if prior['engine']!=a.engine or prior['mode']!='ordinary_capture' or prior['status']!='passed' or len(prior['cases'])!=16:
            raise ValueError('A complete same-engine ordinary capture is required')
    report={'engine':a.engine,'mode':'speculative_compare' if prior else 'ordinary_capture',
            'status':'running','cases':[],
            'scope':'Sixteen same-engine full greedy sequences only; verify actual server configuration separately. Not broad quality or timing.'}
    if prior:report['ordinary_oracle_sha256']=hashlib.sha256(a.ordinary_oracle.read_bytes()).hexdigest()
    path=out/'equivalence.json'
    try:
        for case_index,case in enumerate(['text','image']):
            for i in range(8):
                key='s3-'+case+'-c1-r'+str(i)
                body,_=payload(case,key,301000+case_index*20000+i,'qwen38-27b-nvfp4',128,'jpeg','image-first','binary-grid')
                body['stream']=False
                digest=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                body['return_token_ids']=True
                if a.engine=='sglang':body['return_prompt_token_ids']=True
                wire=json.dumps(body,separators=(',',':')).encode()
                (out/(key+'-request.json')).write_bytes(wire)
                with urlopen(Request(a.url+'/v1/chat/completions',data=wire,headers={'Content-Type':'application/json'}),timeout=300) as response:raw=response.read()
                (out/(key+'-response.json')).write_bytes(raw)
                result=json.loads(raw);ids,generated=reference_tokens(result,a.engine)
                if len(ids)!=(41 if case=='text' else 104):raise ValueError('Unexpected actual reference prompt count')
                if result['usage']['completion_tokens']!=len(generated):raise ValueError('Actual generated IDs/count mismatch')
                row={'engine':a.engine,'key':key,'semantic_request_sha256':digest,'prompt_ids':ids,'generated_ids':generated,
                     'request_sha256':hashlib.sha256(wire).hexdigest(),'response_sha256':hashlib.sha256(raw).hexdigest()}
                if prior:row['exact_match']=compare_pair(prior['cases'][len(report['cases'])],row)
                report['cases'].append(row);path.write_text(json.dumps(report,indent=2),encoding='utf8')
                if prior and not row['exact_match']:raise ValueError('Speculative full-ID mismatch: '+key)
                print('REFERENCE_GREEDY_CAPTURE',key,len(generated),flush=True)
        report['status']='passed'
    except Exception as error:
        report['status']='failed';report['error']=repr(error);raise
    finally:path.write_text(json.dumps(report,indent=2),encoding='utf8')


if __name__=='__main__':main()
