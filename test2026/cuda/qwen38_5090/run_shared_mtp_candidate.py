"""Versioned shared-profile golden gate; historical T16 gate remains unchanged."""
import hashlib,json,subprocess,sys
from pathlib import Path
GOLDEN_SHA256='634e6200658aa74467b575c8d4b5e8dde58bb7522d4de1e5ca813da7576acf4b'
COLORS={'red':'red','green':'green','blue':'blue','bottom-blue':'blue'}
KEYS=list(COLORS)+['s3-'+case+'-c1-r'+str(i) for case in ('text','image') for i in range(8)]

def verify_records(golden,records):
    if (golden.get('status'),golden.get('batch'),golden.get('capacity'),golden.get('prefill_chunk'),golden.get('decode_chunk'))!=('smoke_passed',1,4096,128,4):raise ValueError('Wrong golden profile')
    if set(records)!=set(KEYS):raise ValueError('All twenty target/MTP pairs required')
    for key,pair in records.items():
        teacher,trial=pair
        a,b=teacher['choices'][0],trial['choices'][0]
        if not a['response_token_ids'] or a['response_token_ids']!=b['response_token_ids'] or a['prompt_token_ids']!=b['prompt_token_ids']:raise ValueError('Target/MTP token mismatch: '+key)
        if teacher.get('study',{}).get('generation_mode')!='greedy_target' or trial.get('study',{}).get('generation_mode')!='verified_mtp' or not isinstance(trial.get('study',{}).get('speculative_metrics'),dict):raise ValueError('Actual target/MTP execution required')
        if key in COLORS:
            if any(r['choices'][0]['message']['content'].strip().lower().rstrip('.')!=COLORS[key] for r in pair):raise ValueError('VLM semantic failure')
        elif len(b['response_token_ids'])!=128:raise ValueError('Full128 output required')
    names=list(COLORS)+['s3-text-c1-r0-greedy','s3-image-c1-r0-greedy']
    if len(golden['cases'])!=12 or {row['key'] for row in golden['cases']}!=set(names):raise ValueError('Complete retained golden required')
    for name in names:
        rows=sorted((row for row in golden['cases'] if row['key']==name),key=lambda row:row['repeat'])
        if [row['repeat'] for row in rows]!=[0,1] or any(rows[0][field]!=rows[1][field] for field in ('prompt_ids','generated_ids','normalized_sha256')):raise ValueError('Retained golden repeats differ')
        key=name.removesuffix('-greedy')
        for response in records[key]:
            choice=response['choices'][0]
            if choice['prompt_token_ids']!=rows[0]['prompt_ids'] or choice['response_token_ids']!=rows[0]['generated_ids']:raise ValueError('Retained same-profile golden mismatch: '+name)
    return {'target_mtp_pairs':20,'full128_pairs':16,'retained_profile_goldens':6,'vlm_semantic_cases':4}

def main():
    root,out=map(Path,sys.argv[1:3]);out.mkdir(exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Preserve candidate attempt')
    sys.path.insert(0,str(root/'Garnet/test2026/cuda/qwen38_5090'))
    from validate_native_mtp_http import validate_worker_profile
    from benchmark_stream import payload,request
    from run_logged import source_identity
    from urllib.request import urlopen
    url='http://127.0.0.1:18082';diagnostic=root/'results/shared-mtp-sequence-diagnostic'
    golden_path=root/'results/shared-decoder-full-qualification/report.json';raw=golden_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=GOLDEN_SHA256:raise ValueError('Unqualified golden identity')
    golden=json.loads(raw);prior=json.loads((diagnostic/'report.json').read_text())
    if prior['status']!='diagnostic_passed':raise ValueError('Diagnostic did not complete')
    with urlopen(url+'/health',timeout=5) as response:health=json.load(response)
    validate_worker_profile(health,3,4,'direct',4,128,True)
    if health!=prior['health']:raise ValueError('Worker profile changed after diagnosis')
    records={};files=[golden_path,diagnostic/'report.json',root/'build/bin/garnet.dll',root/'build/bin/xlang3.exe',root/'models/Qwen3.8-27B-NVFP4/config.json']
    for key in KEYS:
        responses=[];bodies=[]
        for mode in ('target','mtp'):
            p=diagnostic/(key+'-'+mode+'-response.json');files.append(p);responses.append(json.loads(p.read_text()))
            p=diagnostic/(key+'-'+mode+'-request.json');files.append(p);body=json.loads(p.read_text());body.pop('study_target_only',None);bodies.append(body)
        if bodies[0]!=bodies[1]:raise ValueError('Target/MTP requests differ')
        records[key]=responses
        name=key if key in COLORS else key+'-greedy'
        matched=[row for row in golden['cases'] if row['key']==name]
        if matched:
            original=root/'results/native-mtp3-http'/(name+'-request.json');files.append(original)
            if hashlib.sha256(original.read_bytes()).hexdigest()!=matched[0]['request_sha256']:raise ValueError('Retained request identity changed')
            body=json.loads(original.read_text());body.pop('study_target_only',None)
            if body!=bodies[0]:raise ValueError('Golden request differs')
    gates=verify_records(golden,records)
    for index,case in enumerate(('text','image')):
        key='s3-'+case+'-c1-r0';body,_=payload(case,key,301000+index*20000,'qwen38-27b-nvfp4',128,'jpeg','image-first','binary-grid')
        stem='stream-gate-'+case;result=request(url,body,out,stem,128)
        if result['status']!='completed' or result['generated_tokens']!=128 or result['prompt_tokens']!=(41 if case=='text' else 104):raise ValueError('SSE/count gate failed')
        events=[json.loads(line)['data'] for line in (out/(stem+'-events.jsonl')).read_text().splitlines()]
        final=next(json.loads(value) for value in reversed(events) if value!='[DONE]')
        if final['response_token_ids']!=records[key][1]['choices'][0]['response_token_ids']:raise ValueError('Streamed full IDs differ')
    metadata={'source':source_identity(),'health':health,'same_profile_gates':gates,'golden_sha256':GOLDEN_SHA256,
        'file_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'scope':'B1 pilot admitted against pinned retained shared-profile goldens plus twenty exact target/MTP pairs and real SSE. Historical T16 gate rejection retained; no cross-profile bit equivalence, broad quality, full-context/concurrency qualification or 2x claim.'}
    path=out/'metadata.json';path.write_text(json.dumps(metadata,indent=2));print('SHARED_PROFILE_EXACT_GOLDENS_MTP_AND_STREAM_GATES_PASSED',flush=True)
    subprocess.run([sys.executable,str(root/'Garnet/test2026/cuda/qwen38_5090/benchmark_stream.py'),'--url',url,'--engine','garnet','--output',str(out/'benchmark'),'--metadata',str(path),'--concurrency','1','--requests','8','--warmups','2','--max-tokens','128','--workload-seed','3','--image-format','jpeg','--image-order','image-first','--image-nonce-pattern','binary-grid'],check=True)

if __name__=='__main__':main()
