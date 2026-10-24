"""Replay actual saved SSE/requests before comparing HTTP benchmark cohorts.

This verifies accounting/workload evidence, not model quality or arithmetic equivalence.
"""
import argparse,hashlib,json,math
from pathlib import Path
from benchmark_stream import TokenEvents
from metrics import summarize

def verify_run(manifest,read):
    if manifest['status']!='completed':raise ValueError('Completed run required')
    verified={}
    for cohort in manifest['cohorts']:
        rows=[];requests={}
        for key in cohort['request_keys']:
            wire=read(key+'-request.json');body=json.loads(wire);saved=json.loads(read(key+'-result.json'))
            if saved['status']!='completed' or hashlib.sha256(wire).hexdigest()!=saved['request_sha256']:raise ValueError('Request digest/status mismatch: '+key)
            state=TokenEvents()
            for line in read(key+'-events.jsonl').decode().splitlines():
                event=json.loads(line);state.feed(event['data'],event['monotonic'])
            replay=state.result(saved['submitted'],saved['completed'],body['max_tokens'])
            for field in ('generated_tokens','prompt_tokens','first_token','last_token','text','finish_reason','first_arrival_batch_tokens','largest_arrival_batch_tokens','single_token_arrivals'):
                if replay[field]!=saved[field]:raise ValueError('Raw stream/result mismatch '+field+': '+key)
            rows.append(saved)
            workload_key=key.removeprefix(manifest['run_id']+'-')
            if workload_key in requests:raise ValueError('Duplicate workload key')
            requests[workload_key]={'sha256':saved['request_sha256'],'prompt_tokens':saved['prompt_tokens'],'generated_tokens':saved['generated_tokens']}
        recomputed=summarize(rows)
        for field,value in recomputed.items():
            if value is None:
                if cohort[field] is not None:raise ValueError('Summary mismatch: '+field)
            elif not math.isclose(value,cohort[field],rel_tol=1e-12,abs_tol=1e-9):raise ValueError('Summary mismatch: '+field)
        name=cohort['case']+'-c'+str(cohort['concurrency'])
        if name in verified:raise ValueError('Duplicate cohort')
        verified[name]={'metrics':recomputed,'requests':requests}
    return {'run_id':manifest['run_id'],'engine':manifest['engine'],'cohorts':verified,'scope':'Raw SSE counts/timestamps/text and request bytes verified; does not establish quality/arithmetic equivalence'}

def compare(native,references):
    result=[]
    for name,n in native['cohorts'].items():
        refs=[r['cohorts'][name] for r in references if name in r['cohorts']]
        same_requests=len(refs)==len(references) and bool(refs) and all(set(n['requests'])==set(r['requests']) and all(n['requests'][k]['sha256']==r['requests'][k]['sha256'] for k in n['requests']) for r in refs)
        same_lengths=same_requests and all(n['requests']==r['requests'] for r in refs)
        ratios={}
        if same_requests:
            for metric in ('decode_median_tokens_per_second','output_tokens_per_second'):
                values=[n['metrics'][metric]]+[r['metrics'][metric] for r in refs]
                ratios[metric]=values[0]/max(values[1:]) if all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in values) else None
            values=[n['metrics']['ttft_median_ms']]+[r['metrics']['ttft_median_ms'] for r in refs]
            ratios['ttft_vs_fastest_reference']=values[0]/min(values[1:]) if all(v>0 for v in values) else None
        result.append({'cohort':name,'identical_request_bytes':same_requests,'identical_prompt_output_counts':same_lengths,'ratios_to_best_supplied_reference':ratios})
    return {'cohorts':result,'has_vllm_and_sglang':{'vllm','sglang'}<={r['engine'] for r in references},'two_x_established':False,
        'scope':'Provisional arithmetic only. Quality/precision/repetition/tuned-reference gates are not proven by request/count matching; no automatic2x claim.'}

def load_directory(path):
    manifest=json.loads(path.read_text());return verify_run(manifest,lambda name:(path.parent/name).read_bytes())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--reference',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    native=load_directory(a.native);references=[load_directory(path) for path in a.reference]
    report={'native':native,'references':references,'comparison':compare(native,references)}
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report['comparison']))
