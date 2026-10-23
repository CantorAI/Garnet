"""Coarse inclusive host diagnostic with exact saved-request preservation gate."""
import argparse,hashlib,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('diagnostic',type=Path);p.add_argument('control',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
assert not a.output.exists(),'Preserve diagnostic evidence'
rows=json.loads(a.diagnostic.read_text());control=json.loads(a.control.read_text())
assert len(rows)==len(control)==28
key=lambda r:(r['trial'],r['workload_group'])
reference={key(r):r for r in control};assert len(reference)==28
for row in rows:
    old=reference[key(row)]
    for field in ('input_sha256','prompt_token_ids','generated_token_ids','sampled_tokens_including_stop','finish_reasons','stop_token_ids','profile','cache_policy'):
        assert row[field]==old[field],('Changed request signature',field,key(row))
    assert row['cache_policy']=={'prefix':False,'image':False,'answer':False}
warm=[r for r in rows if not r['warmup']];assert len(warm)==20
med=statistics.median
categories=sorted({name for r in warm for name in r['host_time_diagnostic']['inclusive_categories']})
summary={}
for name in categories:
    records=[r['host_time_diagnostic']['inclusive_categories'].get(name,{'calls':0,'inclusive_wall_ns':0,'inclusive_thread_cpu_ns':0}) for r in warm]
    summary[name]={'median_calls':med(r['calls'] for r in records),
        'median_inclusive_wall_ms':med(r['inclusive_wall_ns'] for r in records)/1e6,
        'median_inclusive_thread_cpu_ms':med(r['inclusive_thread_cpu_ns'] for r in records)/1e6,
        'aggregate_thread_cpu_ms_per_call':sum(r['inclusive_thread_cpu_ns'] for r in records)/max(1,sum(r['calls'] for r in records))/1e6}
result={'sources':{name:{'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in [('diagnostic',a.diagnostic),('control',a.control)]},
    'same_all28_input_prompt_output_stop_profile_signatures':True,'warm_groups':20,
    'median_request_wall_ms':med(r['host_time_diagnostic']['request_wall_ns'] for r in warm)/1e6,
    'median_request_thread_cpu_ms':med(r['host_time_diagnostic']['request_thread_cpu_ns'] for r in warm)/1e6,
    'inclusive_categories_do_not_sum':summary,
    'scope':'Instrumented calling-thread CPU/wall time, overlapping categories; do not sum. Includes Python/native/driver activity and driver polling on this thread, excludes worker threads. Wall minus CPU is not GPU time. Clocks add overhead. No speedup, causality, independent quality, or 2x acceptance.'}
a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))
