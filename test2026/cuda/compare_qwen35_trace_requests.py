"""Verify real-input/output preservation around a diagnostic GPU capture."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('control',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
assert not a.output.exists(),'Preserve trace comparison'
rows=json.loads(a.trace.read_text());control=json.loads(a.control.read_text());assert len(rows)==len(control)==28
reference={(r['trial'],r['workload_group']):r for r in control};assert len(reference)==28
captured=[]
for i,row in enumerate(rows):
    old=reference[(row['trial'],row['workload_group'])]
    for field in ('input_sha256','prompt_token_ids','generated_token_ids','sampled_tokens_including_stop','finish_reasons','stop_token_ids','profile','cache_policy','speculative_metrics'):
        assert row[field]==old[field],('Trace request differs',i,field)
    assert row['gpu_trace_diagnostic']['request_index']==i
    if row['gpu_trace_diagnostic']['capture']:captured.append(i)
assert captured==[8]
result={'same_all28_input_output_stop_profile_speculative_counters':True,'captured_index':8,'captured_group':[rows[8]['trial'],rows[8]['workload_group']],
    'instrumented_model_request_ms':rows[8]['total_seconds']*1000,'historical_control_model_request_ms':reference[(2,0)]['total_seconds']*1000,
    'sources':{name:{'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in [('trace',a.trace),('control',a.control)]},
    'scope':'Full cohort preservation check and one instrumented request. Historical control; no interleaved overhead estimate, independent quality, or speed acceptance.'}
a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))
