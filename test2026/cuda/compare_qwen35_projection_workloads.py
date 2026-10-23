"""Audit all real requests, including discrete speculation work, across joining."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('control',type=Path);p.add_argument('candidate',type=Path);p.add_argument('output',type=Path)
p.add_argument('--linear-layers',type=int,required=True);a=p.parse_args()
assert not a.output.exists(),'Preserve workload evidence'
load=lambda path:{(row['trial'],row['workload_group']):row for row in json.loads(path.read_text(encoding='utf-8'))}
control=load(a.control);candidate=load(a.candidate)
assert len(control)==len(candidate)==28 and control.keys()==candidate.keys()
fields=('input_sha256','prompt_token_ids','generated_token_ids','sampled_tokens_including_stop',
    'finish_reasons','stop_token_ids','cache_policy','speculative','speculative_metrics','warmup')
differences=[]
for key,left in control.items():
    right=candidate[key]
    for field in fields:
        if left[field]!=right[field]:differences.append({'trial':key[0],'group':key[1],'field':field})
    lp=dict(left['profile']);rp=dict(right['profile'])
    if lp.pop('joinDecodeInputProjections',False) or rp.pop('joinDecodeInputProjections',False) is not True:
        differences.append({'trial':key[0],'group':key[1],'field':'joining transition'})
    if lp!=rp:differences.append({'trial':key[0],'group':key[1],'field':'other profile fields'})
    native=right['owned_memory']['decoder']
    if (right['io_memory']['joined_projection_group_count']!=a.linear_layers or
            sum(graph['joined_projection_inputs'] for graph in native['graphs'])!=a.linear_layers):
        differences.append({'trial':key[0],'group':key[1],'field':'native joining counters'})
    if left['owned_memory']['owned_model_bytes']!=right['owned_memory']['owned_model_bytes']:
        differences.append({'trial':key[0],'group':key[1],'field':'retained allocation'})
result={'sources':[{'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in (a.control,a.candidate)],
    'all_28_checked':True,'checked_fields':fields,'differences':differences,'all_passed':not differences,
    'scope':'All-request inputs, outputs, stops, cache declarations, discrete MTP calls/acceptance/iterations, native joining counters and retained allocation. This is workload preservation, not independent quality, full-state equivalence for every request, timing significance, physical memory or a cross-engine precision equivalence claim.'}
a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'all_passed':result['all_passed'],'differences':differences},indent=2))
assert result['all_passed'],'Joined workload changed; preserve differences'
