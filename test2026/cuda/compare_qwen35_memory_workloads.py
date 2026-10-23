"""Verify six diagnostic input groups and Garnet outputs against saved timing."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('garnet_session',type=Path);p.add_argument('lm_session',type=Path)
p.add_argument('garnet_cohort',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
assert not a.output.exists(),'Preserve memory workload evidence'
load=lambda path:json.loads(path.read_text(encoding='utf-8'))
garnet=load(a.garnet_session);lm=load(a.lm_session);cohort=load(a.garnet_cohort)
selected=[row for row in garnet if 'input_sha256' in row]
lm_hashes=[row.get('input_sha256',row.get('result',{}).get('input_sha256')) for row in lm]
lm_hashes=[value for value in lm_hashes if value is not None]
assert len(selected)==len(lm_hashes)==6
assert [row['input_sha256'] for row in selected]==lm_hashes,'Memory input groups differ'
timing={json.dumps(row['input_sha256'],sort_keys=True):row for row in cohort};assert len(timing)==28
for row in selected:
    reference=timing[json.dumps(row['input_sha256'],sort_keys=True)]
    assert row['profile']==reference['profile'],'Memory and timing profiles differ'
    assert row['output_signature']=={key:reference[key] for key in row['output_signature']},'Memory diagnostic output/stop differs'
paths=(a.garnet_session,a.lm_session,a.garnet_cohort)
result={'sources':[{'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
    'all_six_input_hash_groups_exact':True,'all_six_garnet_output_stop_signatures_match_timing':True,
    'same_garnet_profile':True,
    'scope':'Workload preservation for separate memory diagnostics. Cross-engine output equivalence, true peak memory, physical residency and latency are not established by this check.'}
a.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print('MEMORY_WORKLOAD_CONTROL_PASS')
