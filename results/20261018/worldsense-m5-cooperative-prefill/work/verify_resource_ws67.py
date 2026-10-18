"""Recompute resource snapshot maxima and original thresholds after both workers stop."""
from pathlib import Path
import json,hashlib,datetime,ast,sys
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-cooperative-prefill';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-cooperative-prefill/source';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(s/k)==v for k,v in json.loads((w/'starting-source-manifest.json').read_text()).items())
assert all(sha(Path(k))==v for k,v in json.loads((w/'driver-identities-before-dispatch.json').read_text())['files'].items())
assert json.loads((w/'dependencies-after-resource.json').read_text())['all_aliases_and_file_bytes_unchanged']
analysis=json.loads((w/'resource-analysis.json').read_text());assert analysis['passed']and len(analysis['checks'])==8 and all(row['exact']and row['native_admission']for row in analysis['checks'])
fn=next(n for n in ast.parse((w/'analyze_resource.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='native_route');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'resource native admission','exec'),ns)
maxima={};records=[]
for label in ['control-1','candidate-1']:
 data=json.loads((w/(label+'.json')).read_text());assert data['valid']and data['assignments_restored']and len(data['items'])==2
 calls=[c for item in data['items']for c in item['result']['model_response'].values()if isinstance(c,dict)and c.get('status')=='ok'];assert len(calls)==4 and all(ns['native_route'](c,label=='candidate-1')for c in calls)
 maxima[label]={key:max(c['timings']['memory_after_decode'][key]for c in calls)for key in ['metal_allocated_bytes','task_physical_footprint_bytes']}
 records += [dict(label=label,model=c['model_id'],sole_after=c['timings']['sole_storage_after'],sole_delta=c['timings']['sole_encoded_delta'])for c in calls]
assert maxima==analysis['sampled_maximum_bytes'];gains={key:1-maxima['candidate-1'][key]/maxima['control-1'][key]for key in maxima['control-1']};assert gains==analysis['reduction']and gains['metal_allocated_bytes']>=.10 and gains['task_physical_footprint_bytes']>=.05
assert json.loads((w/'restoration-verification.json').read_text())['benchmark_valid_request_http404']
(w/'resource-independent-audit.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_terminal_session=int(sys.argv[1]),exit_code=0,original8_exact_native_calls_pass=True,source_drivers_dependencies_unchanged=True,recomputed_snapshot_maxima=maxima,recomputed_reductions=gains,actual_native_layout_records=records,original_resource_thresholds_pass=True,no_speed_or_promotion_claim=True),indent=2)+'\n');print(json.dumps(gains))
