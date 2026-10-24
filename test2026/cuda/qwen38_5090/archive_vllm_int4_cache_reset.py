from pathlib import Path
import hashlib,json,sys,zipfile
r=Path('D:/CantorAI/5090-study-20261023');p=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(p));from compare_http_runs import verify_run
prefix='vllm-int4-cache-reset';names=[prefix+'-'+s for s in ('quality','http-pilot','quality-ledger','http-ledger','server-ledger','workflow','workflow-ledger')]
for s in ('quality','http','server','workflow'):
 rows=[json.loads(line) for line in (r/('results/'+prefix+'-'+s+'-ledger/commands.jsonl')).read_text().splitlines()];assert rows[-1]['event']=='finished' and rows[-1]['exit_code']==0
workflow=json.loads((r/('results/'+prefix+'-workflow/report.json')).read_text());assert workflow['status']=='completed' and workflow['exact_prior_same_engine_sequences']==16
pilot=r/('results/'+prefix+'-http-pilot');done=json.loads((pilot/'completed.json').read_text());assert done['status']=='completed' and done['cache_counter_delta']['hit_delta']==0 and done['cache_counter_delta']['query_delta']>0
reset=json.loads((pilot/'cache-control/reset.json').read_text());assert reset['status']=='reset_acknowledged' and [x['endpoint'] for x in reset['reset_responses']]==['/reset_mm_cache','/reset_encoder_cache'] and all(x['status']==200 for x in reset['reset_responses'])
stop=r/('results/'+prefix+'-intentional-stop.json');assert json.loads(stop.read_text())['selected_environment']['VLLM_SERVER_DEV_MODE']=='1'
paths=list((pilot/'benchmark').glob('*-manifest.json'));assert len(paths)==1;m=json.loads(paths[0].read_text());replay=verify_run(m,lambda n:(paths[0].parent/n).read_bytes());assert len(replay['cohorts'])==8
files=[f for n in names for f in (r/'results'/n).rglob('*') if f.is_file()]
files += [r/n for n in ['run-vllm-int4-cache-reset-workflow.py','stop-vllm-int4-cache-reset.py','archive-vllm-int4-cache-reset.py','results/'+prefix+'-server.log','results/'+prefix+'-intentional-stop.json','results/'+prefix+'-source-install.json','vllm-int4-cache-reset-source.zip','study-state.json','preparation-manifest.json','models/Qwen3.8-27B-INT4/config.json','models/Qwen3.8-27B-INT4/preprocessor_config.json','models/Qwen3.8-27B-INT4/model.safetensors.index.json','results/vllm-int4-quality/greedy/equivalence.json']]
files += [p/n for n in ['run_int4_reference_pilot.py','vllm_cache_control.py','serve_vllm_int4.sh','serve_vllm_int4_cache_reset.sh','run_reference_greedy_candidate.py','prepare_vlm_reference.py','reference_mtp_equivalence.py','benchmark_stream.py','metrics.py','sample_vram.py','compare_http_runs.py']]
for n in ('vllm-int4-mm-cache-source-audit','vllm-int4-mm-cache-router-audit'):files += [f for f in (r/'results'/n).rglob('*') if f.is_file()]
assert all(f.is_file() for f in files)
blobs={f.relative_to(r).as_posix():f.read_bytes() for f in files};out=r/'results/vllm-int4-cache-reset-evidence.zip';assert not out.exists()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in blobs.items():z.writestr(n,b)
 z.writestr('evidence-manifest.json',json.dumps({'files':{n:hashlib.sha256(b).hexdigest() for n,b in blobs.items()},'raw_event_replay':replay,'scope':'Explicit idle processor/transport and encoder cache reset acknowledged after exact ordinary quality repeat. Raw metrics/unique fixtures/post-run zero new processor hit gate retained. Not hardware residency, broad/cross-engine quality, fastest tuning or2x proof.'},indent=2))
print(json.dumps({'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'entries':len(blobs)+1,'cache_counter_delta':done['cache_counter_delta'],'cohorts':m['cohorts']}))
