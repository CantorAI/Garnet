from pathlib import Path
import hashlib,json,sys,zipfile
r=Path('D:/CantorAI/5090-study-20261023');source=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(source))
from compare_http_runs import verify_run
from run_shared_mtp_candidate import KEYS,verify_records,GOLDEN_SHA256
from joined_model_gates import validate_joined_model_gates
from validate_native_mtp_http import validate_worker_profile
pilot=r/'results/native-mtp3-shared-joined-profile-pilot';diagnostic=r/'results/shared-joined-mlp-http-diagnostic';bench=pilot/'benchmark'
paths=list(bench.glob('*-manifest.json'));assert len(paths)==1;m=json.loads(paths[0].read_text());assert m['status']=='completed'
replay=verify_run(m,lambda name:(bench/name).read_bytes());assert len(replay['cohorts'])==2
golden=r/'results/shared-decoder-full-qualification/report.json';assert hashlib.sha256(golden.read_bytes()).hexdigest()==GOLDEN_SHA256
records={key:[json.loads((diagnostic/(key+'-'+mode+'-response.json')).read_text()) for mode in ('target','mtp')] for key in KEYS}
gates=verify_records(json.loads(golden.read_text()),records)
metadata=json.loads((pilot/'metadata.json').read_text());validate_worker_profile(metadata['health'],3,4,'direct',4,128,True,False,True)
dll=hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest();state=r/'results/joined-program-equivalence'
model_gates=validate_joined_model_gates(json.loads((state/'original-report.json').read_text()),json.loads((state/'joined-report.json').read_text()),json.loads((r/'results/joined-decoder-full-qualification/report.json').read_text()),json.loads(golden.read_text()),dll)
assert metadata['joined_model_gates']==model_gates
names=['native-mtp3-shared-joined-profile-pilot','shared-joined-mlp-http-diagnostic','native-mtp3-shared-joined-workflow','native-mtp3-shared-joined-workflow-ledger','native-mtp3-shared-joined-server-ledger','native-mtp3-shared-joined-worker']
for name in names:
 p=r/'results'/name/'commands.jsonl'
 if p.exists():assert json.loads(p.read_text().splitlines()[-1])['event']=='finished','Wait for original process terminal state'
stop=json.loads((r/'results/native-mtp3-shared-joined-intentional-stop.json').read_text());assert stop['taskkill_exit']==0
summary={'run_id':m['run_id'],'source':m['source'],'library_sha256':dll,'cohorts':m['cohorts'],'raw_event_replay':replay,'strict_gates':gates,'joined_model_gates':model_gates,'two_x_established':False,
 'scope':'One joined-MLP T4/T128 MTP3/draft4 direct-vision B1 short-prompt pilot. Full-model retained gates,20 exact self-target pairs,6 original goldens and real SSE. No causal speedup, repeated trial confidence, broader quality, tuned references, full context, native concurrency or2x qualification.'}
(r/'results/native-mtp3-shared-joined-summary.json').write_text(json.dumps(summary,indent=2))
files=[p for name in names for p in (r/'results'/name).rglob('*') if p.is_file()]
files += [r/name for name in ['stop-shared-joined-worker.py','stop-joined-after-pilot.py','archive-shared-joined-http.py','results/native-mtp3-shared-joined-summary.json','results/native-mtp3-shared-joined-intentional-stop.json','results/shared-decoder-full-qualification/report.json','results/joined-decoder-full-qualification/report.json','results/joined-program-equivalence/original-report.json','results/joined-program-equivalence/joined-report.json','results/joined-decoder-library-identity.json','results/joined-http-source-install.json','study-state.json','source-manifest.json','preparation-manifest.json','joined-http-source.zip','models/Qwen3.8-27B-NVFP4/config.json']]
for name in ['native_stream_worker.py','native_worker_profile.py','serve_native_mtp3_shared_joined_worker.bat','validate_native_mtp_http.py','run_shared_mtp_candidate.py','run_shared_joined_http.py','probe_shared_mtp_sequences.py','joined_model_gates.py']:files.append(source/name)
for name in ['xModel/qwen38/vl_27b/model.py','xModel/qwen35/vl_0_8b/cuda_speculative.py','xModel/qwen35/vl_0_8b/cuda_packed_replay.py','xModel/qwen35/vl_0_8b/cuda_packed_program.py']:files.append(r/'Garnet'/name)
for key in ('red','green','blue','bottom-blue','s3-text-c1-r0-greedy','s3-image-c1-r0-greedy'):files.append(r/'results/native-mtp3-http'/(key+'-request.json'))
raws={p.relative_to(r).as_posix():p.read_bytes() for p in files};out=r/'results/native-mtp3-shared-joined-evidence.zip';assert not out.exists()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for name,raw in raws.items():z.writestr(name,raw)
 z.writestr('evidence-manifest.json',json.dumps({'scope':summary['scope'],'run_id':m['run_id'],'full_model_archive_sha256':hashlib.sha256((r/'results/joined-full-evidence.zip').read_bytes()).hexdigest(),'files':{name:hashlib.sha256(raw).hexdigest() for name,raw in raws.items()}},indent=2))
print(json.dumps({'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'entries':len(raws)+1,'run_id':m['run_id'],'cohorts':m['cohorts']}))
